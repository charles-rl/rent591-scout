#!/usr/bin/env python3
"""591-mcp-vision main pipeline / cron runner.

Loop: ingest -> dead-link filter -> DINOv3 dedup -> Qwen vision -> score -> notify -> store.

Usage:
  python main.py                        # live run (requires 591 network access)
  python main.py --incoming             # hybrid run over data/incoming/ (relay text always;
                                        # images + vision via PC devtunnel proxy when live)
  python main.py --fixtures --limit 3   # offline test using captured fixtures
  python main.py --train                # train XGBoost head from rated rows
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import (
    database,
    deduplication,
    dynamic_prompt,
    ingestion,
    notifier,
    scoring,
    vision_llm,
)
from src.utils import health_check, image_queue, proxy_check

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "3.5"))
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.95"))
PLACEHOLDER = os.environ.get("PLACEHOLDER_IMAGES", "").lower() in ("1", "true", "yes")
INCOMING_DIR = Path(os.environ.get("INCOMING_DIR", ROOT / "data" / "incoming"))
PROXY_URL = os.environ.get("PROXY_URL", "http://127.0.0.1:8999")

_ROW_JSON_FIELDS = ("tags", "contain_cost", "facilities", "image_urls", "image_paths",
                    "qwen_warnings", "qwen_vision_flags")


def _listing_from_row(row) -> dict:
    """DB row -> listing dict with JSON columns parsed back into Python objects."""
    listing = dict(row)
    for field in _ROW_JSON_FIELDS:
        raw = listing.get(field)
        if isinstance(raw, str):
            try:
                listing[field] = json.loads(raw)
            except ValueError:
                listing[field] = None
    return listing


def extract_text_warnings(listing: dict) -> list[str]:
    """Cheap rule pass over text metadata (floor / utilities / pricing rules).

    Runs at ingestion so listings queued without images already carry warnings;
    the Qwen pass later merges these with its own (deduped in finalize_listing).
    """
    warnings: list[str] = []
    floor = str(listing.get("floor") or "")
    if any(m in floor for m in ("頂樓", "顶楼", "頂層")):
        warnings.append("頂樓：可能炎熱/漏水")
    elif any(m in floor for m in ("一樓", "1樓")):
        warnings.append("一樓：注意採光與隱私")
    blob = str(listing.get("description") or "")
    if any(m in blob for m in ("水電另計", "電費另計", "水費另計", "代管理費", "管理費另")):
        warnings.append("水電/管理費另計")
    deposit = str(listing.get("deposit") or "")
    if "半年" in deposit:
        warnings.append("要求半年付")
    elif "季" in deposit:
        warnings.append("要求季付")
    elif "年" in deposit and "一年" not in deposit:
        warnings.append(f"付款規則：{deposit}")
    return warnings


def load_baseline(conn) -> dict[str, list]:
    """Stored per-image embeddings keyed by listing_id."""
    baseline: dict[str, list] = {}
    for lid, rows in database.get_all_images(conn).items():
        vecs = []
        for r in rows:
            if r["dino_embedding"]:
                vecs.append(np.frombuffer(r["dino_embedding"], dtype=np.float32))
        baseline[lid] = vecs
    return baseline


def process_listing(conn, entry: dict, baseline: dict, bullets, do_notify: bool) -> str:
    """One live-mode listing through the pipeline. Returns outcome: stored|inactive|duplicate|failed.

    Direct 591 access (needs network to rent.591.com.tw); the GitHub-relay
    offline/hybrid path lives in run_incoming().
    """
    item, data = entry["raw_search"], entry["raw_metadata"]
    detail_failed = entry.get("detail_failed", False)
    listing = ingestion.normalize_listing(item, data, detail_failed=detail_failed)

    if not listing["is_active"]:
        logger.info("%s delisted/inactive -> skip", listing["listing_id"])
        listing["is_active"] = False
        listing["image_status"] = "skipped"
        database.upsert_listing(conn, listing)
        return "inactive"

    # Health check: live DOM probe (591 reachable).
    if not health_check.is_listing_active(listing["url"]):
        logger.info("%s health check: dead/expired listing -> is_active=0", listing["listing_id"])
        health_check.mark_listing_inactive(conn, listing["listing_id"])
        return "inactive"

    # 591scraper DOM fallback enrichment (best-effort; needs browser + 591 access).
    scraper = ingestion.scraper_fallback(listing["listing_id"])
    if scraper:
        listing = ingestion.apply_scraper(listing, scraper)
    urls = ingestion.fetch_image_urls(item, data)
    image_rows = ingestion.download_images(listing["listing_id"], urls, placeholder=PLACEHOLDER)

    listing["image_urls"] = urls
    listing["image_paths"] = [r["image_path"] for r in image_rows]
    return finalize_listing(conn, listing, image_rows, baseline, bullets, do_notify)


def finalize_listing(conn, listing: dict, image_rows: list[dict], baseline: dict,
                     bullets, do_notify: bool, proxy: str | None = None) -> str:
    """DINOv3 dedup -> Qwen 27B vision -> XGBoost score -> store -> notify.

    Shared tail of the live path and the hybrid image-queue path.
    Returns outcome: stored|duplicate|failed.
    """
    # DINOv3 embeddings + dedup (exclude this listing's own stored vectors).
    new_vecs = deduplication.embed_image_rows(image_rows)
    for r in image_rows:
        r["dino_embedding"] = new_vecs[r["ordinal"]].tobytes() if r["ordinal"] in new_vecs else None
    own_baseline = {lid: v for lid, v in baseline.items() if lid != listing["listing_id"]}
    dup, matched = deduplication.find_duplicate(list(new_vecs.values()), own_baseline, DEDUP_THRESHOLD)
    if dup:
        # Persist visual fingerprint + is_duplicate flag, bypass the Qwen vision pass.
        logger.info("%s duplicate of %s -> is_duplicate=1, skip vision LLM", listing["listing_id"], matched)
        listing["is_duplicate"] = True
        listing["dino_embedding"] = deduplication.aggregate_embedding(list(new_vecs.values()))
        database.upsert_listing(conn, listing)
        database.replace_images(conn, listing["listing_id"], image_rows)
        baseline[listing["listing_id"]] = list(new_vecs.values())
        return "duplicate"

    # Qwen vision + text.
    analysis = vision_llm.analyze_listing(listing, image_rows, bullets)
    if analysis is None:
        logger.warning("%s vision analysis failed -> skip", listing["listing_id"])
        return "failed"
    # Merge ingestion-time rule warnings (floor/utilities/pricing) with VLM ones.
    text_warnings = [str(w) for w in (listing.get("qwen_warnings") or [])]
    listing["qwen_warnings"] = list(dict.fromkeys(text_warnings + list(analysis["qwen_warnings"])))
    listing["qwen_vision_flags"] = analysis["vision_flags"]
    listing["qwen_direct_score"] = analysis["qwen_direct_score"]

    # Scoring. XGBoost was trained on the aggregate embedding — infer on the same feature.
    agg = deduplication.aggregate_embedding(list(new_vecs.values()))
    listing["dino_embedding"] = agg
    dino_vec = np.frombuffer(agg, dtype=np.float32) if agg else np.zeros(768, dtype=np.float32)
    predicted, source = scoring.predict_score(
        conn, dino_vec, analysis["vision_flags"], listing["qwen_warnings"],
        analysis["qwen_direct_score"],
    )
    listing["predicted_score"] = predicted
    listing["score_source"] = source

    # Store (per-image vectors already set above). Explicit False clears any
    # stale flag if this listing's images changed since a previous duplicate run.
    listing["is_duplicate"] = False
    database.upsert_listing(conn, listing)
    database.replace_images(conn, listing["listing_id"], image_rows)

    logger.info(
        "%s score=%.2f (%s) warnings=%d",
        listing["listing_id"], predicted, source, len(listing["qwen_warnings"]),
    )

    # Notify.
    if do_notify:
        notifier.send_ntfy_alert(listing, predicted, SCORE_THRESHOLD, proxy=proxy)
    return "stored"


def run(fixtures: bool, limit: int, do_notify: bool) -> int:
    conn = None
    failed = 0
    processed = 0
    try:
        conn = database.connect()
        bullets = dynamic_prompt.get_bullets(conn)

        listings = ingestion.fetch_raw_listings(fixtures=fixtures, limit=limit)
        logger.info("fetched %s listings", len(listings))
        baseline = load_baseline(conn)

        for entry in listings:
            try:
                outcome = process_listing(conn, entry, baseline, bullets, do_notify)
                processed += outcome == "stored"
                failed += outcome == "failed"
            except Exception:
                logger.exception("listing %s failed", (entry.get("raw_search") or {}).get("id"))
                failed += 1
                continue
    except Exception:
        logger.exception("fatal pipeline error")
        return 1
    finally:
        if conn is not None:
            conn.close()
    logger.info("done: %s new listings processed, %s failed", processed, failed)
    return 1 if failed else 0


def run_incoming(incoming_dir: Path, limit: int, do_notify: bool | None) -> int:
    """Hybrid pipeline over data/incoming/ payloads pushed by the GitHub relay.

    Phase 1 (runs always): text-only ingestion — store metadata with
    image_status='pending' plus rule-based text warnings; no network at all.
    Phase 2: probe the PC devtunnel proxy (PROXY_URL).
      LIVE    -> drain the pending image queue through it, then run
                 DINOv3 dedup + Qwen vision + XGBoost scoring per listing and
                 push match alerts (score >= 3.5) via the tunnel.
      OFFLINE -> one ntfy "connect the PC proxy" alert per un-notified pending
                 batch (text_only_notified prevents spam).
    """
    listings_dir = incoming_dir / "listings"
    if not listings_dir.is_dir():
        logger.error("incoming dir not found: %s (run the relay or `git pull` first)", listings_dir)
        return 1
    conn = None
    counters = {"ingested": 0, "inactive": 0, "skipped": 0, "stored": 0,
                "duplicate": 0, "failed": 0, "queued": 0}
    vision_targets: dict[str, list[dict]] = {}
    try:
        conn = database.connect()
        bullets = dynamic_prompt.get_bullets(conn)

        # ---- Phase 1: text ingestion (always) --------------------------------
        paths = sorted(listings_dir.glob("*.json"))
        logger.info("incoming: %s payload files", len(paths))
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("unreadable payload %s", path)
                counters["failed"] += 1
                continue
            lid = str(payload.get("listing_id") or "")
            sha = payload.get("payload_sha256")
            if not lid:
                counters["failed"] += 1
                continue
            if database.relay_is_processed(conn, lid, sha):
                counters["skipped"] += 1
                continue
            try:
                listing = ingestion.normalize_listing(
                    payload.get("raw_search"), payload.get("raw_metadata"),
                    detail_failed=payload.get("detail_failed", False),
                )
                if not listing["is_active"]:
                    logger.info("%s delisted/inactive -> skip", lid)
                    listing["image_status"] = "skipped"
                    database.upsert_listing(conn, listing)
                    counters["inactive"] += 1
                elif not health_check.is_listing_active({"status": listing.get("status")}):
                    logger.info("%s health check: dead/expired -> is_active=0", lid)
                    health_check.mark_listing_inactive(conn, lid)
                    counters["inactive"] += 1
                else:
                    listing["qwen_warnings"] = extract_text_warnings(listing)
                    urls = payload.get("image_urls") or []
                    listing["image_urls"] = urls
                    rows = []
                    for ordinal, url in enumerate(urls):
                        p = incoming_dir / "images" / lid / f"{ordinal:02d}.webp"
                        rows.append({"ordinal": ordinal, "image_url": url,
                                     "image_path": str(p) if ingestion._valid_webp(p) else None})
                    if not urls:
                        listing["image_status"] = "skipped"
                        vision_targets[lid] = []
                    elif all(r["image_path"] for r in rows):
                        listing["image_status"] = "completed"
                        listing["image_paths"] = [r["image_path"] for r in rows]
                        vision_targets[lid] = rows
                    else:
                        listing["image_status"] = "pending"
                        counters["queued"] += 1
                    database.upsert_listing(conn, listing)
                    database.reset_text_only_notified(conn, lid)
                    if listing["image_status"] == "completed":
                        database.replace_images(conn, lid, rows)
                database.mark_relay_processed(conn, lid, sha)
                counters["ingested"] += 1
            except Exception:
                logger.exception("incoming listing %s failed", lid)
                counters["failed"] += 1
                continue
            if 0 < limit <= counters["ingested"] + counters["failed"]:
                break

        # ---- Phase 2: proxy check & branching ---------------------------------
        proxy_live = proxy_check.is_proxy_available(PROXY_URL)
        effective_notify = bool(proxy_live) if do_notify is None else do_notify
        logger.info("proxy %s: %s", PROXY_URL, "LIVE" if proxy_live else "OFFLINE")
        if proxy_live:
            for lid, rows in image_queue.process_pending_images(conn, PROXY_URL):
                vision_targets[lid] = rows
        else:
            pending_alerts = database.count_pending_unnotified(conn)
            if pending_alerts:
                logger.info("%d pending listings -> proxy request alert", pending_alerts)
                # Tunnel-first: the probe may fail on 591 while the devtunnel
                # itself still relays ntfy fine; _post falls back to direct.
                notifier.send_proxy_request_alert(pending_alerts, proxy=PROXY_URL)
                database.mark_text_only_notified(conn)

        # Self-heal: images completed earlier whose vision pass never ran.
        for row in database.get_completed_unscored(conn):
            vision_targets.setdefault(str(row["listing_id"]), [])

        if vision_targets:
            baseline = load_baseline(conn)
            for lid, rows in vision_targets.items():
                row = conn.execute("SELECT * FROM listings WHERE listing_id=?", (lid,)).fetchone()
                if row is None:
                    continue
                try:
                    outcome = finalize_listing(
                        conn, _listing_from_row(row), rows, baseline, bullets,
                        effective_notify, proxy=PROXY_URL if proxy_live else None,
                    )
                    counters[outcome] += 1
                except Exception:
                    logger.exception("vision pass failed for %s", lid)
                    counters["failed"] += 1
    except Exception:
        logger.exception("fatal incoming pipeline error")
        return 1
    finally:
        if conn is not None:
            conn.close()
    logger.info("incoming done: %s", counters)
    return 1 if counters["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="591-mcp-vision pipeline")
    parser.add_argument("--fixtures", action="store_true", help="offline mode using captured fixtures")
    parser.add_argument("--incoming", action="store_true",
                        help="offline mode over data/incoming/ (GitHub Actions relay payloads)")
    parser.add_argument("--incoming-dir", type=Path, default=INCOMING_DIR,
                        help="relay payload directory (default: data/incoming)")
    parser.add_argument("--limit", type=int, default=-1, help="max listings to process")
    parser.add_argument("--notify", dest="notify", action="store_true", default=None)
    parser.add_argument("--no-notify", dest="notify", action="store_false")
    parser.add_argument("--train", action="store_true", help="train XGBoost head from rated rows")
    args = parser.parse_args()

    if args.train:
        conn = database.connect()
        try:
            scoring.train_and_save(conn)
            backfilled = scoring.score_all_unrated(conn)
            logger.info("--train: backfilled predicted_score for %d unrated listings", backfilled)
        except RuntimeError as e:
            logger.error("--train aborted: %s", e)
            sys.exit(1)
        finally:
            conn.close()
        return

    if args.incoming:
        # Hybrid: do_notify=None -> alerts auto-enable while the PC proxy is live
        # (ntfy.sh is blocked from the GPU server directly; match alerts route
        # through the devtunnel).
        sys.exit(run_incoming(args.incoming_dir, args.limit, args.notify))
    do_notify = args.notify if args.notify is not None else True
    run(fixtures=args.fixtures, limit=args.limit, do_notify=do_notify)


if __name__ == "__main__":
    main()
