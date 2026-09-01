#!/usr/bin/env python3
"""591-mcp-vision main pipeline / cron runner.

Loop: ingest -> dead-link filter -> DINOv3 dedup -> Qwen vision -> score -> notify -> store.

Usage:
  python main.py                        # live run (requires 591 network access)
  python main.py --incoming             # offline run over data/incoming/ (git relay payloads)
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

from src import database, deduplication, ingestion, notifier, scoring, vision_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "3.5"))
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.95"))
PLACEHOLDER = os.environ.get("PLACEHOLDER_IMAGES", "").lower() in ("1", "true", "yes")
INCOMING_DIR = Path(os.environ.get("INCOMING_DIR", ROOT / "data" / "incoming"))


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


def process_listing(conn, entry: dict, baseline: dict, bullets, do_notify: bool,
                    incoming_dir: Path | None = None) -> str:
    """One listing through the pipeline. Returns outcome: stored|inactive|duplicate|failed.

    incoming_dir: when set, the payload came from the GitHub relay — images are
    already on local disk and no 591 HTTP call (scraper fallback / CDN) is made.
    """
    item, data = entry["raw_search"], entry["raw_metadata"]
    detail_failed = entry.get("detail_failed", False)
    listing = ingestion.normalize_listing(item, data, detail_failed=detail_failed)

    if not listing["is_active"]:
        logger.info("%s delisted/inactive -> skip", listing["listing_id"])
        listing["is_active"] = False
        database.upsert_listing(conn, listing)
        return "inactive"

    if incoming_dir is None:
        # 591scraper DOM fallback enrichment (best-effort; needs browser + 591 access).
        scraper = ingestion.scraper_fallback(listing["listing_id"])
        if scraper:
            listing = ingestion.apply_scraper(listing, scraper)
        urls = ingestion.fetch_image_urls(item, data)
        image_rows = ingestion.download_images(listing["listing_id"], urls, placeholder=PLACEHOLDER)
    else:
        urls = entry.get("image_urls") or []
        image_rows = []
        for ordinal, url in enumerate(urls):
            p = incoming_dir / "images" / listing["listing_id"] / f"{ordinal:02d}.webp"
            image_rows.append({
                "ordinal": ordinal, "image_url": url,
                "image_path": str(p) if p.is_file() else None,
            })

    listing["image_urls"] = urls
    listing["image_paths"] = [r["image_path"] for r in image_rows]

    # DINOv3 embeddings + dedup (exclude this listing's own stored vectors).
    new_vecs = deduplication.embed_image_rows(image_rows)
    own_baseline = {lid: v for lid, v in baseline.items() if lid != listing["listing_id"]}
    dup, matched = deduplication.find_duplicate(list(new_vecs.values()), own_baseline, DEDUP_THRESHOLD)
    if dup:
        logger.info("%s duplicate of %s -> skip", listing["listing_id"], matched)
        return "duplicate"

    # Qwen vision + text.
    analysis = vision_llm.analyze_listing(listing, image_rows, bullets)
    if analysis is None:
        logger.warning("%s vision analysis failed -> skip", listing["listing_id"])
        return "failed"
    listing["qwen_warnings"] = analysis["qwen_warnings"]
    listing["qwen_vision_flags"] = analysis["vision_flags"]
    listing["qwen_direct_score"] = analysis["qwen_direct_score"]

    # Scoring.
    agg = deduplication.aggregate_embedding(list(new_vecs.values()))
    listing["dino_embedding"] = agg
    dino_vec = next(iter(new_vecs.values())) if new_vecs else np.zeros(768, dtype=np.float32)
    predicted, source = scoring.predict_score(
        conn, dino_vec, analysis["vision_flags"], analysis["qwen_warnings"],
        analysis["qwen_direct_score"],
    )
    listing["predicted_score"] = predicted
    listing["score_source"] = source

    # Store (with per-image vectors).
    for r in image_rows:
        r["dino_embedding"] = new_vecs.get(r["ordinal"]).tobytes() if r["ordinal"] in new_vecs else None
    database.upsert_listing(conn, listing)
    database.replace_images(conn, listing["listing_id"], image_rows)

    logger.info(
        "%s score=%.2f (%s) warnings=%d",
        listing["listing_id"], predicted, source, len(listing["qwen_warnings"]),
    )

    # Notify.
    if do_notify:
        notifier.send_ntfy_alert(listing, predicted, SCORE_THRESHOLD)
    return "stored"


def run(fixtures: bool, limit: int, do_notify: bool) -> int:
    conn = None
    failed = 0
    processed = 0
    try:
        conn = database.connect()
        bullets = database.get_latest_preferences(conn)

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


def run_incoming(incoming_dir: Path, limit: int, do_notify: bool) -> int:
    """Offline pipeline over data/incoming/ payloads pushed by the GitHub relay.

    Strictly local: no HTTP to 591 domains, no CDN downloads, no DOM scraper.
    Only localhost inference (Ollama, DINOv3 cache) and the local SQLite DB.
    """
    listings_dir = incoming_dir / "listings"
    if not listings_dir.is_dir():
        logger.error("incoming dir not found: %s (run the relay or `git pull` first)", listings_dir)
        return 1
    conn = None
    counters = {"stored": 0, "inactive": 0, "duplicate": 0, "failed": 0, "skipped": 0}
    try:
        conn = database.connect()
        bullets = database.get_latest_preferences(conn)
        baseline = load_baseline(conn)

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
            entry = {
                "raw_search": payload.get("raw_search"),
                "raw_metadata": payload.get("raw_metadata"),
                "detail_failed": payload.get("detail_failed", False),
                "image_urls": payload.get("image_urls") or [],
            }
            try:
                outcome = process_listing(conn, entry, baseline, bullets, do_notify,
                                          incoming_dir=incoming_dir)
            except Exception:
                logger.exception("incoming listing %s failed", lid)
                counters["failed"] += 1
                continue
            counters[outcome] += 1
            if outcome in ("stored", "inactive", "duplicate"):
                database.mark_relay_processed(conn, lid, sha)
            if 0 < limit <= sum(counters.values()):
                break
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
        scoring.train_and_save(conn)
        conn.close()
        return

    # ntfy.sh is blocked from the GPU server; --incoming defaults to no notifications.
    do_notify = args.notify if args.notify is not None else not args.incoming

    if args.incoming:
        sys.exit(run_incoming(args.incoming_dir, args.limit, do_notify))
    run(fixtures=args.fixtures, limit=args.limit, do_notify=do_notify)


if __name__ == "__main__":
    main()
