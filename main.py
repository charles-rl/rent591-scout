#!/usr/bin/env python3
"""591-mcp-vision main pipeline / cron runner.

Loop: ingest -> dead-link filter -> DINOv2 dedup -> Qwen vision -> score -> notify -> store.

Usage:
  python main.py                        # live run (requires 591 network access)
  python main.py --fixtures --limit 3   # offline test using captured fixtures
  python main.py --train                # train XGBoost head from rated rows
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import database, deduplication, ingestion, notifier, scoring, vision_llm  # noqa: E402

SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "3.5"))
DEDUP_THRESHOLD = float(os.environ.get("DEDUP_THRESHOLD", "0.95"))
PLACEHOLDER = os.environ.get("PLACEHOLDER_IMAGES", "").lower() in ("1", "true", "yes")


def run(fixtures: bool, limit: int, do_notify: bool) -> int:
    conn = database.connect()
    bullets = database.get_latest_preferences(conn)
    processed = 0

    listings = ingestion.fetch_raw_listings(fixtures=fixtures, limit=limit)
    print(f"[main] fetched {len(listings)} listings")

    # Baseline of stored per-image embeddings for dedup.
    baseline_raw = database.get_all_images(conn)
    baseline: dict[str, list] = {}
    for lid, rows in baseline_raw.items():
        vecs = []
        for r in rows:
            if r["dino_embedding"]:
                import numpy as np
                vecs.append(np.frombuffer(r["dino_embedding"], dtype=np.float32))
        baseline[lid] = vecs

    for entry in listings:
        item, data = entry["raw_search"], entry["raw_metadata"]
        listing = ingestion.normalize_listing(item, data)

        if not listing["is_active"]:
            print(f"[main] {listing['listing_id']} delisted/inactive -> skip")
            listing["is_active"] = False
            database.upsert_listing(conn, listing)
            continue

        # 591scraper DOM fallback enrichment (best-effort).
        scraper = ingestion.scraper_fallback(listing["listing_id"])
        if scraper:
            listing = ingestion.apply_scraper(listing, scraper)

        # Images -> WebP.
        urls = ingestion.fetch_image_urls(item, data)
        image_rows = ingestion.download_images(listing["listing_id"], urls, placeholder=PLACEHOLDER)
        listing["image_urls"] = urls
        listing["image_paths"] = [r["image_path"] for r in image_rows]

        # DINOv2 embeddings + dedup.
        new_vecs = deduplication.embed_image_rows(image_rows)
        dup, matched = deduplication.find_duplicate(list(new_vecs.values()), baseline, DEDUP_THRESHOLD)
        if dup:
            print(f"[main] {listing['listing_id']} duplicate of {matched} -> skip")
            continue

        # Qwen vision + text.
        analysis = vision_llm.analyze_listing(listing, image_rows, bullets)
        if analysis is None:
            print(f"[main] {listing['listing_id']} vision analysis failed -> skip")
            continue
        listing["qwen_warnings"] = analysis["qwen_warnings"]
        listing["qwen_vision_flags"] = analysis["vision_flags"]
        listing["qwen_direct_score"] = analysis["qwen_direct_score"]

        # Scoring.
        agg = deduplication.aggregate_embedding(list(new_vecs.values()))
        listing["dino_embedding"] = agg
        dino_vec = next(iter(new_vecs.values())) if new_vecs else __import__("numpy").zeros(768)
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
        processed += 1

        print(f"[main] {listing['listing_id']} score={predicted:.2f} ({source}) "
              f"warnings={len(listing['qwen_warnings'])}")

        # Notify.
        if do_notify:
            notifier.send_ntfy_alert(listing, predicted, SCORE_THRESHOLD)

    conn.close()
    print(f"[main] done: {processed} new listings processed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="591-mcp-vision pipeline")
    parser.add_argument("--fixtures", action="store_true", help="offline mode using captured fixtures")
    parser.add_argument("--limit", type=int, default=-1, help="max listings to process")
    parser.add_argument("--notify", dest="notify", action="store_true", default=True)
    parser.add_argument("--no-notify", dest="notify", action="store_false")
    parser.add_argument("--train", action="store_true", help="train XGBoost head from rated rows")
    args = parser.parse_args()

    if args.train:
        conn = database.connect()
        scoring.train_and_save(conn)
        conn.close()
        return

    run(fixtures=args.fixtures, limit=args.limit, do_notify=args.notify)


if __name__ == "__main__":
    sys.exit(main())
