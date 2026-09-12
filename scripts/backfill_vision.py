"""Re-run the Qwen vision pass over stored listings with images on disk.

Use after prompt changes (e.g. the desk-priority bullets) to refresh
qwen_direct_score / qwen_vision_flags / qwen_warnings / predicted_score for
every non-duplicate listing whose photos are available locally.

Reuses main.finalize_listing (dedup -> Qwen vision -> XGBoost score -> store)
so the stored pipeline stays the single source of truth. Bathroom photo labels
(loading is_bathroom) are NOT re-run: they are already persisted and only feed
the probe estimate, which is computed locally. No ntfy alerts are sent.

Cost: one full vision call per listing (~200s with 8 photos on this box);
~1 day for ~500 listings. Ollama must stay up; a down endpoint is fail-soft
(listing is skipped) and the run is resumable.

Usage:
  .venv/bin/python scripts/backfill_vision.py            # everything not yet done
  .venv/bin/python scripts/backfill_vision.py --limit 5  # staged run
  .venv/bin/python scripts/backfill_vision.py --reset    # clear progress, start over
Progress file: data/backfill_vision_progress.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import main
from src import database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_vision")

PROGRESS_PATH = ROOT / "data" / "backfill_vision_progress.json"


def load_progress() -> set[str]:
    if PROGRESS_PATH.is_file():
        try:
            return set(json.loads(PROGRESS_PATH.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            logger.warning("unreadable progress file, starting fresh")
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(sorted(done)), encoding="utf-8")


def select_listings(conn, done: set[str]) -> list[tuple[str, list[dict]]]:
    """(listing_id, image rows with real on-disk paths) for remaining targets."""
    rows = conn.execute(
        "SELECT listing_id FROM listings "
        "WHERE image_status='completed' AND IFNULL(is_duplicate, 0) = 0"
    ).fetchall()
    all_images = database.get_all_images(conn)
    targets: list[tuple[str, list[dict]]] = []
    for row in rows:
        lid = str(row["listing_id"])
        if lid in done:
            continue
        imgs = [r for r in all_images.get(lid, [])
                if r.get("image_path") and Path(r["image_path"]).is_file()
                and Path(r["image_path"]).stat().st_size > 4096]
        if imgs:
            targets.append((lid, imgs))
    return targets


def run() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0, help="max listings to process this run (0 = all)")
    ap.add_argument("--reset", action="store_true", help="clear the progress file before running")
    args = ap.parse_args()

    if args.reset:
        PROGRESS_PATH.unlink(missing_ok=True)
        logger.info("progress file cleared")

    done = load_progress()
    conn = database.connect()
    try:
        # Bathroom labels are persisted; skip the extra Qwen detection pass.
        main.bathroom_detect.detect_flags = lambda paths, chunk=None: None  # type: ignore[method-assign]
        targets = select_listings(conn, done)
        if args.limit > 0:
            targets = targets[: args.limit]
        logger.info("%d listings to process (%d already done)", len(targets), len(done))
        if not targets:
            return 0

        bullets = main.dynamic_prompt.get_bullets(conn)
        baseline = main.load_baseline(conn)
        counts = {"stored": 0, "duplicate": 0, "failed": 0}
        t0 = time.time()
        for i, (lid, imgs) in enumerate(targets, 1):
            row = conn.execute("SELECT * FROM listings WHERE listing_id=?", (lid,)).fetchone()
            if row is None:
                continue
            listing = main._listing_from_row(row)
            # Do not let the rerun wipe the stored per-image bathroom labels
            # (replace_images would overwrite them with the row values here).
            stored_flags = {
                r["ordinal"]: r["is_bathroom"]
                for r in conn.execute(
                    "SELECT ordinal, is_bathroom FROM listing_images WHERE listing_id=?", (lid,)
                )
            }
            for img in imgs:
                img["is_bathroom"] = stored_flags.get(img.get("ordinal"))
            try:
                outcome = main.finalize_listing(conn, listing, imgs, baseline, bullets, False)
                counts[outcome] = counts.get(outcome, 0) + 1
                done.add(lid)
            except Exception:
                logger.exception("vision pass failed for %s", lid)
                counts["failed"] += 1
                continue
            save_progress(done)
            if i % 5 == 0 or i == len(targets):
                eta = (time.time() - t0) / i * (len(targets) - i) / 3600
                logger.info(
                    "[%d/%d] %s: stored=%d duplicate=%d failed=%d | %.1fh remaining at this pace",
                    i, len(targets), lid, counts["stored"], counts["duplicate"], counts["failed"], eta,
                )
    finally:
        conn.close()
    logger.info("backfill done: %s", counts)
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(run())
