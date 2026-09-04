"""Label bathroom photos + train/refresh the bathroom probe.

For every bathroom-rated listing (listings.bathroom_score NOT NULL), asks Qwen to
flag which photos show the bathroom (only for photos not labelled yet), persists
listing_images.is_bathroom, retrains the probe and reports leave-one-out quality.

With --rescore, additionally writes listings.bath_model_score for unrated listings
that already have DINO embeddings (centroid selection, no extra LLM calls).

Usage: .venv/bin/python scripts/backfill_bathrooms.py [--rescore]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

from src import bathroom_detect, bathroom_probe, database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_bathrooms")


def label_photos(conn) -> int:
    """Detect + persist is_bathroom for photos of bathroom-rated listings. Returns photos labelled."""
    rows = conn.execute(
        "SELECT listing_id, bathroom_score FROM listings WHERE bathroom_score IS NOT NULL"
    ).fetchall()
    total = 0
    for lid, _score in rows:
        imgs = conn.execute(
            "SELECT ordinal, image_path, is_bathroom FROM listing_images "
            "WHERE listing_id=? AND dino_embedding IS NOT NULL ORDER BY ordinal", (lid,)
        ).fetchall()
        todo = [i for i in imgs if i["is_bathroom"] is None and i["image_path"] and Path(i["image_path"]).is_file()]
        if not todo:
            continue
        flags = bathroom_detect.detect_flags([i["image_path"] for i in todo])
        if not flags:
            logger.warning("%s: detection failed, leaving %d photos unlabeled", lid, len(todo))
            continue
        database.set_bath_image_flags(conn, lid, {todo[i]["ordinal"]: v for i, v in flags.items()})
        total += len(flags)
        logger.info("%s: labelled %d photos (%d bathroom)", lid, len(flags), sum(flags.values()))
    return total


def loo_mae(conn) -> tuple[int, float, float]:
    """Leave-one-out MAE of the ridge probe on pooled bathroom vectors (1-5 scale)."""
    X, y, n = bathroom_probe.training_data(conn)
    if n < 3:
        return n, float("nan"), float("nan")
    alpha = bathroom_probe.alpha_for(n)
    errs = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        w, b = bathroom_probe._ridge_fit(X[m], y[m], alpha)
        errs.append(abs((float(np.clip(X[i] @ w + b, 0, 1)) * 4 + 1) - (y[i] * 4 + 1)))
    return n, float(np.mean(errs)), alpha


def rescore(conn) -> int:
    """Write bath_model_score for unrated listings with stored per-photo embeddings."""
    centroid = bathroom_probe.bath_centroid(conn)
    if centroid is None:
        logger.warning("no bathroom centroid yet; skipping rescore")
        return 0
    rows = conn.execute(
        "SELECT listing_id FROM listings WHERE IFNULL(user_rated,0)=0 AND image_status='completed'"
    ).fetchall()
    updates = []
    for (lid,) in rows:
        vecs = [r["dino_embedding"] for r in conn.execute(
            "SELECT dino_embedding FROM listing_images WHERE listing_id=? AND dino_embedding IS NOT NULL "
            "ORDER BY ordinal", (lid,))]
        if not vecs:
            continue
        updates.append((lid, bathroom_probe.predict_listing(conn, vecs, centroid)))
    conn.executemany("UPDATE listings SET bath_model_score=? WHERE listing_id=?", updates)
    conn.commit()
    return len(updates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="score unrated listings too (no LLM calls)")
    args = ap.parse_args()

    conn = database.connect(ROOT / "data" / "apartments.db")

    labelled = label_photos(conn)
    logger.info("photo labelling done: %d photos", labelled)

    n_before = len({r["listing_id"] for r in database.get_bath_rated_samples(conn)})
    probe = bathroom_probe.load_probe()
    if n_before >= bathroom_probe.MIN_BATH_RATED:
        bathroom_probe.train_and_save(conn)
    n, mae, alpha = loo_mae(conn)
    detail = f"alpha={alpha:.1f} LOO-MAE={mae:.2f}" if n > 2 else "insufficient samples"
    print(f"probe trained: n={n} {detail} (was {probe[2] if probe else 'untrained'})")

    if args.rescore:
        print(f"rescored {rescore(conn)} unrated listings with bath_model_score")


if __name__ == "__main__":
    main()
