"""Bathroom-quality model: DINOv3 embeddings of bathroom photos -> bath score 0-5.

Bathroom photos are selected WITHOUT an extra LLM call: a cosine centroid built from
Qwen-labelled bathroom photos (listing_images.is_bathroom=1, seeded by
scripts/backfill_bathroom_imgs.py) picks candidate photos at scoring time.

A Ridge probe maps the pooled bathroom embedding to the user's bathroom_score.
Regularization follows the small-sample schedule requested by the user: alpha stays
heavy (BATH_PROBE_ALPHA_HEAVY) until BATH_PROBE_SOFT_N labels accumulate, then drops
to BATH_PROBE_ALPHA_SOFT. Weights persist to models/bathroom_probe.npz.

A listing with no detectable bathroom photos scores 0.0 ("no bathroom info"), which
scoring.FEATURE_NAMES consumes raw (0 means absent, not low quality).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from . import database
from .deduplication import cosine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = Path(os.environ.get("BATH_PROBE_PATH", str(ROOT / "models" / "bathroom_probe.npz")))
ALPHA_HEAVY = float(os.environ.get("BATH_PROBE_ALPHA_HEAVY", "100.0"))
ALPHA_SOFT = float(os.environ.get("BATH_PROBE_ALPHA_SOFT", "10.0"))
SOFT_N = int(os.environ.get("BATH_PROBE_SOFT_N", "30"))
MIN_BATH_RATED = int(os.environ.get("BATH_PROBE_MIN_RATED", "10"))
BATH_SIM_MIN = float(os.environ.get("BATH_SIM_MIN", "0.30"))
EMBED_DIM = 768


def alpha_for(n: int) -> float:
    """Heavy weight decay until enough bathroom labels, then soften (user schedule)."""
    return ALPHA_HEAVY if n <= SOFT_N else ALPHA_SOFT


def _unit(vecs) -> np.ndarray:
    out = []
    for v in vecs:
        if v is None:
            continue
        arr = np.frombuffer(v, dtype=np.float32) if isinstance(v, (bytes, bytearray, memoryview)) \
            else np.asarray(v, dtype=np.float32).ravel()
        if arr.shape[0] != EMBED_DIM:
            continue
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            out.append(arr / norm)
    return np.stack(out) if out else np.zeros((0, EMBED_DIM), dtype=np.float32)


def bath_centroid(conn) -> np.ndarray | None:
    """Mean unit embedding of Qwen-labelled bathroom photos across rated listings."""
    rows = database.get_bath_labeled_embeddings(conn)
    X = _unit(r["dino_embedding"] for r in rows)
    if X.shape[0] == 0:
        return None
    c = X.mean(axis=0)
    n = float(np.linalg.norm(c))
    return (c / n).astype(np.float32) if n > 0 else None


def select_bath_photos(centroid: np.ndarray | None, vecs: list) -> list[int]:
    """Ordinals (position in vecs) judged to show a bathroom; empty when unlabeled."""
    if centroid is None:
        return []
    hits = []
    for i, v in enumerate(vecs):
        X = _unit([v])
        if X.shape[0] and cosine(X[0], centroid) >= BATH_SIM_MIN:
            hits.append(i)
    return hits


def pool_unit(vecs: np.ndarray) -> np.ndarray | None:
    """L2-normalized mean of per-photo unit embeddings; None when empty."""
    if vecs.shape[0] == 0:
        return None
    p = vecs.mean(axis=0)
    n = float(np.linalg.norm(p))
    return (p / n).astype(np.float32) if n > 0 else None


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    aug = np.hstack([X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)])
    reg = alpha * np.eye(aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(aug.T @ aug + reg, aug.T @ y.astype(np.float64))
    return coef[:-1].astype(np.float32), float(coef[-1])


def training_data(conn) -> tuple[np.ndarray, np.ndarray, int]:
    """Pooled bathroom-photo vectors X and y=(bath_score-1)/4 for rated listings."""
    labels: dict[str, list[np.ndarray]] = {}
    scores: dict[str, float] = {}
    for r in database.get_bath_rated_samples(conn):
        v = _unit([r["dino_embedding"]])
        if v.shape[0] == 0:
            continue
        labels.setdefault(r["listing_id"], []).append(v[0])
        scores[r["listing_id"]] = float(r["bathroom_score"])
    X, y = [], []
    for lid, vecs in labels.items():
        p = pool_unit(np.stack(vecs))
        if p is not None:
            X.append(p)
            y.append((scores[lid] - 1.0) / 4.0)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), len(X))


def train_and_save(conn) -> tuple[np.ndarray, float, int]:
    X, y, n = training_data(conn)
    if n < MIN_BATH_RATED:
        raise RuntimeError(f"only {n} bathroom-rated listings; need {MIN_BATH_RATED}")
    w, b = _ridge_fit(X, y, alpha_for(n))
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROBE_PATH.with_suffix(".tmp.npz")
    np.savez(tmp, weights=w, bias=np.float32(b), n_trained=np.int32(n))
    tmp.replace(PROBE_PATH)
    logger.info("bathroom probe trained on %d listings (alpha=%.1f)", n, alpha_for(n))
    return w, b, n


def load_probe() -> tuple[np.ndarray, float, int] | None:
    try:
        with np.load(PROBE_PATH) as data:
            w = np.asarray(data["weights"], dtype=np.float32)
            b = float(data["bias"])
            n = int(data["n_trained"])
    except (OSError, ValueError, KeyError):
        return None
    return (w, b, n) if w.shape == (EMBED_DIM,) else None


def train_if_stale(conn) -> tuple[np.ndarray, float, int] | None:
    """Load saved probe; retrain when bathroom labels grew past what it saw."""
    probe = load_probe()
    n_now = len({r["listing_id"] for r in database.get_bath_rated_samples(conn)})
    if probe is None:
        if n_now < MIN_BATH_RATED:
            return None
        return train_and_save(conn)
    if n_now > probe[2]:
        try:
            return train_and_save(conn)
        except RuntimeError:
            return probe
    return probe


def probe_value(pool: np.ndarray | None, weights: np.ndarray, bias: float) -> float:
    """Ridge output mapped to 1-5; 0.0 means 'no bathroom evidence'."""
    if pool is None:
        return 0.0
    return float(np.clip(float(pool @ weights) + bias, 0.0, 1.0) * 4.0 + 1.0)


def predict_listing(conn, dino_vecs: list, centroid: np.ndarray | None) -> float:
    """Bathroom score 0-5 (0 = no bathroom photo detected) for one listing's photos."""
    probe = train_if_stale(conn)
    if probe is None or centroid is None:
        return 0.0
    idx = select_bath_photos(centroid, dino_vecs)
    if not idx:
        return 0.0
    return probe_value(pool_unit(_unit([dino_vecs[i] for i in idx])), probe[0], probe[1])
