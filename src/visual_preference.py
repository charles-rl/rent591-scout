"""Layer 1: Visual Preference Engine — DINOv3 embeddings -> single dino_visual_score scalar (0.0-1.0).

Phase 1 (<= RATED_THRESHOLD labeled listings): centroid cosine similarity against the
mean embedding of liked listings (user_score >= LIKED_MIN_SCORE). Avoids parameter
overfitting when labeled data is scarce.

Phase 2 (> RATED_THRESHOLD labeled listings): closed-form Ridge linear probe on the raw
768-d DINOv3 vectors (regression target (user_score-1)/4, output clipped to [0,1]),
learning per-dimension weights that reward liked aesthetics and penalize dark/dated
spaces. Weights persist atomically to models/dino_probe.npz (env DINO_PROBE_PATH).

Phase selection is automatic from database.get_rating_count(); VisualScorer caches the
probe across a batch so consolidation triggers once per scoring pass.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from . import database, scoring
from .deduplication import cosine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = Path(os.environ.get("DINO_PROBE_PATH", str(ROOT / "models" / "dino_probe.npz")))
PROBE_ALPHA = float(os.environ.get("DINO_PROBE_ALPHA", "10.0"))
LIKED_MIN_SCORE = float(os.environ.get("LIKED_MIN_SCORE", "4.0"))
EMBED_DIM = 768


def _vectors(rows) -> np.ndarray:
    mats = [np.frombuffer(r["dino_embedding"], dtype=np.float32) for r in rows if r["dino_embedding"]]
    mats = [m for m in mats if m.shape[0] == EMBED_DIM]
    if not mats:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.stack(mats)


def liked_centroid(conn, min_score: float = LIKED_MIN_SCORE) -> np.ndarray | None:
    """Mean L2-normalized embedding across all photos of liked (score >= min_score) listings."""
    X = _vectors(database.get_liked_embeddings(conn, min_score))
    if X.shape[0] == 0:
        return None
    centroid = X.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    return (centroid / norm).astype(np.float32) if norm > 0 else None


def centroid_score(vec: np.ndarray, centroid: np.ndarray | None) -> float:
    """Phase 1: cosine similarity to the liked centroid clipped to [0,1]; neutral 0.5 when unlabeled."""
    if centroid is None:
        return 0.5
    return float(np.clip(cosine(np.asarray(vec, dtype=np.float32), centroid), 0.0, 1.0))


def train_probe(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form Ridge w = (X^T X + aI)^-1 X^T y with a bias column (not penalized)."""
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    aug = np.hstack([X64, np.ones((X64.shape[0], 1), dtype=np.float64)])
    reg = PROBE_ALPHA * np.eye(aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(aug.T @ aug + reg, aug.T @ y64)
    return coef[:-1].astype(np.float32), float(coef[-1])


def save_probe(weights: np.ndarray, bias: float) -> None:
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROBE_PATH.with_suffix(".tmp.npz")
    np.savez(tmp, weights=weights, bias=np.float32(bias))
    tmp.replace(PROBE_PATH)


def load_probe() -> tuple[np.ndarray, float] | None:
    try:
        with np.load(PROBE_PATH) as data:
            weights = np.asarray(data["weights"], dtype=np.float32)
            bias = float(data["bias"])
    except (OSError, ValueError, KeyError):
        return None
    return (weights, bias) if weights.shape == (EMBED_DIM,) else None


def fit_probe(conn) -> tuple[np.ndarray, float]:
    rows = database.get_rated_samples(conn)
    X = _vectors(rows)
    if X.shape[0] == 0:
        raise RuntimeError("no embeddings available for probe training")
    y = (np.asarray([float(r["user_score"] or 0.0) for r in rows], dtype=np.float32) - 1.0) / 4.0
    return train_probe(X, y)


def probe_value(vec: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Linear projection clipped to [0,1]."""
    return float(np.clip(float(np.asarray(vec, dtype=np.float32) @ weights) + bias, 0.0, 1.0))


class VisualScorer:
    """Batch-cached dino_visual_score producer: centroid (cold) or Ridge probe (matured)."""

    def __init__(self, conn, phase: str, centroid: np.ndarray | None):
        self._conn = conn
        self.phase = phase
        self._centroid = centroid
        self._probe: tuple[np.ndarray, float] | None = None

    @classmethod
    def for_conn(cls, conn) -> VisualScorer:
        if database.get_rating_count(conn) > scoring.RATED_THRESHOLD:
            return cls(conn, "probe", None)
        return cls(conn, "centroid", liked_centroid(conn))

    def score(self, vec: np.ndarray) -> float:
        if self.phase == "centroid":
            return centroid_score(vec, self._centroid)
        if self._probe is None:
            self._probe = load_probe() or fit_probe(self._conn)
            try:
                save_probe(*self._probe)
            except OSError:
                logger.warning("probe persistence failed; continuing in-memory", exc_info=True)
        return probe_value(vec, *self._probe)


def compute_visual_score(conn, vec: np.ndarray) -> float:
    """One-shot entry point honoring the automatic Phase 1 -> Phase 2 transition."""
    return VisualScorer.for_conn(conn).score(vec)
