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
import zipfile
from pathlib import Path

import numpy as np

from . import database, scoring
from .deduplication import cosine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = Path(os.environ.get("DINO_PROBE_PATH", str(ROOT / "models" / "dino_probe.npz")))
# Ridge strength. `deduplication.aggregate_embedding` stores the UNNORMALIZED mean
# of a listing's per-photo unit vectors, so these rows are unit-norm at most:
# trace(X^T X) = sum of squared row norms < n, and the n nonzero eigenvalues
# average that same mean -- NOT n/768. The old alpha=10 therefore over-shrank the
# data-carrying directions by ~10-40x, not the ~370x you get by dividing alpha by
# the mean over all 769 eigenvalues (that mean counts the ~750 structural zeros
# ridge exists to zero out). At that strength the solve collapsed onto the bias
# term: dino_visual_score spanned 0.676-0.698 across 965 listings (2026-09-28
# snapshot), i.e. worse than a constant predictor. Measured LOO-MAE on live rated
# rows (n=27, 2026-10-05): alpha=10 -> 0.169, alpha=0.01 -> 0.097, constant
# baseline 0.165. Keep alpha in [0.001, 0.1] and strictly positive (alpha=0 makes
# the augmented Gram exactly rank-deficient and LAPACK returns garbage rather
# than raising); tests/test_probe_quality.py enforces the band.
PROBE_ALPHA = float(os.environ.get("DINO_PROBE_ALPHA", "0.01"))
if not PROBE_ALPHA > 0.0:
    raise ValueError(f"DINO_PROBE_ALPHA must be > 0 (got {PROBE_ALPHA}); a zero/negative "
                     "alpha makes the ridge system singular")
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
    """Closed-form Ridge w = (X^T X + aI)^-1 X^T y with a bias column (not penalized).

    PROBE_ALPHA must stay in [0.001, 0.1] -- see its definition for why larger
    values collapse the probe to a constant.
    """
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    aug = np.hstack([X64, np.ones((X64.shape[0], 1), dtype=np.float64)])
    reg = PROBE_ALPHA * np.eye(aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(aug.T @ aug + reg, aug.T @ y64)
    return coef[:-1].astype(np.float32), float(coef[-1])


def train_id_of(weights: np.ndarray, bias: float) -> str:
    """Content hash of a probe's parameters; ties the head to the exact probe it trained on."""
    import hashlib
    payload = np.asarray(weights, dtype=np.float32).tobytes() + np.float32(bias).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_probe(weights: np.ndarray, bias: float, n_trained: int = 0) -> None:
    """Atomically write the probe, stamped with the geometry a consumer needs to detect staleness.

    Without `alpha`/`n_trained`/`train_id`, two probes of different vintage are byte-indistinguishable
    to `load_probe`, and `models/xgboost_head.json` cannot be checked against the probe it was fit on
    (the head now trains on a cross-fitted column that is NOT derivable from this file alone).
    """
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROBE_PATH.with_name(f"{PROBE_PATH.stem}.tmp.{os.getpid()}.npz")
    np.savez(tmp, weights=weights, bias=np.float32(bias),
             alpha=np.float64(PROBE_ALPHA), n_trained=np.int32(n_trained),
             train_id=np.str_(train_id_of(weights, bias)))
    tmp.replace(PROBE_PATH)


def load_probe_meta() -> dict | None:
    """Stamped metadata only; None when the probe is absent or predates stamping."""
    try:
        with np.load(PROBE_PATH) as data:
            if "train_id" not in data:
                return None
            return {"alpha": float(data["alpha"]), "n_trained": int(data["n_trained"]),
                    "train_id": str(data["train_id"])}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def load_probe() -> tuple[np.ndarray, float] | None:
    try:
        with np.load(PROBE_PATH) as data:
            weights = np.asarray(data["weights"], dtype=np.float32)
            bias = float(data["bias"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return (weights, bias) if weights.shape == (EMBED_DIM,) else None


def fit_probe(conn) -> tuple[np.ndarray, float]:
    rows = database.get_rated_samples(conn)
    X, y = probe_dataset(rows)
    if X.shape[0] == 0:
        raise RuntimeError("no embeddings available for probe training")
    return train_probe(X, y)


def probe_dataset(rows) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) restricted to rows with a usable 768-d embedding, index-aligned.

    `_vectors` drops rows whose blob is NULL or short, so `y` MUST be built from the
    same surviving positions -- otherwise `aug.T @ y` raises a shape error and the
    probe path dies silently. `cross_fitted_values` already worked this way.
    """
    X = _vectors(rows)
    idx = [j for j, r in enumerate(rows)
           if r["dino_embedding"] and np.frombuffer(r["dino_embedding"], dtype=np.float32).shape[0] == EMBED_DIM]
    y = (np.asarray([float(rows[j]["user_score"] or 0.0) for j in idx], dtype=np.float32) - 1.0) / 4.0
    return X, y


def probe_value(vec: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """Linear projection clipped to [0,1]."""
    return float(np.clip(float(np.asarray(vec, dtype=np.float32) @ weights) + bias, 0.0, 1.0))


def cross_fitted_values(rows) -> np.ndarray | None:
    """Leave-one-out Layer-1 scores for the labeled rows, aligned to `rows`.

    The probe is fit on the very rows the Layer-3 head trains on, so its in-sample
    values are a near-perfect predictor of user_score (measured corr 1.000 at both
    n=21 and n=27). XGBoost then reaches 100% of its gain from a single split on
    that leaked column, which is why the head ignored all 17 other features.

    These out-of-fold values describe the probe's behaviour on UNSEEN listings --
    which is what the head actually has to generalize to. Entries are NaN where a
    row has no usable embedding; callers should fall back to the full probe there.
    Returns None when there is too little data to hold anything out.
    """
    vecs: list[np.ndarray] = []
    idx: list[int] = []
    for j, r in enumerate(rows):
        if not r["dino_embedding"]:
            continue
        arr = np.frombuffer(r["dino_embedding"], dtype=np.float32)
        if arr.shape[0] != EMBED_DIM:
            continue
        vecs.append(arr)
        idx.append(j)
    if len(vecs) < 3:  # holding out one of <3 rows is not meaningful
        return None
    X = np.stack(vecs)
    if float(np.mean(X.std(axis=0))) < 1e-6:
        # Every labeled row carries the same embedding, so the probe can only ever
        # emit its bias. Leave-one-out would then vary only with the *fold mean*,
        # i.e. hand the head a noisy proxy for user_score to overfit. Nothing to
        # generalize -> let the caller keep the full-probe value instead.
        return None
    y = (np.asarray([float(rows[j]["user_score"] or 0.0) for j in idx], dtype=np.float32) - 1.0) / 4.0
    out = np.full(len(rows), np.nan, dtype=np.float32)
    for i in range(X.shape[0]):
        keep = np.ones(X.shape[0], dtype=bool)
        keep[i] = False
        w, b = train_probe(X[keep], y[keep])
        out[idx[i]] = probe_value(X[i], w, b)
    return out


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
            probe = load_probe()
            if probe is None:
                probe = fit_probe(self._conn)
                try:
                    save_probe(probe[0], probe[1], database.get_rating_count(self._conn))
                except OSError:
                    logger.warning("probe persistence failed; continuing in-memory", exc_info=True)
            self._probe = probe
        return probe_value(vec, *self._probe)


def compute_visual_score(conn, vec: np.ndarray) -> float:
    """One-shot entry point honoring the automatic Phase 1 -> Phase 2 transition."""
    return VisualScorer.for_conn(conn).score(vec)
