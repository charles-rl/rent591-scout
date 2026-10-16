"""Bathroom-quality model: DINOv3 embeddings of bathroom photos -> bath score 0-5.

Photo selection is normally AUTHORED by Qwen: a detection pass
(src/bathroom_detect.py, seeded into listing_images.is_bathroom by
scripts/backfill_bathrooms.py) confirms which photos show the bathroom, and the
probe pools exactly those. The cosine centroid (built from labelled bathroom
photos) is only the FALLBACK, used when detection did not run or did not cover
every photo (see `predict_listing`, which takes the explicit `photo_indices`).

A Ridge probe maps the pooled bathroom embedding to the user's bathroom_score.
Regularization follows the small-sample schedule requested by the user: alpha stays
higher (BATH_PROBE_ALPHA_HEAVY) until BATH_PROBE_SOFT_N labels accumulate, then drops
to BATH_PROBE_ALPHA_SOFT. Both defaults are small in absolute terms (0.05 / 0.01) --
the 768-d unit-norm design matrix makes X^T X tiny, so only small alphas leave any
signal in the solve. Weights persist to models/bathroom_probe.npz.

A listing with no bathroom evidence scores 0.0, which scoring.FEATURE_NAMES
consumes raw. 0.0 covers BOTH "detection confirmed there is no bathroom" and
"no evidence at all" (probe never scored it) -- the head sees one bucket; the
tri-state verdict lives in listing_images.is_bathroom (1/0/NULL).
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import numpy as np

from . import database
from .deduplication import cosine

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = Path(os.environ.get("BATH_PROBE_PATH", str(ROOT / "models" / "bathroom_probe.npz")))
# Ridge strength. `_unit` and `pool_unit` both L2-normalize every training row, so
# trace(X^T X) = n EXACTLY; with rank(X) = n < 768, the n NONZERO eigenvalues average
# exactly 1.0. The system actually solved is augmented, [X|1], so its nonzero
# eigenvalues average 2.0. Read alpha against that scale -- NOT against trace/768,
# which averages in ~750 near-zero eigenvalues and overstates the shrinkage ~37x.
# At alpha=100 the solve collapsed onto the bias term: 97% of live bath_model_score
# were a constant 3.79, and LOO-MAE (1-5 scale) was worse than a constant predictor.
# Keep the heavy/soft pair inside [0.001, 0.5], ordered heavy > soft, and strictly
# positive (alpha=0 makes the system exactly singular and LAPACK returns garbage
# rather than raising); tests/test_probe_quality.py enforces the band and the check.
ALPHA_HEAVY = float(os.environ.get("BATH_PROBE_ALPHA_HEAVY", "0.05"))
ALPHA_SOFT = float(os.environ.get("BATH_PROBE_ALPHA_SOFT", "0.01"))
for _alpha_name, _alpha in (("BATH_PROBE_ALPHA_HEAVY", ALPHA_HEAVY), ("BATH_PROBE_ALPHA_SOFT", ALPHA_SOFT)):
    if not _alpha > 0.0:
        raise ValueError(f"{_alpha_name} must be > 0 (got {_alpha}); a zero/negative alpha "
                         "makes the ridge system singular")
SOFT_N = int(os.environ.get("BATH_PROBE_SOFT_N", "30"))
MIN_BATH_RATED = int(os.environ.get("BATH_PROBE_MIN_RATED", "10"))
BATH_SIM_MIN = float(os.environ.get("BATH_SIM_MIN", "0.30"))
EMBED_DIM = 768


def alpha_for(n: int) -> float:
    """Heavy weight decay while labels are few (n <= SOFT_N, inclusive), then soften.

    Both ends are small in absolute terms (0.05 / 0.01) relative to the ~1.0 data
    eigenvalue scale, so neither arm over-shrinks; the schedule controls the *ratio*,
    not the magnitude. The boundary is inclusive: n == SOFT_N still uses ALPHA_HEAVY.
    """
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
    """Closed-form Ridge with an unpenalized bias column.

    `alpha` must stay in [0.001, 0.5] -- see ALPHA_HEAVY for why larger values
    collapse the probe to a constant. Kept float64 for solve stability.
    """
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
    # pid-unique temp name: two concurrent trainers (main.py and backfill_bathrooms.py
    # both call this) otherwise interleave np.savez into the same file and one
    # tmp.replace() can publish a half-written archive.
    tmp = PROBE_PATH.with_name(f"{PROBE_PATH.stem}.tmp.{os.getpid()}.npz")
    np.savez(tmp, weights=w, bias=np.float32(b), n_trained=np.int32(n),
             alpha=np.float64(alpha_for(n)))
    tmp.replace(PROBE_PATH)
    logger.info("bathroom probe trained on %d listings (alpha=%.3f)", n, alpha_for(n))
    return w, b, n


def is_degenerate(conn, min_spread: float = 0.05) -> bool:
    """True when the probe carries no usable signal (collapsed to a constant).

    An over-regularized Ridge solve returns the training mean for every input. When
    that happens, injecting the estimate into the vision prompt is pure noise (and
    the value can even read as mild positive evidence), so callers use this to gate
    the hint. Conservative: an absent probe or too few labels both count as
    degenerate, so the hint is withheld rather than sent as noise.
    """
    probe = load_probe()
    if probe is None:
        return True
    X, _y, n = training_data(conn)
    if n < MIN_BATH_RATED:
        return True
    w, b = probe[0], probe[1]
    return float(np.std(X @ w + b)) < min_spread


def load_probe() -> tuple[np.ndarray, float, int] | None:
    # BadZipFile subclasses Exception but NOT OSError, so a truncated/copied archive
    # escaped the tuple below and crashed predict_listing instead of failing soft.
    try:
        with np.load(PROBE_PATH) as data:
            w = np.asarray(data["weights"], dtype=np.float32)
            b = float(data["bias"])
            n = int(data["n_trained"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return (w, b, n) if w.shape == (EMBED_DIM,) else None


def train_if_stale(conn) -> tuple[np.ndarray, float, int] | None:
    """Load saved probe; retrain when bathroom labels grew past what it saw.

    `n_now` must be counted the same way `training_data` counts `n_trained` (i.e. post
    `_unit` filtering), otherwise a listing whose blob is zero-norm or short-length is
    counted here but dropped there, leaving n_now permanently above n_trained and
    forcing a full 769x769 solve plus an npz rewrite on every predict_listing call.
    """
    probe = load_probe()
    if probe is None:
        _X, _y, n_now = training_data(conn)
        if n_now < MIN_BATH_RATED:
            return None
        return train_and_save(conn)
    if probe[2] < training_data(conn)[2]:
        try:
            return train_and_save(conn)
        except RuntimeError:
            return probe
    return probe


def probe_value(pool: np.ndarray | None, weights: np.ndarray, bias: float) -> float:
    """Ridge output mapped to 1-5; 0.0 = no bathroom evidence (see module docstring)."""
    if pool is None:
        return 0.0
    return float(np.clip(float(pool @ weights) + bias, 0.0, 1.0) * 4.0 + 1.0)


def predict_listing(conn, dino_vecs: list, centroid: np.ndarray | None,
                    photo_indices: list[int] | None = None) -> float:
    """Bathroom score 0-5 for one listing's photos (0.0 = no bathroom evidence).

    `photo_indices` are positional indices (into `dino_vecs`) of photos that a
    Qwen detection pass already confirmed show a bathroom. When supplied they are
    authoritative: the explicit "no bathroom in any of these photos" verdict must
    win, otherwise the centroid heuristic below re-selects a photo that detection
    rejected and reports a score for a bathroom nobody can see. Pass None only
    when detection did not run (e.g. the LLM was down), which is the case the
    centroid selector exists to cover.
    """
    probe = train_if_stale(conn)
    if probe is None:
        return 0.0
    if photo_indices is not None:
        # Authoritative: the probe alone suffices, so a missing centroid (no
        # bath-labelled photo exists anywhere, e.g. after a label wipe) must not
        # discard a confirmed bathroom.
        idx = list(photo_indices)
    else:
        idx = select_bath_photos(centroid, dino_vecs)
    idx = [i for i in idx if 0 <= i < len(dino_vecs)]
    if not idx:
        return 0.0
    return probe_value(pool_unit(_unit([dino_vecs[i] for i in idx])), probe[0], probe[1])
