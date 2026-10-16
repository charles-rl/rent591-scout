"""Layer 3: feature fusion + XGBoost scoring head.

Phase 1 (<=RATED_THRESHOLD labeled samples): predicted_score = qwen_direct_score.
Phase 2 (>RATED_THRESHOLD): XGBRegressor over the compressed fusion vector -> user_score.
Order matches FEATURE_NAMES exactly (XGBoost is position-sensitive):
[dino_visual_score (Layer 1 scalar); qwen_score = (qwen_direct_score-1)/4;
 5 vision flags + 1 has-warnings bit; tabular: log1p price / area_ping / price_per_ping /
 floor_num; HIGH_ELEC_FEE; MANUAL_TRASH; NO_PETS; ELEC_EXTRA_HIGH_COST;
 log_days_on_market (pre-computed snapshot of parse_days_on_market; -1.0 sentinel =
 never parsed, 0.0 = brand new; a fresh still-open listing is a quality signal --
 good units rent fast);
 bath_model_score (bathroom probe 1-5 mapped through normalize_bath; 0.0 covers BOTH
 "detection confirmed no bathroom" and "no evidence at all" -- they are the same
 input to the head, and the tri-state lives in listing_images.is_bathroom) ].
The raw 768-d DINOv3 vector is
compressed to dino_visual_score by src/visual_preference.py, never concatenated directly.

Also hosts the Stage-3 deterministic penalty engine (docs/591research.md §4):
baseline 100 minus regex-extracted red flags. Penalties fire only on positive
textual evidence — ambiguous listings stay penalty-free and go to the user.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import xgboost as xgb

from . import database
from .refresh_age import parse_days_on_market  # noqa: F401  # re-export; relay import graph must stay light

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RATED_THRESHOLD = int(os.environ.get("RATED_THRESHOLD", "20"))
MODEL_PATH = Path(os.environ.get("XGB_MODEL_PATH", str(ROOT / "models" / "xgboost_head.json")))
EMBED_DIM = 768
_FLAG_ORDER = [
    "has_bathroom_img",
    "shower_sink_combo",
    "drainage_risk",
    "has_kitchen_sink",
    "has_exterior_window",
]


def flag_vector(flags: dict | None, warnings: list | None) -> np.ndarray:
    flags = flags or {}
    vec = [1.0 if flags.get(k) else 0.0 for k in _FLAG_ORDER]
    vec.append(1.0 if warnings else 0.0)
    return np.asarray(vec, dtype=np.float32)


def _as_embedding(vec) -> np.ndarray:
    """Normalize a stored BLOB (or ndarray) to a fixed-width (768,) float32 row."""
    if vec is None:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    if isinstance(vec, (bytes, bytearray, memoryview)):
        if len(vec) % 4:
            logger.warning("embedding blob length %d not float-aligned; zero-filling", len(vec))
            return np.zeros(EMBED_DIM, dtype=np.float32)
        arr = np.frombuffer(vec, dtype=np.float32)
    else:
        arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.shape[0] != EMBED_DIM:
        logger.warning("embedding dim %d != %d; padding/truncating", arr.shape[0], EMBED_DIM)
        fixed = np.zeros(EMBED_DIM, dtype=np.float32)
        n = min(EMBED_DIM, arr.shape[0])
        fixed[:n] = arr[:n]
        arr = fixed
    return arr


def _load_json(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


FEATURE_NAMES = [
    "dino_visual_score", "qwen_score",
    "has_bathroom_img", "shower_sink_combo", "drainage_risk", "has_kitchen_sink",
    "has_exterior_window", "has_warnings",
    "log_price", "area_ping", "price_per_ping", "floor_num",
    "HIGH_ELEC_FEE", "MANUAL_TRASH", "NO_PETS", "ELEC_EXTRA_HIGH_COST",
    "log_days_on_market", "bath_model_score",
]
# Stamped into the head so a rename/reorder of FEATURE_NAMES invalidates it even when
# the width is unchanged (width alone was the only integrity check before).
FEATURE_NAMES_SHA = hashlib.sha256("\0".join(FEATURE_NAMES).encode()).hexdigest()[:16]


def normalize_bath(bath_model_score) -> float:
    """Bathroom probe score 1-5 -> [0,1]; 0.0 when absent.

    0.0 is the shared bucket for "detection confirmed there is no bathroom" and
    "no evidence at all" (probe never scored it) -- the head cannot tell them
    apart, and the tri-state verdict lives in listing_images.is_bathroom.
    """
    s = _num(bath_model_score)
    if s <= 0.0:
        return 0.0
    return float(np.clip(s, 1.0, 5.0) / 5.0)


def _num(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def parse_floor_number(floor) -> float:
    """'12樓' / '5' / 'B1' -> leading floor number as float; basement/unknown -> 0."""
    m = re.search(r"\d+", str(floor or ""))
    if not m:
        return 0.0
    n = int(m.group(0))
    return -float(n) if "地" in str(floor) or str(floor).strip().upper().startswith("B") else float(n)


def normalize_qwen(qwen_direct_score) -> float:
    """Layer 2 score on the native 1-5 scale compressed to qwen_score in [0,1]; neutral 0.5 when absent."""
    if qwen_direct_score is None:
        return 0.5
    return float(np.clip((_num(qwen_direct_score) - 1.0) / 4.0, 0.0, 1.0))


def tabular_vector(listing: dict | None) -> np.ndarray:
    listing = listing or {}
    price = _num(listing.get("price"))
    area = _num(listing.get("area"))
    per_ping = price / area if area > 0 else 0.0
    penalty_flags = set(heuristic_penalties(listing))
    # days_on_market is a pre-computed snapshot (see parse_days_on_market); the
    # -1.0 sentinel below represents "never parsed". A non-numeric value is
    # treated as unparsed, NOT as 0 (which would read as "brand new").
    raw_dom = listing.get("days_on_market")
    try:
        dom = float(raw_dom) if raw_dom is not None else None
    except (TypeError, ValueError):
        dom = None
    if dom is not None and not np.isfinite(dom):
        dom = None  # nan/inf would otherwise survive max(0.0, dom) as a fake 0.0 = "brand new"
    vec = [
        float(np.log1p(price)),
        area,
        per_ping,
        parse_floor_number(listing.get("floor")),
        1.0 if "HIGH_ELEC_FEE" in penalty_flags else 0.0,
        1.0 if "MANUAL_TRASH" in penalty_flags else 0.0,
        1.0 if "NO_PETS" in penalty_flags else 0.0,
        1.0 if "ELEC_EXTRA_HIGH_COST" in penalty_flags else 0.0,
        # -1.0 sentinel: an unparseable/ABSENT age must not read as "brand new"
        # (log1p(0) == 0.0). XGBoost splits the sentinel into its own branch,
        # which is cheaper than adding a 19th has_days_on_market bit to an 18-wide vector.
        -1.0 if dom is None else float(np.log1p(max(0.0, dom))),
    ]
    return np.asarray(vec, dtype=np.float32)


def fusion_vector(visual_score: float, qwen_direct_score, flags: dict | None,
                  warnings: list | None, listing: dict | None) -> np.ndarray:
    """Layer 1 scalar + Layer 2 scalar + vision flags + tabular + bathroom probe -> XGB input."""
    listing = listing or {}
    return np.concatenate([
        np.asarray([visual_score, normalize_qwen(qwen_direct_score)], dtype=np.float32),
        flag_vector(flags, warnings),
        tabular_vector(listing),
        np.asarray([normalize_bath(listing.get("bath_model_score"))], dtype=np.float32),
    ])


def _row_listing(row) -> dict:
    return {
        "price": row["price"], "area": row["area"], "floor": row["floor"],
        "shape": row["shape"], "tags": _load_json(row["tags"], []),
        "facilities": _load_json(row["facilities"], []),
        "contain_cost": _load_json(row["contain_cost"], []),
        "description": row["description"],
        "days_on_market": row["days_on_market"],
    }


def _row_features(row, visual_score: float) -> np.ndarray:
    flags = _load_json(row["qwen_vision_flags"], {})
    warnings = _load_json(row["qwen_warnings"], [])
    listing = _row_listing(row)
    try:
        listing["bath_model_score"] = row["bath_model_score"]
    except (IndexError, KeyError):
        pass
    return fusion_vector(visual_score, row["qwen_direct_score"], flags, warnings, listing)


def _load_trained_model(conn, allow_retrain: bool = True) -> xgb.XGBRegressor | None:
    """Load the head, retraining it when its stamps disagree with the live state.

    `allow_retrain=False` is the read-only path (`check_score_health`): a missing or
    stale head returns None instead of triggering a train_and_save, which would
    rewrite models/ and rated-row preference scores from a "health check".
    """
    if MODEL_PATH.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        stale = _head_is_stale(model, conn)
        if stale:
            if not allow_retrain:
                return None
            train_and_save(conn)
            model = xgb.XGBRegressor()
            model.load_model(str(MODEL_PATH))
        return model
    if not allow_retrain:
        return None
    train_and_save(conn)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def _head_is_stale(model, conn) -> bool:
    """True when the saved head cannot be trusted against the current feature set / probe.

    Feature WIDTH alone is not enough: the head now trains on a cross-fitted Layer-1
    column that is not derivable from `dino_probe.npz`, so two same-width heads from
    different probe vintages are otherwise byte-indistinguishable.
    """
    from . import visual_preference

    got = model.get_booster().num_features()
    if got != len(FEATURE_NAMES):
        return True
    attrs = model.get_booster().attributes() or {}
    stamp = attrs.get("feature_names_sha")
    if stamp and stamp != FEATURE_NAMES_SHA:
        return True
    width = attrs.get("feature_width")
    if width and int(width) != len(FEATURE_NAMES):
        return True
    meta = visual_preference.load_probe_meta()
    if meta is None:
        # Unstamped/absent probe (predates metadata) — a head that claims a probe
        # cannot be proven against it.
        return bool(attrs.get("probe_train_id"))
    if attrs.get("probe_train_id"):
        if attrs["probe_train_id"] != meta["train_id"]:
            return True
        if attrs.get("probe_alpha") and float(attrs["probe_alpha"]) != float(meta["alpha"]):
            return True
    elif attrs.get("dino_probe") != "0":
        # A stamped probe exists but the head does not identify one: an unstamped
        # legacy head restored next to a newer probe (the false-fresh case) must
        # retrain. The one accepted gap: a head explicitly stamped dino_probe=0
        # (fit_probe failed at train time) is kept as-is, because a retrain would
        # fail the same way and this check would then fire on every predict.
        return True
    n_now = database.get_rating_count(conn)
    return False if n_now <= meta["n_trained"] else attrs.get("n_trained") != str(n_now)


def predict_score(conn, dino_vec: np.ndarray, flags: dict, warnings: list,
                  qwen_direct_score: float, listing: dict | None = None) -> tuple[float, str]:
    """Return (predicted_score 1-5, score_source 'qwen'|'xgboost')."""
    from . import visual_preference

    rated = database.get_rating_count(conn)
    if rated <= RATED_THRESHOLD:
        return float(qwen_direct_score), "qwen"

    model = None
    try:
        # Load the model FIRST: it may retrain and overwrite the probe, and the visual
        # scalar must come from the same probe generation as the head's split thresholds.
        model = _load_trained_model(conn)
        visual = visual_preference.VisualScorer.for_conn(conn).score(_as_embedding(dino_vec))
        vec = fusion_vector(visual, qwen_direct_score, flags, warnings, listing)
        pred = float(model.predict(vec.reshape(1, -1))[0])
        return max(1.0, min(5.0, pred)), "xgboost"
    except Exception:
        logger.exception("xgboost prediction failed -> falling back to qwen_direct_score")
        return float(qwen_direct_score), "qwen"
    finally:
        del model
        gc.collect()


_MIN_SCORE_SPREAD = 0.05       # std floor for a believable 1-5 score distribution
_MIN_SCORE_DISTINCT_RATIO = 0.10  # distinct-value/row ratio floor
_MIN_COLLAPSE_SAMPLES = 20      # below this the distinct-ratio test is unsatisfiable (0.10*10 == 1.0)
_MIN_COLLAPSE_STD = 1e-12       # "exactly constant" — std below this is benign


def _warn_if_collapsed(preds: dict[str, float], X: np.ndarray, model=None) -> None:
    """Surface a degenerate Layer-3 output loudly, naming the collapsed features.

    The XGBoost head only auto-retrains on a feature-WIDTH/stamp mismatch, so a model
    that leans on an over-regularized (constant) probe can persist indefinitely: the
    split thresholds stay, the signal underneath them does not. Fails soft (warn, never
    raise) per the house style.

    A column is only implicated when the head ACTUALLY SPLIT ON IT *and* it is near
    constant: XGBoost cannot split a constant column, so a zero-gain near-constant
    feature is benign by construction (all-NULL flags, an unset penalty bit) and must
    not be reported. The previous `std < 1e-3` test could not name the real failure
    (a collapsed probe sits at std ~0.005-0.01, two orders of magnitude higher).
    """
    values = np.asarray(list(preds.values()), dtype=np.float64)
    if values.size < 2:
        return
    stored = np.clip(values, 1.0, 5.0)  # what actually persists; the guard used to see raw
    spread = float(stored.std())
    distinct = len({round(float(v), 3) for v in stored})
    flat = spread < _MIN_SCORE_SPREAD
    if values.size >= _MIN_COLLAPSE_SAMPLES:
        flat = flat or distinct < _MIN_SCORE_DISTINCT_RATIO * values.size
    if not flat:
        return

    gain: dict[str, float] = {}
    if model is not None:
        try:
            gain = model.get_booster().get_score(importance_type="gain") or {}
        except Exception:
            gain = {}
    gain_known = bool(gain)
    stds = X.std(axis=0) if X.size else np.zeros(0)
    used_std = sorted((float(s) for s in stds), reverse=True)
    # Relative yardstick: a collapsed probe is far flatter than the features actually
    # carrying gain. Absolute cross-column std comparison is meaningless (area_ping
    # varies by tens, log_price by tenths), and the 0.02 floor covers the historical
    # collapsed-probe band (std ~0.005-0.01) that the old `1e-3` test could not name.
    reference = used_std[len(used_std) // 2] if used_std else 0.0
    collapsed, benign = [], []
    for i, name in enumerate(FEATURE_NAMES):
        if i >= stds.size:
            continue
        share = float(gain.get(f"f{i}", 0.0)) if gain_known else 1.0
        std = float(stds[i])
        if share <= 0.0:
            # XGBoost cannot split a constant column, so a zero-gain near-constant
            # feature is benign by construction (all-NULL flags, an unset penalty bit).
            if std < _MIN_COLLAPSE_STD:
                benign.append(name)
            continue
        if std < _MIN_COLLAPSE_STD or std < max(0.02, 0.25 * reference):
            collapsed.append(f"{name}(std={std:.4f})")
    logger.error(
        "collapsed score distribution: std=%.4f distinct=%d/%d -- the head is ranking on "
        "degenerate features. Splitting on near-constant columns: %s. Constant but "
        "unsplit (benign): %s. Retrain with `main.py --train` after checking the probe alphas.",
        spread, distinct, values.size,
        ", ".join(collapsed) or "(none detected; head itself may be degenerate)",
        ", ".join(benign) or "none",
    )


def check_score_health(conn) -> None:
    """Run the collapse guard over the current unrated set WITHOUT persisting anything.

    Used by the 30-min --incoming cycle, whose per-listing `predict_score` path never
    reaches the guard. Strictly read-only: a missing or stale head is skipped with a
    log (`allow_retrain=False`) rather than retrained, so a health check cannot
    rewrite models/, rated-row preference scores, or any stored predicted_score.
    """
    from . import visual_preference

    rows = database.get_scoring_rows(conn)
    if not rows or database.get_rating_count(conn) <= RATED_THRESHOLD:
        return
    try:
        model = _load_trained_model(conn, allow_retrain=False)
        if model is None:
            logger.warning("score health check skipped: head missing or stale (would retrain)")
            return
        scorer = visual_preference.VisualScorer.for_conn(conn)
        X = np.stack([_row_features(r, scorer.score(_as_embedding(r["dino_embedding"]))) for r in rows])
        preds = {r["listing_id"]: float(p) for r, p in zip(rows, model.predict(X), strict=True)}
        _warn_if_collapsed(preds, X, model)
    except Exception:
        logger.exception("score health check failed")
    finally:
        gc.collect()


def score_all_unrated(conn) -> int:
    """Batch-score unrated listings and persist predicted_score/score_source to the DB."""
    from . import visual_preference

    rows = database.get_scoring_rows(conn)
    if not rows:
        return 0

    rated = database.get_rating_count(conn)
    matured = rated > RATED_THRESHOLD
    # Layer 1 scalars are produced in both phases (centroid cold / probe matured).
    scorer = visual_preference.VisualScorer.for_conn(conn)
    preds: dict[str, float] = {}
    model = None
    if matured:
        try:
            model = _load_trained_model(conn)
            X = np.stack([
                _row_features(r, scorer.score(_as_embedding(r["dino_embedding"]))) for r in rows
            ])
            preds = {r["listing_id"]: float(p) for r, p in zip(rows, model.predict(X), strict=True)}
            _warn_if_collapsed(preds, X, model)
        except Exception:
            logger.exception("xgboost batch scoring failed -> falling back to qwen scores")
            preds = {}
        finally:
            del model
            gc.collect()

    updates: list[tuple[str, float, str]] = []
    for r in rows:
        if r["listing_id"] in preds:
            updates.append((r["listing_id"], max(1.0, min(5.0, preds[r["listing_id"]])), "xgboost"))
        elif r["qwen_direct_score"] is not None:
            updates.append((r["listing_id"], float(r["qwen_direct_score"]), "qwen"))
    if updates:
        database.set_predicted_scores(conn, updates)
    database.set_preference_scores(conn, [
        (r["listing_id"], scorer.score(_as_embedding(r["dino_embedding"])), normalize_qwen(r["qwen_direct_score"]))
        for r in rows
    ])
    logger.info("scored %d unrated listings", len(updates))
    return len(updates)


# --------------------------------------------------------------------------
# Stage-3 deterministic heuristic penalty engine (docs/591research.md §4)
# Baseline 100; every penalty requires positive textual evidence so uncertain
# listings keep 100 and are surfaced to the user via warnings instead.
# --------------------------------------------------------------------------
HEURISTIC_BASELINE = 100.0
ELEC_EXTRA_COST_RENT = float(os.environ.get("ELEC_EXTRA_COST_RENT", "15600"))
PENALTY_POINTS: dict[str, int] = {
    "HIGH_ELEC_FEE": -15,
    "NO_PETS": -10,
    "HIGH_WALKUP": -25,
    "ILLEGAL_ROOFTOP": -10,
    "MANUAL_TRASH": -10,
    "SHARED_WASHER": -5,
    "ELEC_EXTRA_HIGH_COST": -10,
}
PENALTY_MESSAGES: dict[str, str] = {
    "HIGH_ELEC_FEE": "Electricity billed above 5 NTD/kWh",
    "NO_PETS": "No pets allowed",
    "HIGH_WALKUP": "5F+ walk-up, no elevator",
    "ILLEGAL_ROOFTOP": "Suspected illegal rooftop addition",
    "MANUAL_TRASH": "Manual trash disposal (garbage-truck chasing)",
    "SHARED_WASHER": "Shared / coin-operated laundry",
    "ELEC_EXTRA_HIGH_COST": f"Rent over {ELEC_EXTRA_COST_RENT:,.0f} with electricity billed separately",
}
_PET_PATTERNS = ("不可寵", "禁寵", "嚴禁寵物", "不可養寵", "不開放寵")
_ROOFTOP_PATTERNS = ("頂樓加蓋", "頂加", "鐵皮加蓋")
_TRASH_PATTERNS = ("追垃圾車",)
_WASHER_PATTERNS = ("投幣洗衣", "投幣式洗衣", "共享洗衣")
_ELEC_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|塊|NTD?|TWD?)\s*/\s*(?:度|kwh)", re.IGNORECASE)
_ELEC_PER_RE = re.compile(r"(?:每|一)\s*度\s*(?:是|為|=)?\s*(\d+(?:\.\d+)?)")
_ELEC_REVERSED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*度\s*(\d+(?:\.\d+)?)\s*(?:元|塊|NTD?|TWD?)", re.IGNORECASE)
_ELEC_INCLUDED_PATTERNS = ("含電費", "電費含", "含水電", "水電含", "免電費", "電費免")
_FLOOR_RE = re.compile(r"(\d+)")


def heuristic_penalties(listing: dict) -> list[str]:
    blob = " ".join(filter(None, [
        str(listing.get("description") or ""),
        " ".join(str(t) for t in (listing.get("tags") or [])),
        " ".join(str(f) for f in (listing.get("facilities") or [])),
    ]))
    flags: list[str] = []
    rates = [float(m.group(1)) for rx in (_ELEC_RATE_RE, _ELEC_PER_RE) for m in rx.finditer(blob)]
    rates += [float(m.group(2)) for m in _ELEC_REVERSED_RE.finditer(blob)]
    if any(r > 5.0 for r in rates):
        flags.append("HIGH_ELEC_FEE")
    if any(p in blob for p in _PET_PATTERNS):
        flags.append("NO_PETS")
    if any(p in blob for p in _ROOFTOP_PATTERNS):
        flags.append("ILLEGAL_ROOFTOP")
    if any(p in blob for p in _TRASH_PATTERNS):
        flags.append("MANUAL_TRASH")
    if any(p in blob for p in _WASHER_PATTERNS):
        flags.append("SHARED_WASHER")
    facilities = [str(f) for f in (listing.get("facilities") or [])]
    has_elevator = listing.get("shape") == "電梯大樓" or any("電梯" in f for f in facilities)
    fm = _FLOOR_RE.search(str(listing.get("floor") or ""))
    if fm and int(fm.group(1)) >= 5 and not has_elevator:
        flags.append("HIGH_WALKUP")
    if _num(listing.get("price")) > ELEC_EXTRA_COST_RENT and not _electricity_included(listing, blob):
        flags.append("ELEC_EXTRA_HIGH_COST")
    return flags


def _electricity_included(listing: dict, blob: str) -> bool:
    """True when rent demonstrably covers electricity (contain_cost entry or description wording)."""
    contain = listing.get("contain_cost") or []
    if isinstance(contain, str):
        contain = _load_json(contain, [])
    for entry in contain:
        text = str(entry if isinstance(entry, str) else (entry or {}).get("name") or (entry or {}).get("value"))
        if "電費" in text:
            return True
    return any(p in blob for p in _ELEC_INCLUDED_PATTERNS)


def compute_heuristic_score(listing: dict) -> tuple[float, list[str]]:
    """Return (score, warning strings) — baseline 100 minus evidenced penalties."""
    flags = heuristic_penalties(listing)
    score = HEURISTIC_BASELINE + sum(PENALTY_POINTS[f] for f in flags)
    warnings = [f"{f}：{PENALTY_MESSAGES[f]}" for f in flags]
    return score, warnings


def train_and_save(conn) -> None:
    """Train the Layer-3 XGBoost head on fusion vectors (Layer-1 probe trained alongside)."""
    from . import visual_preference

    rows = database.get_rated_samples(conn)
    if not rows:
        raise RuntimeError("no rated samples to train on")
    y = np.asarray([float(r["user_score"]) for r in rows], dtype=np.float32)

    visual_probe = None
    try:
        visual_probe = visual_preference.fit_probe(conn)
    except Exception:
        logger.exception("Layer-1 probe training failed; falling back to liked centroid features "
                         "(note: the SAVED probe is left untouched and will keep being served "
                         "against a head trained on centroid features)")
    if visual_probe is not None:
        try:
            visual_preference.save_probe(visual_probe[0], visual_probe[1], len(rows))
        except Exception:
            # Not fatal, but the head will now claim a probe identity the on-disk
            # file does not carry -- _head_is_stale treats that as stale and
            # retrains, which is the honest outcome over silently pairing a new
            # head with an old probe.
            logger.exception("saving dino_probe.npz failed; the head trains on the new "
                             "in-memory probe and will look stale against the on-disk one")
    centroid = None if visual_probe else visual_preference.liked_centroid(conn)

    from . import bathroom_probe
    try:
        bathroom_probe.train_and_save(conn)
    except Exception:
        logger.info("bathroom probe not trained (labels below threshold or no photo labels yet)")

    def visual_of(row) -> float:
        vec = _as_embedding(row["dino_embedding"])
        if visual_probe:
            return visual_preference.probe_value(vec, *visual_probe)
        return visual_preference.centroid_score(vec, centroid)

    # Build the head's training matrix on OUT-OF-FOLD Layer-1 values. The probe
    # was fit on these same rows, so its in-sample output is a near-perfect
    # predictor of user_score (corr 1.000) -- a leak that hands XGBoost all of its
    # gain on one column and starves the other 17. Cross-fitting makes the
    # training features match what the probe produces for unseen listings.
    # Only the probe path is cross-fitted: the centroid is a single direction fit
    # over liked rows, so it cannot memorise individual labels the way a
    # 768-parameter probe can.
    if visual_probe:
        cross_fitted = visual_preference.cross_fitted_values(rows)
        if cross_fitted is None:
            logger.warning(
                "cross_fitted_values returned None -> training head on IN-SAMPLE Layer-1 "
                "features, which reintroduces the structural leak (XGBoost will take all "
                "its gain from dino_visual_score). Cause is <3 usable embeddings or the "
                "zero-variance bail in visual_preference.cross_fitted_values.")
    else:
        cross_fitted = None
        logger.info("no dino probe -> training head on centroid Layer-1 features")
    if cross_fitted is not None:
        logger.info("training head on out-of-fold Layer-1 features (%d/%d rows)",
                    int(np.isfinite(cross_fitted).sum()), len(rows))
    X = np.stack([
        _row_features(r, float(cross_fitted[i])
                      if cross_fitted is not None and np.isfinite(cross_fitted[i])
                      else visual_of(r))
        for i, r in enumerate(rows)
    ])
    model = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
    try:
        # Stamp the head with the identity of the probe (and feature set) it trained on.
        # The head's training column is CROSS-FITTED and therefore not derivable from
        # dino_probe.npz, so without this stamp the pair cannot be validated after a
        # restore. The stamp comes from the IN-MEMORY fit, not a post-save re-read of
        # the npz: a save that failed would otherwise stamp the OLD on-disk identity
        # onto a head trained on NEW geometry, and _head_is_stale would then call the
        # mismatched pair "fresh".
        probe_alpha = str(visual_preference.PROBE_ALPHA) if visual_probe else ""
        probe_train_id = (visual_preference.train_id_of(visual_probe[0], visual_probe[1])
                          if visual_probe else "")
        model.fit(X, y)
        booster = model.get_booster()
        for key, value in (
            ("feature_names_sha", FEATURE_NAMES_SHA),
            ("feature_width", str(X.shape[1])),
            ("n_trained", str(len(rows))),
            ("basis", "out-of-fold" if cross_fitted is not None else "in-sample"),
            ("probe_alpha", probe_alpha),
            ("probe_train_id", probe_train_id),
            ("dino_probe", "1" if visual_probe else "0"),
        ):
            booster.set_attr(**{key: value})
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_name(f"{MODEL_PATH.stem}.tmp.{os.getpid()}.json")
        model.save_model(str(tmp))
        tmp.replace(MODEL_PATH)
        database.set_preference_scores(conn, [
            (row["listing_id"], visual_of(row), normalize_qwen(row["qwen_direct_score"]))
            for row in rows
        ])
        logger.info("trained and saved %s on %d rated samples (basis=%s)", MODEL_PATH, len(y),
                    "out-of-fold" if cross_fitted is not None else "in-sample")
    finally:
        del model, X, y
        gc.collect()
