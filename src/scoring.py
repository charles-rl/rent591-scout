"""Layer 3: feature fusion + XGBoost scoring head.

Phase 1 (<=RATED_THRESHOLD labeled samples): predicted_score = qwen_direct_score.
Phase 2 (>RATED_THRESHOLD): XGBRegressor over the compressed fusion vector -> user_score:
[dino_visual_score (Layer 1 scalar); qwen_score = (qwen_direct_score-1)/4;
5 vision flags + 1 has-warnings bit; tabular: log1p price / area_ping / price_per_ping /
floor_num; HIGH_ELEC_FEE; MANUAL_TRASH; NO_PETS; ELEC_EXTRA_HIGH_COST;
bath_model_score (bathroom probe 0-5 / 5, 0 = no bathroom photo) ] (FEATURE_NAMES).
The raw 768-d DINOv3 vector is
compressed to dino_visual_score by src/visual_preference.py, never concatenated directly.

Also hosts the Stage-3 deterministic penalty engine (docs/591research.md §4):
baseline 100 minus regex-extracted red flags. Penalties fire only on positive
textual evidence — ambiguous listings stay penalty-free and go to the user.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import xgboost as xgb

from . import database

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
    "bath_model_score",
]


def normalize_bath(bath_model_score) -> float:
    """Bathroom probe score 1-5 -> [0,1]; 0.0 when absent (no bathroom photo)."""
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
    vec = [
        float(np.log1p(price)),
        area,
        per_ping,
        parse_floor_number(listing.get("floor")),
        1.0 if "HIGH_ELEC_FEE" in penalty_flags else 0.0,
        1.0 if "MANUAL_TRASH" in penalty_flags else 0.0,
        1.0 if "NO_PETS" in penalty_flags else 0.0,
        1.0 if "ELEC_EXTRA_HIGH_COST" in penalty_flags else 0.0,
    ]
    return np.asarray(vec, dtype=np.float32)


def fusion_vector(visual_score: float, qwen_direct_score, flags: dict | None,
                  warnings: list | None, listing: dict | None) -> np.ndarray:
    """Layer 1 scalar + Layer 2 scalar + vision flags + tabular + bathroom probe -> XGB input."""
    return np.concatenate([
        np.asarray([visual_score, normalize_qwen(qwen_direct_score)], dtype=np.float32),
        flag_vector(flags, warnings),
        tabular_vector(listing),
        np.asarray([normalize_bath((listing or {}).get("bath_model_score"))], dtype=np.float32),
    ])


def _row_listing(row) -> dict:
    return {
        "price": row["price"], "area": row["area"], "floor": row["floor"],
        "shape": row["shape"], "tags": _load_json(row["tags"], []),
        "facilities": _load_json(row["facilities"], []),
        "contain_cost": _load_json(row["contain_cost"], []),
        "description": row["description"],
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


def _load_trained_model(conn) -> xgb.XGBRegressor:
    if MODEL_PATH.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        if model.get_booster().num_features() != len(FEATURE_NAMES):
            logger.warning("saved model has %d features, FEATURE_NAMES has %d -> retraining",
                           model.get_booster().num_features(), len(FEATURE_NAMES))
            train_and_save(conn)
            model.load_model(str(MODEL_PATH))
        return model
    train_and_save(conn)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def predict_score(conn, dino_vec: np.ndarray, flags: dict, warnings: list,
                  qwen_direct_score: float, listing: dict | None = None) -> tuple[float, str]:
    """Return (predicted_score 1-5, score_source 'qwen'|'xgboost')."""
    from . import visual_preference

    rated = database.get_rating_count(conn)
    if rated <= RATED_THRESHOLD:
        return float(qwen_direct_score), "qwen"

    model = None
    try:
        visual = visual_preference.VisualScorer.for_conn(conn).score(_as_embedding(dino_vec))
        model = _load_trained_model(conn)
        vec = fusion_vector(visual, qwen_direct_score, flags, warnings, listing)
        pred = float(model.predict(vec.reshape(1, -1))[0])
        return max(1.0, min(5.0, pred)), "xgboost"
    except Exception:
        logger.exception("xgboost prediction failed -> falling back to qwen_direct_score")
        return float(qwen_direct_score), "qwen"
    finally:
        del model
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
            preds = {r["listing_id"]: float(p) for r, p in zip(rows, model.predict(X))}
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
        visual_preference.save_probe(*visual_probe)
    except Exception:
        logger.exception("Layer-1 probe training failed; falling back to liked centroid features")
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

    X = np.stack([_row_features(r, visual_of(r)) for r in rows])
    model = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
    try:
        model.fit(X, y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".tmp.json")
        model.save_model(str(tmp))
        tmp.replace(MODEL_PATH)
        database.set_preference_scores(conn, [
            (row["listing_id"], visual_of(row), normalize_qwen(row["qwen_direct_score"]))
            for row in rows
        ])
        logger.info("trained and saved %s on %d rated samples", MODEL_PATH, len(y))
    finally:
        del model, X, y
        gc.collect()
