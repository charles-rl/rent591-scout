"""Cold-start router + XGBoost scoring head.

Phase 1 (<=RATED_THRESHOLD labeled samples): predicted_score = qwen_direct_score.
Phase 2 (>RATED_THRESHOLD): XGBRegressor over [DINO 768-d embedding; binary flags] -> user_score.
Feature vector layout: 768 float32 DINOv3 dims + 5 vision flags + 1 has-warnings bit (774).
"""

from __future__ import annotations

import gc
import json
import logging
import os
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


def _row_features(row) -> np.ndarray:
    flags = _load_json(row["qwen_vision_flags"], {})
    warnings = _load_json(row["qwen_warnings"], [])
    return np.concatenate([_as_embedding(row["dino_embedding"]), flag_vector(flags, warnings)])


def _features(dino_blob, flags, warnings) -> np.ndarray:
    return np.concatenate([_as_embedding(dino_blob), flag_vector(flags, warnings)])


def _load_trained_model(conn) -> xgb.XGBRegressor:
    if not MODEL_PATH.exists():
        train_and_save(conn)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def predict_score(conn, dino_vec: np.ndarray, flags: dict, warnings: list,
                  qwen_direct_score: float) -> tuple[float, str]:
    rated = database.get_rating_count(conn)
    if rated <= RATED_THRESHOLD:
        return float(qwen_direct_score), "qwen"

    model = None
    try:
        model = _load_trained_model(conn)
        vec = np.concatenate([_as_embedding(dino_vec), flag_vector(flags, warnings)])
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
    rows = database.get_scoring_rows(conn)
    if not rows:
        return 0

    rated = database.get_rating_count(conn)
    preds: dict[str, float] = {}
    model = None
    if rated > RATED_THRESHOLD:
        try:
            model = _load_trained_model(conn)
            X = np.stack([_row_features(r) for r in rows])
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
    logger.info("scored %d unrated listings", len(updates))
    return len(updates)


def train_and_save(conn) -> None:
    rows = database.get_rated_samples(conn)
    if not rows:
        raise RuntimeError("no rated samples to train on")
    X = np.stack([_row_features(r) for r in rows])
    y = np.asarray([float(r["user_score"]) for r in rows], dtype=np.float32)
    model = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
    try:
        model.fit(X, y)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".tmp.json")
        model.save_model(str(tmp))
        tmp.replace(MODEL_PATH)
        logger.info("trained and saved %s on %d rated samples", MODEL_PATH, len(y))
    finally:
        del model, X, y
        gc.collect()
