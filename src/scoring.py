"""Cold-start router + XGBoost scoring head.

Phase 1 (<=RATED_THRESHOLD labeled samples): predicted_score = qwen_direct_score.
Phase 2 (>RATED_THRESHOLD): XGBRegressor over [DINO 768-d embedding; binary flags] -> user_score.
"""

from __future__ import annotations

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
_FLAG_ORDER = ["has_bathroom_img", "shower_sink_combo", "drainage_risk", "has_kitchen_sink", "has_exterior_window"]


def flag_vector(flags: dict | None, warnings: list | None) -> np.ndarray:
    flags = flags or {}
    vec = [1 if flags.get(k) else 0 for k in _FLAG_ORDER]
    vec.append(1 if warnings else 0)
    return np.asarray(vec, dtype=np.float32)


def _features(dino_blob, flags, warnings) -> np.ndarray:
    emb = np.frombuffer(dino_blob, dtype=np.float32) if dino_blob else np.zeros(768, dtype=np.float32)
    return np.concatenate([emb, flag_vector(flags, warnings)])


def predict_score(conn, dino_vec: np.ndarray, flags: dict, warnings: list,
                  qwen_direct_score: float) -> tuple[float, str]:
    rated = database.get_rating_count(conn)
    if rated <= RATED_THRESHOLD:
        return float(qwen_direct_score), "qwen"

    if not MODEL_PATH.exists():
        train_and_save(conn)

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    vec = np.concatenate([dino_vec.astype(np.float32), flag_vector(flags, warnings)])
    pred = float(model.predict(vec.reshape(1, -1))[0])
    return max(1.0, min(5.0, pred)), "xgboost"


def train_and_save(conn) -> None:
    rows = database.get_rated_samples(conn)
    if not rows:
        raise RuntimeError("no rated samples to train on")
    X, y = [], []
    for r in rows:
        try:
            flags = json.loads(r["qwen_vision_flags"]) if r["qwen_vision_flags"] else {}
            warnings = json.loads(r["qwen_warnings"]) if r["qwen_warnings"] else []
        except Exception:
            flags, warnings = {}, []
        X.append(_features(r["dino_embedding"], flags, warnings))
        y.append(float(r["user_score"]))
    model = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
    model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    logger.info("trained and saved %s on %d rated samples", MODEL_PATH, len(y))
