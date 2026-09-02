"""Session 5: adaptive scoring router + XGBoost head tests (offline, temp DBs only)."""

import numpy as np
import pytest

from src import database, scoring

FLAGS_A = {"has_bathroom_img": True, "shower_sink_combo": False, "drainage_risk": True,
           "has_kitchen_sink": True, "has_exterior_window": False}
FLAGS_B = {k: False for k in scoring._FLAG_ORDER}


@pytest.fixture(autouse=True)
def isolated_model(tmp_path, monkeypatch):
    """Never touch the real models/xgboost_head.json during tests; pin router threshold."""
    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "xgboost_head.json")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 20)


def _embedding(rng, scale=1.0) -> bytes:
    return (rng.standard_normal(scoring.EMBED_DIM).astype(np.float32) * scale).tobytes()


def _seed_listing(conn, lid, *, emb, flags, warnings=None, qwen=3.0, rated=None):
    database.upsert_listing(conn, {
        "listing_id": lid, "title": lid,
        "dino_embedding": emb,
        "qwen_vision_flags": flags,
        "qwen_warnings": warnings or [],
        "qwen_direct_score": qwen,
    })
    if rated is not None:
        assert database.rate_listing(conn, lid, rated)


def _seed_labeled_set(conn, n, seed=7):
    rng = np.random.default_rng(seed)
    for i in range(n):
        hot = i % 2 == 0
        _seed_listing(
            conn, f"R{i}",
            emb=_embedding(rng, scale=1.0 if hot else 0.5),
            flags=FLAGS_A if hot else FLAGS_B,
            warnings=["dirty grout"] if hot else [],
            qwen=4.5 if hot else 1.5,
            rated=4.5 if hot else 1.5,
        )


def test_flag_vector_layout_and_dim():
    vec = scoring.flag_vector(FLAGS_A, ["w"])
    assert vec.shape == (6,)
    assert vec.dtype == np.float32
    assert vec.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert scoring.flag_vector({}, []).tolist() == [0.0] * 6
    feats = scoring._features(_embedding(np.random.default_rng(0)), {"has_kitchen_sink": 1}, None)
    assert feats.shape == (scoring.EMBED_DIM + 6,)
    assert feats[scoring.EMBED_DIM + 2] == 0.0  # drainage_risk position preserved


def test_as_embedding_pads_and_truncates():
    rng = np.random.default_rng(1)
    short = rng.standard_normal(100, dtype=np.float32).tobytes()
    long = rng.standard_normal(900, dtype=np.float32).tobytes()
    assert scoring._as_embedding(short).shape == (scoring.EMBED_DIM,)
    assert scoring._as_embedding(long).shape == (scoring.EMBED_DIM,)
    assert scoring._as_embedding(None).shape == (scoring.EMBED_DIM,)
    arr = rng.standard_normal(scoring.EMBED_DIM).astype(np.float32)
    np.testing.assert_array_equal(scoring._as_embedding(arr), arr)
    np.testing.assert_array_equal(scoring._as_embedding(b"\x01\x02\x03"), np.zeros(scoring.EMBED_DIM))


def test_cold_start_returns_qwen_score(tmp_path):
    conn = database.connect(tmp_path / "cold.db")
    rng = np.random.default_rng(2)
    _seed_labeled_set(conn, 5)
    assert database.get_rating_count(conn) == 5
    pred, source = scoring.predict_score(conn, np.frombuffer(_embedding(rng), dtype=np.float32),
                                         FLAGS_A, [], 3.7)
    assert source == "qwen"
    assert pred == pytest.approx(3.7)
    assert not scoring.MODEL_PATH.exists()
    conn.close()


def test_cold_start_zero_ratings(tmp_path):
    conn = database.connect(tmp_path / "zero.db")
    rng = np.random.default_rng(3)
    pred, source = scoring.predict_score(conn, np.frombuffer(_embedding(rng), dtype=np.float32),
                                         FLAGS_B, ["mold smell"], 2.2)
    assert (source, pred) == ("qwen", pytest.approx(2.2))
    conn.close()


def test_xgboost_trains_on_22_samples_and_predicts(tmp_path):
    conn = database.connect(tmp_path / "warm.db")
    _seed_labeled_set(conn, 22)
    assert database.get_rating_count(conn) == 22 > scoring.RATED_THRESHOLD

    scoring.train_and_save(conn)
    assert scoring.MODEL_PATH.exists()

    rng = np.random.default_rng(99)
    pred, source = scoring.predict_score(conn, np.frombuffer(_embedding(rng), dtype=np.float32),
                                         FLAGS_A, ["w"], 3.0)
    assert source == "xgboost"
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 5.0
    conn.close()


def test_predict_autotrains_when_model_missing(tmp_path):
    conn = database.connect(tmp_path / "autotrain.db")
    _seed_labeled_set(conn, 25)
    assert not scoring.MODEL_PATH.exists()
    pred, source = scoring.predict_score(conn, np.zeros(scoring.EMBED_DIM, dtype=np.float32),
                                         FLAGS_B, [], 3.0)
    assert source == "xgboost"
    assert scoring.MODEL_PATH.exists()
    assert 1.0 <= pred <= 5.0
    conn.close()


def test_model_save_load_roundtrip_matches(tmp_path):
    conn = database.connect(tmp_path / "roundtrip.db")
    _seed_labeled_set(conn, 22)
    scoring.train_and_save(conn)
    model = scoring._load_trained_model(conn)
    rng = np.random.default_rng(11)
    feats = np.stack([scoring._features(_embedding(rng), FLAGS_A, ["w"]) for _ in range(8)])
    reloaded = scoring._load_trained_model(conn)
    np.testing.assert_allclose(model.predict(feats), reloaded.predict(feats), rtol=1e-6)


def test_predict_falls_back_to_qwen_on_broken_model(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "broken.db")
    _seed_labeled_set(conn, 22)
    scoring.train_and_save(conn)
    scoring.MODEL_PATH.write_bytes(b"not a valid xgboost model")
    pred, source = scoring.predict_score(conn, np.zeros(scoring.EMBED_DIM, dtype=np.float32),
                                         FLAGS_A, [], 4.1)
    assert (source, pred) == ("qwen", pytest.approx(4.1))
    conn.close()


def test_score_all_unrated_cold_backfills_qwen(tmp_path):
    conn = database.connect(tmp_path / "backfill-cold.db")
    _seed_labeled_set(conn, 3)
    _seed_listing(conn, "U1", emb=_embedding(np.random.default_rng(4)), flags=FLAGS_A, qwen=4.2)
    _seed_listing(conn, "U2", emb=None, flags=FLAGS_B, qwen=2.4)
    assert scoring.score_all_unrated(conn) == 2
    rows = {r["listing_id"]: (r["predicted_score"], r["score_source"]) for r in
            conn.execute("SELECT listing_id, predicted_score, score_source FROM listings")}
    assert rows["U1"] == (pytest.approx(4.2), "qwen")
    assert rows["U2"] == (pytest.approx(2.4), "qwen")
    # Rated rows are excluded from backfill (predicted_score stays NULL).
    assert rows["R0"] == (None, None)
    conn.close()


def test_score_all_unrated_warm_backfills_xgboost(tmp_path):
    conn = database.connect(tmp_path / "backfill-warm.db")
    _seed_labeled_set(conn, 22)
    rng = np.random.default_rng(5)
    _seed_listing(conn, "U1", emb=_embedding(rng), flags=FLAGS_A, warnings=["w"], qwen=3.3)
    _seed_listing(conn, "U2", emb=_embedding(rng, scale=0.5), flags=FLAGS_B, qwen=3.3)
    written = scoring.score_all_unrated(conn)
    assert written == 2
    rows = {r["listing_id"]: (r["predicted_score"], r["score_source"]) for r in
            conn.execute("SELECT listing_id, predicted_score, score_source FROM listings")}
    assert rows["U1"][1] == "xgboost" and 1.0 <= rows["U1"][0] <= 5.0
    assert rows["U2"][1] == "xgboost" and 1.0 <= rows["U2"][0] <= 5.0
    # Higher-signal (hot-flagged) listing should score above the cold one on trained data.
    assert rows["U1"][0] > rows["U2"][0]
    conn.close()


def test_train_requires_samples(tmp_path):
    conn = database.connect(tmp_path / "empty.db")
    with pytest.raises(RuntimeError):
        scoring.train_and_save(conn)
    conn.close()


def test_train_survives_malformed_flags_json(tmp_path):
    conn = database.connect(tmp_path / "malformed.db")
    _seed_labeled_set(conn, 21)
    conn.execute("UPDATE listings SET qwen_vision_flags='{broken', qwen_warnings='[:' WHERE listing_id='R0'")
    conn.commit()
    scoring.train_and_save(conn)
    assert scoring.MODEL_PATH.exists()
    conn.close()
