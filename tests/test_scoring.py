"""Session 5: adaptive scoring router + Layer-3 fusion XGBoost head tests (offline, temp DBs)."""

import numpy as np
import pytest

from src import database, scoring, visual_preference

FLAGS_A = {"has_bathroom_img": True, "shower_sink_combo": False, "drainage_risk": True,
           "has_kitchen_sink": True, "has_exterior_window": False}
FLAGS_B = {k: False for k in scoring._FLAG_ORDER}


@pytest.fixture(autouse=True)
def isolated_model(tmp_path, monkeypatch):
    """Never touch the real models/ artifacts during tests; pin router threshold."""
    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "xgboost_head.json")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 20)
    monkeypatch.setattr(visual_preference, "PROBE_PATH", tmp_path / "dino_probe.npz")


def _axis(*idx) -> np.ndarray:
    v = np.zeros(scoring.EMBED_DIM, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v / np.linalg.norm(v)


BASE_LIKE = _axis(0, 1)
BASE_DISLIKE = _axis(2, 3)


def _embedding(rng, scale=1.0) -> bytes:
    return (rng.standard_normal(scoring.EMBED_DIM).astype(np.float32) * scale).tobytes()


def _structured(rng, base) -> bytes:
    v = base + rng.standard_normal(scoring.EMBED_DIM).astype(np.float32) * 0.01
    return (v / np.linalg.norm(v)).astype(np.float32).tobytes()


def _seed_listing(conn, lid, *, emb, flags, warnings=None, qwen=3.0, rated=None, **tabular):
    database.upsert_listing(conn, {
        "listing_id": lid, "title": lid,
        "dino_embedding": emb,
        "qwen_vision_flags": flags,
        "qwen_warnings": warnings or [],
        "qwen_direct_score": qwen,
        **tabular,
    })
    if rated is not None:
        assert database.rate_listing(conn, lid, rated)


def _seed_labeled_set(conn, n):
    """Half hot / half cold. All embeddings are identical (Layer 1 carries no class
    signal), so the tree must split on the robust qwen_score/flag margins — Layer-1
    discriminative power is unit-tested in tests/test_visual_preference.py instead."""
    for i in range(n):
        hot = i % 2 == 0
        # Varied qwen/rating inside each class keeps the split margin wide instead of
        # razor-edged at a single value.
        score = (4.5 if i % 4 == 0 else 4.0) if hot else (1.5 if i % 4 == 0 else 2.0)
        _seed_listing(
            conn, f"R{i}",
            emb=BASE_LIKE.astype(np.float32).tobytes(),
            flags=FLAGS_A if hot else FLAGS_B,
            warnings=["dirty grout"] if hot else [],
            qwen=score,
            rated=score,
        )


def test_flag_vector_layout_and_dim():
    vec = scoring.flag_vector(FLAGS_A, ["w"])
    assert vec.shape == (6,)
    assert vec.dtype == np.float32
    assert vec.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    assert scoring.flag_vector({}, []).tolist() == [0.0] * 6


def test_fusion_vector_layout():
    listing = {"price": 12000, "area": 8.0, "floor": "5樓", "shape": "公寓",
               "description": "電費6元/度，需追垃圾車"}
    vec = scoring.fusion_vector(0.7, 3.0, FLAGS_A, ["w"], listing)
    assert vec.shape == (len(scoring.FEATURE_NAMES),) == (16,)
    assert vec[0] == pytest.approx(0.7)                      # dino_visual_score
    assert vec[1] == pytest.approx(0.5)                      # qwen_score = (3-1)/4
    np.testing.assert_allclose(vec[2:8], scoring.flag_vector(FLAGS_A, ["w"]))
    assert vec[8] == pytest.approx(np.log1p(12000), abs=1e-4)
    assert vec[9] == pytest.approx(8.0)
    assert vec[10] == pytest.approx(1500.0)                  # price per ping
    assert vec[11] == pytest.approx(5.0)
    assert vec[12] == 1.0 and vec[13] == 1.0                 # HIGH_ELEC_FEE / MANUAL_TRASH
    assert vec[14] == 0.0 and vec[15] == 0.0                 # NO_PETS / ELEC_EXTRA_HIGH_COST
    empty = scoring.fusion_vector(0.0, None, None, None, None)
    assert empty.shape == (16,) and empty[1] == pytest.approx(0.5)  # neutral qwen_score


def test_parse_floor_number():
    assert scoring.parse_floor_number("5樓") == 5.0
    assert scoring.parse_floor_number(3) == 3.0
    assert scoring.parse_floor_number("地下2樓") == -2.0
    assert scoring.parse_floor_number("B1") == -1.0
    assert scoring.parse_floor_number(None) == 0.0


def test_no_pets_detected_from_facilities_and_wording():
    assert "NO_PETS" in scoring.heuristic_penalties({"facilities": ["不可養寵物"]})
    assert "NO_PETS" in scoring.heuristic_penalties({"description": "為維護品質不可養寵物和吸煙"})
    assert "NO_PETS" in scoring.heuristic_penalties({"description": "不開放寵物"})
    assert "NO_PETS" not in scoring.heuristic_penalties({"facilities": ["可養寵物"]})


def test_elec_rate_reversed_wording_flags_high_fee():
    assert "HIGH_ELEC_FEE" in scoring.heuristic_penalties({"description": "獨立電表（電費1度6元）"})
    assert "HIGH_ELEC_FEE" not in scoring.heuristic_penalties({"description": "電費1度4元"})


def test_elec_extra_high_cost_rule():
    base = {"description": "獨立套房"}
    assert "ELEC_EXTRA_HIGH_COST" in scoring.heuristic_penalties({**base, "price": 16000})
    assert "ELEC_EXTRA_HIGH_COST" not in scoring.heuristic_penalties({**base, "price": 15000})
    assert "ELEC_EXTRA_HIGH_COST" not in scoring.heuristic_penalties(
        {**base, "price": 16000, "contain_cost": ["含電費"]})
    assert "ELEC_EXTRA_HIGH_COST" not in scoring.heuristic_penalties(
        {**base, "price": 16000, "description": "租金含電費"})
    assert "ELEC_EXTRA_HIGH_COST" in scoring.heuristic_penalties(
        {**base, "price": 16000, "contain_cost": "[{\"name\": \"含清潔費\"}]"})


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
    # Layer-1 Ridge probe must be trained and persisted alongside the XGBoost head.
    assert visual_preference.PROBE_PATH.exists()

    rng = np.random.default_rng(99)
    pred, source = scoring.predict_score(conn, np.frombuffer(_embedding(rng), dtype=np.float32),
                                         FLAGS_A, ["w"], 3.0, {"price": 13000, "area": 8.0})
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
    feats = np.stack([
        scoring.fusion_vector(float(rng.random()), float(rng.uniform(1, 5)), FLAGS_A, ["w"],
                              {"price": int(rng.uniform(10000, 17000)), "area": 8.0})
        for _ in range(8)
    ])
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
    # Cold start still produces Layer-1/2 scalars (centroid cosine + normalized qwen).
    prefs = {r["listing_id"]: (r["dino_visual_score"], r["qwen_score"]) for r in conn.execute(
        "SELECT listing_id, dino_visual_score, qwen_score FROM listings")}
    assert prefs["U1"][1] == pytest.approx((4.2 - 1) / 4)
    assert 0.0 <= prefs["U1"][0] <= 1.0
    conn.close()


def test_score_all_unrated_warm_backfills_xgboost(tmp_path):
    conn = database.connect(tmp_path / "backfill-warm.db")
    _seed_labeled_set(conn, 22)
    rng = np.random.default_rng(5)
    _seed_listing(conn, "U1", emb=_structured(rng, BASE_LIKE), flags=FLAGS_A, warnings=["w"], qwen=4.4)
    _seed_listing(conn, "U2", emb=_structured(rng, BASE_DISLIKE), flags=FLAGS_B, qwen=1.6)
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
