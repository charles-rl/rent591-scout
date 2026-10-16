"""Session 5: adaptive scoring router + Layer-3 fusion XGBoost head tests (offline, temp DBs)."""

import subprocess
import sys
from pathlib import Path

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
    assert vec.shape == (len(scoring.FEATURE_NAMES),) == (18,)
    assert vec[0] == pytest.approx(0.7)                      # dino_visual_score
    assert vec[1] == pytest.approx(0.5)                      # qwen_score = (3-1)/4
    np.testing.assert_allclose(vec[2:8], scoring.flag_vector(FLAGS_A, ["w"]))
    assert vec[8] == pytest.approx(np.log1p(12000), abs=1e-4)
    assert vec[9] == pytest.approx(8.0)
    assert vec[10] == pytest.approx(1500.0)                  # price per ping
    assert vec[11] == pytest.approx(5.0)
    assert vec[12] == 1.0 and vec[13] == 1.0                 # HIGH_ELEC_FEE / MANUAL_TRASH
    assert vec[14] == 0.0 and vec[15] == 0.0                 # NO_PETS / ELEC_EXTRA_HIGH_COST
    assert vec[16] == -1.0                                   # log_days_on_market sentinel (absent)
    empty = scoring.fusion_vector(0.0, None, None, None, None)
    assert empty.shape == (18,) and empty[1] == pytest.approx(0.5)  # neutral qwen_score
    assert empty[17] == 0.0  # bath_model_score absent -> 0, distinct from bad bathroom


def test_fusion_vector_days_on_market():
    # days_on_market is a pre-computed snapshot (ingestion.normalize_listing), not
    # raw refresh_time text -- scoring must not re-parse the time-relative form.
    listing = {"price": 12000, "area": 8.0, "days_on_market": 1.0}
    vec = scoring.fusion_vector(0.0, 3.0, FLAGS_B, [], listing)
    assert vec[16] == pytest.approx(np.log1p(1.0), abs=1e-4)


def test_days_on_market_sentinel_distinct_from_brand_new():
    absent = scoring.fusion_vector(0.0, 3.0, FLAGS_B, [], {"price": 12000, "area": 8.0})
    just_posted = scoring.fusion_vector(0.0, 3.0, FLAGS_B, [],
                                        {"price": 12000, "area": 8.0, "days_on_market": 0.0})
    # 0.0 age == log1p(0) == 0.0; "never parsed" must be a different value
    assert just_posted[16] == pytest.approx(0.0)
    assert absent[16] == -1.0
    assert absent[16] != just_posted[16]


def test_parse_days_on_market():
    from datetime import date

    today = date(2026, 9, 27)
    assert scoring.parse_days_on_market("此房屋在1天前發佈", today) == 1.0
    assert scoring.parse_days_on_market("此房屋在3小時前發佈", today) == pytest.approx(3 / 24)
    assert scoring.parse_days_on_market("此房屋在2天前更新", today) == 2.0
    assert scoring.parse_days_on_market("此房屋在9月7日發佈", today) == 20.0
    # '剛剛' = "just now" -> age 0, NOT unknown
    assert scoring.parse_days_on_market("此房屋在剛剛發佈", today) == 0.0
    # absolute date in the (future) this year -> last year
    assert scoring.parse_days_on_market("此房屋在11月20日發佈", today) > 300
    assert scoring.parse_days_on_market(None, today) is None
    assert scoring.parse_days_on_market("未知時間", today) is None
    # the search-list spelling uses 內, not 前 -- these were unparseable before
    assert scoring.parse_days_on_market("20分鐘內更新", today) == pytest.approx(20 / 1440)
    assert scoring.parse_days_on_market("4小時內更新", today) == pytest.approx(4 / 24)
    assert scoring.parse_days_on_market("昨日更新", today) == 1.0
    # simplified-Chinese units
    assert scoring.parse_days_on_market("此房屋在1周前发布", today) == 7.0
    assert scoring.parse_days_on_market("此房屋在2周前發佈", today) == 14.0


def test_parse_days_on_market_never_yields_a_negative_age():
    """A date after today must never produce a stored negative age.

    tabular_vector clamps with max(0.0, dom), so a stored negative age would become
    log1p(0) == 0.0 -- bit-identical to "brand new", the exact conflation the sentinel
    exists to prevent. The old `-1 <= age` bound let that value through. A date just
    one or two days ahead is clock/timezone skew (neither year's listing), so it
    returns None -> the -1.0 "unknown" sentinel; anything further ahead is read as
    last year's occurrence instead of being stored negative.
    """
    from datetime import date

    # Dec 30 seen on Dec 29: 1 day ahead is skew -> unknown, NOT -1.0 and NOT
    # "364 days ago" (that would report a fresh listing as year-old).
    assert scoring.parse_days_on_market("此房屋在12月30日發佈", date(2026, 12, 29)) is None
    # Dec 31 seen on Jan 2 is genuinely last year -> 2.0
    assert scoring.parse_days_on_market("此房屋在12月31日發佈", date(2026, 1, 2)) == 2.0
    for day in (1, 15, 28, 31):
        for month in (1, 6, 12):
            got = scoring.parse_days_on_market(
                f"此房屋在{month}月{day}日發佈", date(2026, 12, 31))
            assert got is None or got >= 0.0, f"{month}/{day} produced a negative age {got}"
    # 昨日 is a real 1-day age
    assert scoring.parse_days_on_market("昨日更新", date(2026, 12, 30)) == 1.0


def test_tabular_vector_treats_non_finite_days_on_market_as_unknown():
    """nan/inf would otherwise survive `max(0.0, dom)`: max(0.0, nan) is 0.0, i.e.
    "brand new", and log1p(inf) would blow up the head's predict."""
    i = scoring.FEATURE_NAMES.index("log_days_on_market")
    for bad in (float("nan"), float("inf"), float("-inf"), "not-a-number", None):
        vec = scoring.fusion_vector(0.0, 3.0, FLAGS_B, [],
                                    {"price": 12000, "area": 8.0, "days_on_market": bad})
        assert vec[i] == -1.0, f"days_on_market={bad!r} must encode as the -1.0 sentinel"


def test_relay_import_graph_is_lightweight():
    """GHA relay runner installs requirements.txt only (requests + Pillow): src.ingestion
    must import with every OTHER third-party package unavailable.

    Regression guard: a module-level `from .scoring import ...` in src/ingestion.py pulled
    numpy/xgboost into the relay graph and killed the cron relay SILENTLY (the workflow's
    self-test has continue-on-error and the scraper step only echoes a WARN), starving the
    local pipeline with no alert. Blocking a denylist of {numpy, xgboost, torch} would miss
    any future heavy import, so block everything outside the runner's actual dependency set.
    """
    allow = {"requests", "PIL", "urllib3", "certifi", "idna", "charset_normalizer"}
    code = (
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        f"ALLOW = {allow!r}\n"
        "class _Blocker(MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        top = name.split('.')[0]\n"
        "        if top in sys.stdlib_module_names or top in ALLOW or top == 'src':\n"
        "            return None\n"
        "        raise ImportError(f'{top} is not installed on the relay runner')\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import src.ingestion\n"
        "assert callable(src.ingestion.dump_relay_payloads)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       check=False, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr


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


# --------------------------------------------------------------------------- collapsed-output guard


def test_warn_if_collapsed_flags_constant_predictions(caplog):
    """A head leaning on collapsed features must be reported loudly, naming them."""
    n = len(scoring.FEATURE_NAMES)
    X = np.zeros((40, n), dtype=np.float32)
    X[:, 0] = 0.69                      # dino_visual_score: constant
    X[:, 1] = np.linspace(0.0, 1.0, 40)  # qwen_score: varies
    preds = {f"L{i}": 3.995 for i in range(40)}  # single plateau
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    assert any("collapsed score distribution" in r.message for r in caplog.records)
    assert any("dino_visual_score" in r.message for r in caplog.records)


def test_warn_if_collapsed_names_a_near_constant_split_column(caplog):
    """The real historical failure was std ~0.005-0.01, not exactly 0.

    The old `std < 1e-3` naming test could not see this band and printed
    "(none detected)" for exactly the failure it was written to catch.
    """
    n = len(scoring.FEATURE_NAMES)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, n)).astype(np.float32) * 0.5 + 0.3
    X[:, 0] = 0.69 + rng.standard_normal(40) * 0.004  # collapsed probe: std ~0.004
    assert 0.001 < float(X[:, 0].std()) < 0.02
    preds = {f"L{i}": 3.995 for i in range(40)}
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    msg = "\n".join(r.message for r in caplog.records)
    assert "collapsed score distribution" in msg
    assert "dino_visual_score" in msg, f"near-constant split column must be named; got: {msg}"


def test_warn_if_collapsed_names_nothing_when_the_head_is_at_its_limit(caplog):
    """A flat head is only blameable if the inputs are flat too."""
    n = len(scoring.FEATURE_NAMES)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((40, n)).astype(np.float32) * 0.5 + 0.3
    preds = {f"L{i}": 3.995 for i in range(40)}
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    msg = "\n".join(r.message for r in caplog.records)
    assert "collapsed score distribution" in msg
    assert "(none detected" in msg, f"healthy features must not be blamed; got: {msg}"


def test_warn_if_collapsed_ignores_the_distinct_clause_below_twenty_rows(caplog):
    """`distinct < 0.10*n` is unsatisfiable for n <= 10 (0.1*10 == 1.0 exactly), so
    at small n only the spread test may fire."""
    n = len(scoring.FEATURE_NAMES)
    X = np.random.default_rng(0).standard_normal((8, n)).astype(np.float32)
    preds = {f"L{i}": 1.0 + i * 0.09 for i in range(8)}  # 8 distinct values, tiny n
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    assert not any("collapsed score distribution" in r.message for r in caplog.records)


def test_warn_if_collapsed_measures_the_persisted_clipped_values(caplog):
    """Raw predictions all above 5.0 persist as a constant 5.0 column; the guard used
    to measure the raw values and stay silent."""
    n = len(scoring.FEATURE_NAMES)
    X = np.random.default_rng(0).standard_normal((40, n)).astype(np.float32)
    preds = {f"L{i}": 5.2 + i * 0.02 for i in range(40)}  # all clip to 5.0 on persist
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    assert any("collapsed score distribution" in r.message for r in caplog.records)


def test_warn_if_collapsed_quiet_when_scores_vary(caplog):
    n = len(scoring.FEATURE_NAMES)
    X = np.random.default_rng(0).standard_normal((40, n)).astype(np.float32)
    preds = {f"L{i}": 1.0 + i * 0.09 for i in range(40)}  # well spread
    with caplog.at_level("ERROR", logger="src.scoring"):
        scoring._warn_if_collapsed(preds, X)
    assert not any("collapsed score distribution" in r.message for r in caplog.records)
