"""Bathroom layer tests: photo labelling (mocked Ollama), ridge probe, fusion dim, hint."""

import numpy as np
import pytest

from src import bathroom_detect, bathroom_probe, database, scoring, vision_llm


def _axis(*idx) -> np.ndarray:
    v = np.zeros(bathroom_probe.EMBED_DIM, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v / np.linalg.norm(v)


def _rng_vec(seed, base, noise=0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + rng.standard_normal(bathroom_probe.EMBED_DIM).astype(np.float32) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def bath_conn(tmp_path, monkeypatch):
    monkeypatch.setattr(bathroom_probe, "PROBE_PATH", tmp_path / "bath_probe.npz")
    # Unit-vector features have tiny covariance; shrinkage must be small enough
    # for the synthetic two-cluster separation to be learnable at all.
    monkeypatch.setattr(bathroom_probe, "ALPHA_HEAVY", 0.1)
    monkeypatch.setattr(bathroom_probe, "ALPHA_SOFT", 0.01)
    conn = database.connect(tmp_path / "bath.db")
    good, bad = _axis(0, 1, 2), _axis(3, 4, 5)
    for i in range(12):
        lid = f"L{i}"
        bath = 5.0 if i % 2 == 0 else 1.0
        database.upsert_listing(conn, {"listing_id": lid, "title": lid, "price": 12000})
        database.rate_listing(conn, lid, 4.0, bath)
        bath_vec = good if i % 2 == 0 else bad
        rows = [
            {"ordinal": 0, "image_path": f"/p/{lid}_0", "is_bathroom": 1,
             "dino_embedding": _rng_vec(i * 7, bath_vec).astype(np.float32).tobytes()},
            {"ordinal": 1, "image_path": f"/p/{lid}_1", "is_bathroom": 0,
             "dino_embedding": _rng_vec(i * 7 + 1, _axis(7)).astype(np.float32).tobytes()},
        ]
        database.replace_images(conn, lid, rows)
    return conn


# --------------------------------------------------------------------------- probe


def test_alpha_schedule_softens_after_threshold():
    assert bathroom_probe.alpha_for(15) == bathroom_probe.ALPHA_HEAVY
    assert bathroom_probe.alpha_for(bathroom_probe.SOFT_N) == bathroom_probe.ALPHA_HEAVY
    assert bathroom_probe.alpha_for(bathroom_probe.SOFT_N + 1) == bathroom_probe.ALPHA_SOFT
    assert bathroom_probe.ALPHA_SOFT < bathroom_probe.ALPHA_HEAVY


def test_probe_learns_good_and_bad_bathrooms(bath_conn):
    w, b, n = bathroom_probe.train_and_save(bath_conn)
    assert n == 12
    good = bathroom_probe.pool_unit(bathroom_probe._unit(
        [_rng_vec(0 * 7, _axis(0, 1, 2))]))
    bad = bathroom_probe.pool_unit(bathroom_probe._unit(
        [_rng_vec(1 * 7, _axis(3, 4, 5))]))
    assert bathroom_probe.probe_value(good, w, b) > 4.0
    assert bathroom_probe.probe_value(bad, w, b) < 2.0


def test_probe_absent_neutral_and_below_min(bath_conn, tmp_path):
    assert bathroom_probe.load_probe() is None
    empty = database.connect(tmp_path / "empty.db")
    assert bathroom_probe.train_if_stale(empty) is None  # below MIN_BATH_RATED -> None


def test_stale_probe_retrains_when_labels_grow(bath_conn):
    _, _, n = bathroom_probe.train_and_save(bath_conn)
    database.upsert_listing(bath_conn, {"listing_id": "X", "title": "X"})
    database.rate_listing(bath_conn, "X", 3.0, 3.0)
    database.replace_images(bath_conn, "X", [{
        "ordinal": 0, "image_path": "/p/x0", "is_bathroom": 1,
        "dino_embedding": _rng_vec(99, _axis(0, 1, 2)).astype(np.float32).tobytes(),
    }])
    _, _, n2 = bathroom_probe.train_if_stale(bath_conn)
    assert n2 == n + 1


def test_predict_listing_no_bath_photo_is_zero(bath_conn):
    bathroom_probe.train_and_save(bath_conn)
    centroid = bathroom_probe.bath_centroid(bath_conn)
    assert centroid is not None
    score = bathroom_probe.predict_listing(bath_conn, [_rng_vec(50, _axis(7))], centroid)
    assert score == 0.0  # no photo passes the centroid gate -> "no bathroom info"


def test_predict_listing_finds_good_bathroom(bath_conn):
    bathroom_probe.train_and_save(bath_conn)
    centroid = bathroom_probe.bath_centroid(bath_conn)
    vecs = [_rng_vec(60, _axis(7)), _rng_vec(61, _axis(0, 1, 2))]
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid) > 4.0


def test_select_bath_photos_gate(bath_conn):
    centroid = _axis(0, 1, 2)
    assert bathroom_probe.select_bath_photos(centroid, []) == []
    assert bathroom_probe.select_bath_photos(None, [_axis(0, 1, 2)]) == []
    assert bathroom_probe.select_bath_photos(centroid, [_axis(0, 1, 2), _axis(9)]) == [0]


# --------------------------------------------------------------------------- detect


def _fake_ask(responses):
    calls = {"n": 0}

    def ask(messages, timeout=0):
        out = responses[calls["n"]]
        calls["n"] += 1
        return out
    return ask


def test_detect_flags_maps_and_clamps(monkeypatch):
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: "x")
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: "x")
    monkeypatch.setattr(vision_llm, "ask_ollama", _fake_ask([
        '{"bathroom_indices":[0,2,99]}', '{"bathroom_indices":[0]}',
    ]))
    flags = bathroom_detect.detect_flags(["a", "b", "c", "d"], chunk=3)
    assert flags == {0: 1, 1: 0, 2: 1, 3: 1}


def test_detect_flags_all_fail_returns_none(monkeypatch):
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: "x")

    def boom(messages, timeout=0):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(vision_llm, "ask_ollama", boom)
    assert bathroom_detect.detect_flags(["a", "b"]) is None


def test_detect_flags_partial_chunk_failure(monkeypatch):
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: "x")
    seq = [RuntimeError("dead"), '{"bathroom_indices":[0]}']

    def flaky(messages, timeout=0):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    monkeypatch.setattr(vision_llm, "ask_ollama", flaky)
    flags = bathroom_detect.detect_flags(["a", "b", "c"], chunk=2)
    assert flags == {2: 1}  # first chunk unlabeled, second labelled


# --------------------------------------------------------------------------- plumbing


def test_bath_hint_reaches_qwen_messages():
    msgs = vision_llm.build_messages({"title": "t", "bath_model_score": 2.3}, [], None)
    assert "2.3/5" in msgs[1]["content"]
    msgs2 = vision_llm.build_messages({"title": "t", "bath_model_score": None}, [], None)
    assert "estimate" not in msgs2[1]["content"]


def test_normalize_bath_scale():
    assert scoring.normalize_bath(None) == 0.0
    assert scoring.normalize_bath(0) == 0.0          # absent stays distinct from bad
    assert scoring.normalize_bath(1.0) == pytest.approx(0.2)
    assert scoring.normalize_bath(5.0) == pytest.approx(1.0)


def test_fusion_vector_has_bath_dim():
    vec = scoring.fusion_vector(0.5, 3.0, {}, [], {"bath_model_score": 4.0})
    assert vec.shape == (len(scoring.FEATURE_NAMES),)
    assert vec[-1] == pytest.approx(0.8)
    assert scoring.fusion_vector(0.5, 3.0, {}, [], None)[-1] == 0.0


def test_is_bathroom_roundtrip(tmp_path):
    conn = database.connect(tmp_path / "m.db")
    assert "is_bathroom" in {r[1] for r in conn.execute("PRAGMA table_info(listing_images)")}
    database.upsert_listing(conn, {"listing_id": "A", "title": "A"})
    database.replace_images(conn, "A", [
        {"ordinal": 0, "image_path": "/a", "is_bathroom": 1},
        {"ordinal": 1, "image_path": "/b"},
    ])
    rows = {r["ordinal"]: r["is_bathroom"] for r in conn.execute(
        "SELECT ordinal, is_bathroom FROM listing_images WHERE listing_id='A'")}
    assert rows == {0: 1, 1: None}
