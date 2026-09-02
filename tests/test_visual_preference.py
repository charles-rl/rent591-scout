"""Layer 1 visual preference engine tests: centroid cosine (cold) -> Ridge probe (matured)."""

import numpy as np
import pytest

from src import database, scoring
from src import visual_preference as vp


@pytest.fixture(autouse=True)
def isolated_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(vp, "PROBE_PATH", tmp_path / "dino_probe.npz")
    monkeypatch.setattr(scoring, "RATED_THRESHOLD", 20)


def _axis(*idx) -> np.ndarray:
    v = np.zeros(vp.EMBED_DIM, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v / np.linalg.norm(v)


LIKE = _axis(0, 1)
DISLIKE = _axis(2, 3)


def _near(rng, base, noise=0.01) -> np.ndarray:
    v = base + rng.standard_normal(vp.EMBED_DIM).astype(np.float32) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed(conn, lid, vec, score):
    database.upsert_listing(conn, {"listing_id": lid, "title": lid,
                                   "dino_embedding": vec.astype(np.float32).tobytes()})
    assert database.rate_listing(conn, lid, score)


def test_cold_start_uses_liked_centroid_cosine(tmp_path):
    conn = database.connect(tmp_path / "centroid.db")
    rng = np.random.default_rng(1)
    for i in range(3):
        _seed(conn, f"L{i}", _near(rng, LIKE), 5.0)
    for i in range(2):
        _seed(conn, f"D{i}", _near(rng, DISLIKE), 1.0)
    assert database.get_rating_count(conn) == 5 <= scoring.RATED_THRESHOLD

    scorer = vp.VisualScorer.for_conn(conn)
    assert scorer.phase == "centroid"
    liked = scorer.score(_near(rng, LIKE))
    disliked = scorer.score(_near(rng, DISLIKE))
    assert liked > 0.8
    assert disliked < 0.3
    assert 0.0 <= disliked and liked <= 1.0
    assert not vp.PROBE_PATH.exists()  # no probe training during cold start
    conn.close()


def test_cold_start_without_likes_is_neutral(tmp_path):
    conn = database.connect(tmp_path / "no-likes.db")
    rng = np.random.default_rng(2)
    _seed(conn, "D0", _near(rng, DISLIKE), 2.0)
    assert vp.compute_visual_score(conn, _near(rng, DISLIKE)) == pytest.approx(0.5)
    conn.close()


def test_phase_switch_to_probe_above_threshold(tmp_path):
    conn = database.connect(tmp_path / "phase.db")
    rng = np.random.default_rng(3)
    for i in range(11):
        _seed(conn, f"L{i}", _near(rng, LIKE), 5.0)
        _seed(conn, f"D{i}", _near(rng, DISLIKE), 1.0)
    assert database.get_rating_count(conn) == 22 > scoring.RATED_THRESHOLD

    scorer = vp.VisualScorer.for_conn(conn)
    assert scorer.phase == "probe"
    liked = scorer.score(_near(rng, LIKE))
    disliked = scorer.score(_near(rng, DISLIKE))
    assert liked > disliked
    assert 0.0 <= disliked <= liked <= 1.0
    assert vp.PROBE_PATH.exists()  # trained lazily on first matured score
    conn.close()


def test_probe_roundtrip_persistence(tmp_path):
    conn = database.connect(tmp_path / "roundtrip.db")
    rng = np.random.default_rng(4)
    for i in range(22):
        _seed(conn, f"R{i}", _near(rng, LIKE if i % 2 else DISLIKE), 5.0 if i % 2 else 1.0)
    weights, bias = vp.fit_probe(conn)
    vp.save_probe(weights, bias)
    loaded = vp.load_probe()
    assert loaded is not None
    np.testing.assert_allclose(loaded[0], weights, rtol=1e-6)
    assert loaded[1] == pytest.approx(bias, abs=1e-5)
    vec = _near(rng, LIKE)
    assert vp.probe_value(vec, *loaded) == pytest.approx(vp.probe_value(vec, weights, bias))


def test_load_probe_rejects_missing_and_corrupt(tmp_path):
    assert vp.load_probe() is None
    vp.PROBE_PATH.write_bytes(b"garbage")
    assert vp.load_probe() is None
    np.savez(vp.PROBE_PATH, weights=np.zeros(13, dtype=np.float32), bias=np.float32(0.0))
    assert vp.load_probe() is None  # wrong dimensionality rejected


def test_visual_score_bounds_on_random_vectors(tmp_path):
    conn = database.connect(tmp_path / "bounds.db")
    rng = np.random.default_rng(5)
    for i in range(22):
        _seed(conn, f"R{i}", _near(rng, LIKE if i % 2 else DISLIKE), 5.0 if i % 2 else 1.0)
    scores = [vp.compute_visual_score(conn, rng.standard_normal(vp.EMBED_DIM).astype(np.float32))
              for _ in range(30)]
    assert all(0.0 <= s <= 1.0 for s in scores)
    conn.close()


def test_liked_centroid_ignores_negative_ratings(tmp_path):
    conn = database.connect(tmp_path / "liked-only.db")
    rng = np.random.default_rng(6)
    for i in range(4):
        _seed(conn, f"D{i}", _near(rng, DISLIKE), 3.0)  # below LIKED_MIN_SCORE=4.0
    assert vp.liked_centroid(conn) is None
    _seed(conn, "L0", _near(rng, LIKE), 4.0)
    centroid = vp.liked_centroid(conn)
    assert centroid is not None
    assert float(centroid @ LIKE) > 0.9
    conn.close()
