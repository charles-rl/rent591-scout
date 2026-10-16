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
    # No alpha monkeypatching here on purpose: this fixture must pass against the
    # SHIPPED defaults so a regression in ALPHA_HEAVY/ALPHA_SOFT is caught here
    # rather than hidden by a test-only override. Band + discrimination are
    # asserted directly in tests/test_probe_quality.py.
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
    # The hint is gated on bath_hint_ok (set by finalize_listing from
    # bathroom_probe.is_degenerate) so a collapsed probe never injects a constant.
    msgs = vision_llm.build_messages(
        {"title": "t", "bath_model_score": 2.3, "bath_hint_ok": True}, [], None)
    assert "2.3/5" in msgs[1]["content"]
    msgs2 = vision_llm.build_messages(
        {"title": "t", "bath_model_score": None, "bath_hint_ok": True}, [], None)
    assert "estimate" not in msgs2[1]["content"]
    # gate off -> no hint even when a score exists
    msgs3 = vision_llm.build_messages({"title": "t", "bath_model_score": 2.3}, [], None)
    assert "estimate" not in msgs3[1]["content"]


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


# ---------------------------------------------------------------------------
# Explicit detection outranks the centroid heuristic
# ---------------------------------------------------------------------------
def test_explicit_photo_indices_override_centroid_selection(bath_conn):
    """If detection confirmed no bathroom, the centroid must not invent one.

    Previously the centroid selector ran unconditionally, so a listing whose
    every photo Qwen labelled is_bathroom=0 still got a bath_model_score -- a
    score for a bathroom nobody can see.
    """
    centroid = bathroom_probe.bath_centroid(bath_conn)
    good = _rng_vec(50, _axis(0, 1, 2))
    # A photo the probe scores as a BAD bathroom, sitting at the index we assert on,
    # so photo_indices=[1] is pinned by the score band rather than by "> 0.0" (which
    # probe_value's [1,5] clip satisfied for ANY non-empty index).
    bad = _rng_vec(51, _axis(3, 4, 5))
    vecs = [good, bad, _rng_vec(52, _axis(0, 1, 2))]
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid, photo_indices=[0]) > 4.0
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid, photo_indices=[1]) < 2.0
    # detection naming no photo -> 0.0 regardless of what the centroid would pick
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid, photo_indices=[]) == 0.0
    # out-of-range indices are ignored rather than crashing
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid, photo_indices=[99]) == 0.0


def test_predict_listing_photo_indices_none_uses_centroid(bath_conn):
    """None keeps the old behaviour: detection did not run, centroid covers for it."""
    centroid = bathroom_probe.bath_centroid(bath_conn)
    vecs = [_rng_vec(50, _axis(0, 1, 2))]
    assert bathroom_probe.predict_listing(bath_conn, vecs, centroid, photo_indices=None) > 4.0


def test_predict_listing_authoritative_indices_survive_a_missing_centroid(bath_conn):
    """Confirmed bathrooms must score even when no bath-labelled photo exists anywhere.

    The centroid is only needed for the heuristic fallback; short-circuiting on
    `centroid is None` discarded authoritative detection results entirely.
    """
    w, b, _ = bathroom_probe.train_and_save(bath_conn)
    good = bathroom_probe.pool_unit(bathroom_probe._unit([_rng_vec(50, _axis(0, 1, 2))]))
    scored = bathroom_probe.probe_value(good, w, b)
    assert scored > 4.0
    assert bathroom_probe.select_bath_photos(None, []) == []


def test_clear_unsupported_resets_only_stale_scores(tmp_path, monkeypatch):
    """Rows that lost their photos, or that detection fully cleared, go to 0.0.

    Rows whose labels are still NULL are left alone: detection never ran for
    them, so the centroid selection is the only evidence available and 0.0 would
    wrongly claim "no bathroom photo" instead of "not labelled yet".
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bb", str(__import__("pathlib").Path(__file__).resolve().parents[1]
                  / "scripts" / "backfill_bathrooms.py"))
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)

    conn = database.connect(tmp_path / "clear.db")
    for lid in ("no-images", "all-zero", "unlabelled", "mixed", "genuine"):
        database.upsert_listing(conn, {"listing_id": lid, "title": lid, "price": 12000})
    for lid, rows in (
        ("no-images", []),
        ("all-zero", [{"ordinal": 0, "is_bathroom": 0}, {"ordinal": 1, "is_bathroom": 0}]),
        ("unlabelled", [{"ordinal": 0, "is_bathroom": None}, {"ordinal": 1, "is_bathroom": None}]),
        ("mixed", [{"ordinal": 0, "is_bathroom": 0}, {"ordinal": 1, "is_bathroom": None}]),
        ("genuine", [{"ordinal": 0, "is_bathroom": 0}, {"ordinal": 1, "is_bathroom": 1}]),
    ):
        # ONE call: replace_images DELETEs all rows for the listing, so calling it
        # per row left every listing with a single photo and the multi-photo
        # "every photo" predicate was never actually exercised.
        database.replace_images(conn, lid, [{**r, "image_url": "u", "image_path": "p"} for r in rows])
        count = conn.execute("SELECT COUNT(*) FROM listing_images WHERE listing_id=?",
                             (lid,)).fetchone()[0]
        assert count == len(rows), f"{lid}: expected {len(rows)} image rows, got {count}"
    conn.executemany("UPDATE listings SET bath_model_score=2.5 WHERE listing_id=?",
                     [(lid,) for lid in ("no-images", "all-zero", "unlabelled", "mixed", "genuine")])
    conn.commit()

    assert bb.clear_unsupported(conn) == 2
    got = dict(conn.execute("SELECT listing_id, bath_model_score FROM listings").fetchall())
    assert got["no-images"] == 0.0      # photos gone -> stale
    assert got["all-zero"] == 0.0       # detection explicitly cleared -> stale
    assert got["unlabelled"] == 2.5     # never labelled -> centroid is all we have
    assert got["mixed"] == 2.5          # detection ran but missed one photo -> keep
    assert got["genuine"] == 2.5        # real bathroom photo -> keep
    # idempotent
    assert bb.clear_unsupported(conn) == 0
    conn.close()


def test_replace_images_preserve_labels_keeps_prior_verdicts(tmp_path):
    """The image-queue drain and the duplicate short-circuit both pass rows with no
    `is_bathroom` key; writing those as NULL destroyed good labels on every cycle."""
    conn = database.connect(tmp_path / "preserve.db")
    database.upsert_listing(conn, {"listing_id": "P1", "title": "P1", "price": 12000})
    database.replace_images(conn, "P1", [
        {"ordinal": 0, "image_url": "u", "image_path": "p", "is_bathroom": 1},
        {"ordinal": 1, "image_url": "u", "image_path": "p", "is_bathroom": 0},
    ])
    # re-drain with fresh embeddings and NO is_bathroom key at all
    database.replace_images(conn, "P1", [
        {"ordinal": 0, "image_url": "u", "image_path": "p"},
        {"ordinal": 1, "image_url": "u", "image_path": "p"},
    ], preserve_labels=True)
    assert [r[0] for r in conn.execute(
        "SELECT is_bathroom FROM listing_images WHERE listing_id='P1' ORDER BY ordinal")] == [1, 0]
    # an explicit new verdict still wins
    database.replace_images(conn, "P1", [
        {"ordinal": 0, "image_url": "u", "image_path": "p", "is_bathroom": 0},
    ], preserve_labels=True)
    assert conn.execute(
        "SELECT is_bathroom FROM listing_images WHERE listing_id='P1' AND ordinal=0").fetchone()[0] == 0
    # without the flag, behaviour is unchanged (explicit None writes NULL)
    database.replace_images(conn, "P1", [{"ordinal": 0, "image_url": "u", "image_path": "p"}])
    assert conn.execute(
        "SELECT is_bathroom FROM listing_images WHERE listing_id='P1'").fetchone()[0] is None
    conn.close()


def test_replace_images_preserve_labels_matches_by_photo_identity(tmp_path):
    """A re-push that INSERTS a photo shifts every ordinal; ordinal-only matching
    would re-attach verdicts to the wrong photos and make the mislabel permanent."""
    conn = database.connect(tmp_path / "identity.db")
    database.upsert_listing(conn, {"listing_id": "P2", "title": "P2", "price": 12000})
    database.replace_images(conn, "P2", [
        {"ordinal": 0, "image_url": "https://img/a", "image_path": "a", "is_bathroom": 1},
        {"ordinal": 1, "image_url": "https://img/b", "image_path": "b", "is_bathroom": 0},
    ])
    # landlord prepends a new photo -> old ordinals shift up by one
    database.replace_images(conn, "P2", [
        {"ordinal": 0, "image_url": "https://img/new", "image_path": "n"},
        {"ordinal": 1, "image_url": "https://img/a", "image_path": "a"},
        {"ordinal": 2, "image_url": "https://img/b", "image_path": "b"},
    ], preserve_labels=True)
    got = {r[0]: r[1] for r in conn.execute(
        "SELECT image_url, is_bathroom FROM listing_images WHERE listing_id='P2'")}
    assert got == {"https://img/new": None, "https://img/a": 1, "https://img/b": 0}, (
        f"labels must follow the photo, not the ordinal; got {got}")
    conn.close()


def test_detect_chunk_raises_when_no_image_is_readable(monkeypatch):
    """An all-dead chunk is an infrastructure failure, not 'there is no bathroom'.

    Returning set() made the caller persist is_bathroom=0 for every photo and
    force bath_model_score to 0.0 -- permanently.
    """
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: None)
    with pytest.raises(ValueError):
        bathroom_detect.detect_chunk(["/a.webp", "/b.webp"])


def test_detect_chunk_raises_on_a_reply_without_bathroom_indices(monkeypatch):
    """A malformed model reply is not a verdict of 'no bathroom' either.

    parse_json returns {} (or a non-dict) on empty/malformed output; treating that
    as an empty hit set wrote is_bathroom=0 for the whole chunk, permanently.
    """
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: "b64")

    def ask(messages, timeout=None):
        return "{}"
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    with pytest.raises(ValueError):
        bathroom_detect.detect_chunk(["/a.webp", "/b.webp"])


def test_detect_chunk_maps_indices_through_unreadable_photos(monkeypatch):
    """The model only sees the READABLE photos, so its indices must be translated back.

    paths=[A(dead),B,C] -> the model sees [B,C]; index 0 means B, not A.
    `shown` reports the readable positions so the caller leaves A unlabelled.
    """
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: None if p == "/a.webp" else "b64")
    monkeypatch.setattr(vision_llm, "ask_ollama", lambda m, timeout=None: '{"bathroom_indices":[0]}')
    hits, shown = bathroom_detect.detect_chunk(["/a.webp", "/b.webp", "/c.webp"])
    assert hits == {1}
    assert shown == {1, 2}, "the dead photo was never shown and must not be reported as shown"


def test_detect_flags_leaves_unshown_photos_absent(monkeypatch):
    """A photo whose b64 failed inside a LIVE chunk was never shown to the model.

    Writing 0 for it recorded an authoritative 'no bathroom' the model never gave.
    """
    monkeypatch.setattr(vision_llm, "_image_b64", lambda p: None if p == "/a" else "b64")
    monkeypatch.setattr(vision_llm, "ask_ollama", lambda m, timeout=None: '{"bathroom_indices":[0]}')
    flags = bathroom_detect.detect_flags(["/a", "/b", "/c"], chunk=3)
    assert flags == {1: 1, 2: 0}, f"unreadable photo must stay absent, got {flags}"
    assert 0 not in flags, "the unshown photo's position must stay UNKNOWN (absent), not 0"


def test_detect_flags_leaves_failed_chunk_photos_absent(monkeypatch):
    """A timed-out chunk must not become a row of zeros."""
    calls = {"n": 0}

    def flaky(paths, chunk=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("timeout")
        return {0}, set(range(len(paths)))

    monkeypatch.setattr(bathroom_detect, "detect_chunk", flaky)
    flags = bathroom_detect.detect_flags(["/a", "/b"], chunk=1)
    assert flags == {1: 1}, "only the surviving chunk's position is labelled"
    assert 0 not in flags, "the failed chunk's position must stay UNKNOWN (absent), not 0"
