"""Hybrid orchestration (main.run_incoming): offline queue+alert vs live drain+vision+notify."""

import io
import json
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

import main
from src import bathroom_probe, database, ingestion

PROXY = "http://127.0.0.1:8999"
VISION_OK = {
    "qwen_warnings": ["浴室無窗"],
    "vision_flags": {"has_bathroom_img": True, "shower_sink_combo": False,
                     "drainage_risk": False, "has_kitchen_sink": True,
                     "has_exterior_window": False},
    "qwen_direct_score": 4.5,
}


def _write_payload(incoming_dir, lid, image_urls, sha="sha-A"):
    listings = incoming_dir / "listings"
    listings.mkdir(parents=True, exist_ok=True)
    payload = {
        "listing_id": lid,
        "detail_failed": False,
        "payload_sha256": sha,
        "image_urls": image_urls,
        "images": [],
        "raw_search": {"id": int(lid), "title": f"Listing {lid}", "price": "15000",
                       "kind_name": "獨立套房", "area": 8.0,
                       "floor_name": "頂樓", "photoList": list(image_urls)},
        "raw_metadata": {"status": "open", "title": f"Listing {lid}",
                         "remark": {"content": "水電另計"}},
    }
    (listings / f"{lid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_webp(incoming_dir, lid, ordinal=0):
    d = incoming_dir / "images" / lid
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 200, 30)).save(buf, "WEBP")
    (d / f"{ordinal:02d}.webp").write_bytes(buf.getvalue())


@pytest.fixture
def env(tmp_path, monkeypatch):
    real_connect = database.connect
    db_path = tmp_path / "apts.db"
    monkeypatch.setattr(main.database, "connect",
                        lambda path=None: real_connect(path or db_path))
    monkeypatch.setattr(main.vision_llm, "analyze_listing",
                        lambda listing, image_rows, bullets: dict(VISION_OK))
    monkeypatch.setattr(main.bathroom_detect, "detect_flags",
                        lambda paths, chunk=None: None)
    monkeypatch.setattr(main.deduplication, "embed_image_rows", lambda rows: {})
    calls = {"proxy_alert": [], "ntfy": []}
    monkeypatch.setattr(main.notifier, "send_proxy_request_alert",
                        lambda n, proxy=None: calls["proxy_alert"].append((n, proxy)) or True)
    monkeypatch.setattr(main.notifier, "send_ntfy_alert",
                        lambda listing, predicted=None, threshold=3.5, proxy=None:
                        calls["ntfy"].append((str(listing["listing_id"]), predicted, proxy)) or True)
    incoming = tmp_path / "incoming"
    return {"db_path": db_path, "incoming": incoming, "calls": calls, "monkeypatch": monkeypatch,
            "tmp_path": tmp_path}


def _row(env, lid):
    conn = database.connect(env["db_path"])
    try:
        return conn.execute("SELECT * FROM listings WHERE listing_id=?", (lid,)).fetchone()
    finally:
        conn.close()


def test_gender_restricted_hard_filter(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: False)
    cases = {
        "2010": {"houseInfo": {"data": [{"key": "sex", "name": "性別", "value": "限女生"}]}},
        "2011": {"service": {"notice": [{"key": "sex_2", "name": "限男性"}]}},
        "2012": {"houseInfo": {"data": [{"key": "sex", "name": "性別", "value": "不拘"}]}},
        "2013": {"service": {"notice": [{"key": "pet", "name": "可養寵物"}]}},
    }
    (env["incoming"] / "listings").mkdir(parents=True, exist_ok=True)
    for lid, extra in cases.items():
        payload = {"listing_id": lid, "detail_failed": False, "payload_sha256": f"sha-{lid}",
                   "image_urls": [], "images": [],
                   "raw_search": {"id": int(lid), "title": "x", "price": "15000",
                                  "kind_name": "獨立套房", "area": 8.0},
                   "raw_metadata": dict({"status": "open"}, **extra)}
        (env["incoming"] / "listings" / f"{lid}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    conn = database.connect(env["db_path"])
    try:
        for lid in ("2010", "2011"):
            assert conn.execute("SELECT 1 FROM listings WHERE listing_id=?", (lid,)).fetchone() is None
            assert conn.execute("SELECT 1 FROM relay_state WHERE listing_id=?", (lid,)).fetchone() is not None
        for lid in ("2012", "2013"):
            assert conn.execute("SELECT 1 FROM listings WHERE listing_id=?", (lid,)).fetchone() is not None
    finally:
        conn.close()


def test_normalize_extracts_gender():
    restricted = ingestion.normalize_listing(
        {"id": 1, "title": "x", "price": "12500"},
        {"status": "open", "service": {"notice": [{"key": "sex_2", "name": "限女生"}]}})
    assert restricted["gender"] == "限女生"
    assert ingestion.passes_hard_filters(restricted)[0] is False
    any_gender = ingestion.normalize_listing(
        {"id": 2, "title": "x", "price": "12500"},
        {"status": "open", "houseInfo": {"data": [{"key": "sex", "value": "不拘"}]}})
    assert any_gender["gender"] == "不拘"
    assert ingestion.passes_hard_filters(any_gender)[0] is True
    absent = ingestion.normalize_listing({"id": 3, "title": "鄰居都女性", "price": "12500"},
                                         {"status": "open"})
    assert absent["gender"] is None
    assert ingestion.passes_hard_filters(absent)[0] is True
    variant = ingestion.normalize_listing(
        {"id": 4, "title": "x", "price": "12500"},
        {"status": "open", "houseInfo": {"data": [{"key": "sex", "value": "女性限定"}]}})
    assert ingestion.passes_hard_filters(variant)[0] is False
    in_title = ingestion.normalize_listing(
        {"id": 5, "title": "近捷運(限女性)套房", "price": "12500"}, {"status": "open"})
    assert ingestion.passes_hard_filters(in_title)[0] is False
    open_to_all = ingestion.normalize_listing(
        {"id": 6, "title": "x", "price": "12500"},
        {"status": "open", "remark": {"content": "不限男女，學生上班族皆可"}})
    assert ingestion.passes_hard_filters(open_to_all)[0] is True
    any_wording = ingestion.normalize_listing(
        {"id": 7, "title": "x", "price": "12500"},
        {"status": "open", "remark": {"content": "性別限制：不拘"}})
    assert ingestion.passes_hard_filters(any_wording)[0] is True


def test_hard_filters_drop_noncompliant_incoming(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: False)
    base = {"listing_id": None, "detail_failed": False, "payload_sha256": "s",
            "image_urls": [], "images": [], "raw_metadata": {"status": "open"}}
    bad = {
        "2001": {"price": "9000"},                          # under min price
        "2002": {"price": "15000", "kind_name": "整層住家"},  # wrong kind
        "2003": {"price": "15000", "kind_name": "雅房"},      # kind 4 equivalent
        "2004": {"price": "15000", "kind_name": "獨立套房", "area": 4.5},  # too small
        "2005": {"price": "15000", "kind_name": "分租套房",
                 "remark": "公共設施完整，嚴禁開伙"},            # cooking banned
    }
    for lid, over in bad.items():
        payload = dict(base, listing_id=lid)
        raw_search = {"id": int(lid), "title": "x", "price": over.pop("price", "15000")}
        raw_search.update({k: over.pop(k) for k in ("kind_name", "area") if k in over})
        meta = {"status": "open"}
        if over:
            meta["remark"] = {"content": over["remark"]}
        payload["raw_search"], payload["raw_metadata"] = raw_search, meta
        (env["incoming"] / "listings").mkdir(parents=True, exist_ok=True)
        (env["incoming"] / "listings" / f"{lid}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    conn = database.connect(env["db_path"])
    try:
        for lid in bad:
            assert conn.execute("SELECT 1 FROM listings WHERE listing_id=?", (lid,)).fetchone() is None
            assert conn.execute("SELECT 1 FROM relay_state WHERE listing_id=?", (lid,)).fetchone() is not None
    finally:
        conn.close()


def test_heuristic_flags_merge_into_warnings(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: False)
    lid = "2100"
    listings = env["incoming"] / "listings"
    listings.mkdir(parents=True, exist_ok=True)
    payload = {
        "listing_id": lid, "detail_failed": False, "payload_sha256": "sha-H",
        "image_urls": [], "images": [],
        "raw_search": {"id": int(lid), "title": "t", "price": "12000",
                       "kind_name": "獨立套房", "area": 7.0},
        "raw_metadata": {"status": "open",
                         "remark": {"content": "頂樓加蓋，電費每度6元，需追垃圾車"}},
    }
    (listings / f"{lid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    row = _row(env, lid)
    warnings = json.loads(row["qwen_warnings"])
    assert row["heuristic_score"] == pytest.approx(100 - 15 - 10 - 10)
    assert any("ILLEGAL_ROOFTOP" in w for w in warnings)
    assert any("HIGH_ELEC_FEE" in w for w in warnings)
    assert any("MANUAL_TRASH" in w for w in warnings)


def test_extract_text_warnings_rules():
    w = main.extract_text_warnings({
        "floor": "5樓/12樓(頂樓)", "description": "整租，水電另計", "deposit": "半年",
    })
    assert any("Top floor" in x for x in w)
    assert any("Utilities" in x for x in w)
    assert "6-month upfront payment required" in w
    assert main.extract_text_warnings({"floor": "3樓", "description": "", "deposit": "押一"}) == []


def test_normalize_maps_structured_pet_to_facilities():
    listing = ingestion.normalize_listing(
        {"id": 1, "title": "t", "price": "12000"},
        {"status": "open",
         "houseInfo": {"data": [{"key": "pet", "name": "寵物", "value": "不可養寵物"}]},
         "service": {"notice": [{"key": "pet", "name": "不可養寵物"}]}},
    )
    assert "不可養寵物" in listing["facilities"]
    listing = ingestion.normalize_listing(
        {"id": 2, "title": "t", "price": "12000"},
        {"status": "open",
         "houseInfo": {"data": [{"key": "pet", "name": "寵物", "value": "可養貓狗"}]}},
    )
    assert "可養寵物" in listing["facilities"]
    listing = ingestion.normalize_listing(
        {"id": 3, "title": "t", "price": "12000"},
        {"status": "open", "houseInfo": {"data": [{"key": "area", "name": "坪數", "value": "8"}]}},
    )
    assert listing["facilities"] == []


def test_offline_stores_pending_and_alerts_once(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: False)
    _write_payload(env["incoming"], "9001", ["https://img2.591.com.tw/h/a.jpg"])

    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    row = _row(env, "9001")
    assert row["image_status"] == "pending"
    assert bool(row["text_only_notified"]) is True
    warnings = json.loads(row["qwen_warnings"])
    assert any("Top floor" in x for x in warnings) and any("Utilities" in x for x in warnings)
    assert env["calls"]["proxy_alert"] == [(1, PROXY)]  # tunnel-first delivery
    assert env["calls"]["ntfy"] == []

    # Re-run over the same payloads: sha-skipped, pending already notified -> no alert spam.
    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    assert env["calls"]["proxy_alert"] == [(1, PROXY)]


def test_online_local_images_complete_and_notify_via_proxy(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: True)
    lid = "9002"
    _write_payload(env["incoming"], lid, ["https://img2.591.com.tw/h/b.jpg"])
    _write_webp(env["incoming"], lid, 0)

    # do_notify=None -> auto-on because the proxy is live.
    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 0
    row = _row(env, lid)
    assert row["image_status"] == "completed"
    assert row["predicted_score"] == pytest.approx(4.5)
    assert env["calls"]["ntfy"] and env["calls"]["ntfy"][0][0] == lid
    assert env["calls"]["ntfy"][0][2] == PROXY  # routed through the tunnel
    assert env["calls"]["proxy_alert"] == []


def test_rated_listing_repush_notified_only_once(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: True)
    lid = "9010"
    _write_payload(env["incoming"], lid, ["https://img2.591.com.tw/h/e.jpg"])
    _write_webp(env["incoming"], lid, 0)

    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 0
    assert [c[0] for c in env["calls"]["ntfy"]] == [lid]  # first alert

    conn = database.connect(env["db_path"])
    try:
        assert database.rate_listing(conn, lid, 4.2, 5.0, "")
    finally:
        conn.close()

    # Relay repushes the same listing with changed content (new sha) -> full re-score
    # but no second alert: rated listings are training data, not new matches.
    _write_payload(env["incoming"], lid, ["https://img2.591.com.tw/h/e.jpg",
                                          "https://img2.591.com.tw/h/f.jpg"], sha="sha-B")
    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 0
    row = _row(env, lid)
    assert row["predicted_score"] is not None  # re-scored
    assert [c[0] for c in env["calls"]["ntfy"]] == [lid]  # still exactly one alert


def test_online_drains_pending_then_scores(env, tmp_path):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: True)
    out_dir = tmp_path / "dl"
    env["monkeypatch"].setattr(main.image_queue, "IMAGES_DIR", out_dir)

    class Resp:
        content: bytes
        status_code = 200
        headers: ClassVar[dict] = {"content-type": "image/jpeg"}

        def raise_for_status(self):
            pass

    class FakeSession:
        def __init__(self):
            self.requested = []

        def get(self, url, timeout=None):
            self.requested.append(url)
            resp = Resp()
            buf = io.BytesIO()
            Image.new("RGB", (16, 16), (5, 5, 5)).save(buf, "JPEG")
            resp.content = buf.getvalue()
            return resp

    session = FakeSession()
    env["monkeypatch"].setattr(main.image_queue, "_make_session", lambda proxy_url: session)

    lid = "9003"
    _write_payload(env["incoming"], lid, ["https://img1.591.com.tw/h/c.jpg"])
    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 0
    row = _row(env, lid)
    assert row["image_status"] == "completed"
    assert (out_dir / lid / "00.webp").is_file()
    assert all(u.endswith(main.image_queue.PROXY_IMAGE_SUFFIX) for u in session.requested)
    assert env["calls"]["ntfy"] and env["calls"]["ntfy"][0][0] == lid


def test_completed_but_unscored_is_retried_next_run(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: True)
    lid = "9004"
    _write_payload(env["incoming"], lid, ["https://img2.591.com.tw/h/d.jpg"])
    _write_webp(env["incoming"], lid, 0)
    # Vision fails this run -> predicted_score stays NULL.
    env["monkeypatch"].setattr(main.vision_llm, "analyze_listing",
                               lambda listing, image_rows, bullets: None)
    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 1  # failed vision -> exit 1
    assert _row(env, lid)["predicted_score"] is None
    assert _row(env, lid)["image_status"] == "completed"
    # Next run: self-heal retries the vision pass without any new payload.
    env["monkeypatch"].setattr(main.vision_llm, "analyze_listing",
                               lambda listing, image_rows, bullets: dict(VISION_OK))
    assert main.run_incoming(env["incoming"], -1, do_notify=None) == 0
    assert _row(env, lid)["predicted_score"] == pytest.approx(4.5)
    # Regression: self-heal must reload stored image rows, not wipe them with [].
    conn = database.connect(env["db_path"])
    try:
        imgs = conn.execute(
            "SELECT image_path FROM listing_images WHERE listing_id=?", (lid,)).fetchall()
    finally:
        conn.close()
    assert len(imgs) == 1 and imgs[0]["image_path"]



# ---------------------------------------------------------------------------
# Bathroom authority on the live path (index alignment + partial-detection handling)
# ---------------------------------------------------------------------------

def _bath_axis(*idx):
    v = np.zeros(768, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v / np.linalg.norm(v)


def _seed_bath_labels(env):
    """Give the test DB a trained bathroom probe + a bath centroid.

    `predict_listing` returns 0.0 whenever `train_if_stale` has nothing to train on,
    so without these rows every assertion below would pass vacuously at 0.0.
    """
    good, bad = _bath_axis(0, 1, 2), _bath_axis(30, 31, 32)
    rng = np.random.default_rng(3)
    conn = database.connect(env["db_path"])
    try:
        for i in range(12):
            lid = f"SEED{i}"
            is_bath = i % 2 == 0
            database.upsert_listing(conn, {"listing_id": lid, "title": lid, "price": 12000})
            database.rate_listing(conn, lid, 4.0, 5.0 if is_bath else 1.0)
            vec = good if is_bath else bad
            vec = vec + rng.standard_normal(768).astype(np.float32) * 0.05
            database.replace_images(conn, lid, [{
                "ordinal": 0, "image_url": "u", "image_path": f"/p/{lid}_0",
                "is_bathroom": 1,
                "dino_embedding": (vec / np.linalg.norm(vec)).astype(np.float32).tobytes(),
            }])
    finally:
        conn.close()
    assert bathroom_probe.bath_centroid(database.connect(env["db_path"])) is not None


def _ingest(env, lid, n_photos):
    urls = [f"https://img/{i}.jpg" for i in range(n_photos)]
    _write_payload(env["incoming"], lid, urls)
    env["monkeypatch"].setattr(main.vision_llm, "analyze_listing",
                               lambda listing, image_rows, bullets: dict(VISION_OK))
    env["monkeypatch"].setattr(main.scoring, "RATED_THRESHOLD", 99)  # keep Qwen scoring
    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    _seed_bath_labels(env)
    return [{"ordinal": i, "image_url": f"u{i}", "image_path": f"/p/{lid}_{i}"}
            for i in range(n_photos)]


def _finalize(env, lid, image_rows, bath_flags, new_vecs):
    mp = env["monkeypatch"]
    mp.setattr(main.deduplication, "embed_image_rows", lambda rows: dict(new_vecs))
    mp.setattr(main.deduplication, "find_duplicate", lambda vecs, baseline, thr: (False, None))
    mp.setattr(main.bathroom_detect, "detect_flags", lambda paths, chunk=None: bath_flags)
    conn = database.connect(env["db_path"])
    try:
        row = conn.execute("SELECT * FROM listings WHERE listing_id=?", (lid,)).fetchone()
        assert row is not None, f"{lid} was not ingested"
        main.finalize_listing(
            conn, main._listing_from_row(row), image_rows, {}, [], False, proxy=None)
        got = conn.execute(
            "SELECT bath_model_score FROM listings WHERE listing_id=?", (lid,)).fetchone()[0]
    finally:
        conn.close()
    return got


def test_finalize_scores_the_confirmed_bathroom_after_a_dropped_embedding(env):
    """Ordinal 1 is the bathroom, but its embedding FAILED; ordinal 2 survived.

    embed_image_rows() keys by ordinal and omits failed rows, so position i in its
    output is NOT position i in image_rows. A stale index-blind implementation would
    score ordinal 2's vector (the bedroom) and still return a plausible number, so
    both directions are pinned here.
    """
    lid = "9101"
    rows = _ingest(env, lid, 3)
    bedroom = _bath_axis(600, 601, 602)          # far from the bath centroid
    vec0 = _bath_axis(0, 1, 2)                   # looks like a bathroom
    new_vecs = {0: vec0, 2: bedroom}             # ordinal 1's embedding is missing

    # Ordinal 1 is the confirmed bathroom but has no vector, so `confirmed_bath` ends
    # up empty and nothing is scored: the flag and its vector are dropped TOGETHER.
    assert _finalize(env, lid, rows, {0: 0, 1: 1, 2: 0}, new_vecs) == 0.0
    # Flagging ordinal 2 scores ordinal 2's (bedroom) vector, which the probe rates
    # low; flagging ordinal 0 scores ordinal 0's (bathroom) vector, which it rates
    # high. An index-blind implementation that ignored the ordinal map would swap
    # these two, so the ORDER of the two scores is the actual regression guard.
    scored_ordinal_2 = _finalize(env, lid, rows, {0: 0, 1: 0, 2: 1}, new_vecs)
    scored_ordinal_0 = _finalize(env, lid, rows, {0: 1, 1: 0, 2: 0}, new_vecs)
    assert scored_ordinal_0 > scored_ordinal_2, (
        f"ordinal 0 (bathroom-like) scored {scored_ordinal_0:.3f} but ordinal 2 "
        f"(bedroom-like) scored {scored_ordinal_2:.3f} -- indices are misaligned")


def test_finalize_falls_back_to_centroid_when_a_chunk_failed(env):
    """A timed-out chunk leaves positions ABSENT from bath_flags -- unknown, not 0.

    Passing photo_indices=[] in that state both excluded the surviving photos and
    disabled the centroid, so a real bathroom in the untried chunk scored 0.0.
    """
    lid = "9102"
    rows = _ingest(env, lid, 2)
    vec0 = _bath_axis(0, 1, 2)                   # bathroom-like
    vec1 = _bath_axis(600, 601, 602)             # bedroom-like
    new_vecs = {0: vec0, 1: vec1}

    # complete detection, nothing is a bathroom -> authoritative 0.0
    assert _finalize(env, lid, rows, {0: 0, 1: 0}, new_vecs) == 0.0
    # ordinal 1's chunk failed -> position 1 is ABSENT -> centroid must stay eligible,
    # and it selects ordinal 0 (bathroom-like) for a non-zero score
    assert _finalize(env, lid, rows, {0: 1}, new_vecs) > 0.0
    # complete detection that found one -> authoritative; the probe is applied to
    # exactly the flagged photo, so the score comes from vec1, not the centroid pick
    assert _finalize(env, lid, rows, {0: 0, 1: 1}, new_vecs) > 0.0
    # ...and the centroid alone would have picked ordinal 0 instead (see the partial case)
    assert _finalize(env, lid, rows, {0: 1}, new_vecs) > 0.0


def test_finalize_keeps_stored_labels_authoritative_when_detection_is_skipped(env):
    """scripts/backfill_vision.py re-finalizes with detect_flags -> None while feeding
    the STORED labels on the rows.

    Coverage must be decided from the labels themselves, not from whether detection
    ran this round: requiring `bath_flags is not None` made every backfill pass
    re-score Qwen's verdicts through the centroid and overwrite them.
    """
    lid = "9103"
    rows = _ingest(env, lid, 2)
    bath_like = _bath_axis(0, 1, 2)
    bedroom_like = _bath_axis(600, 601, 602)
    new_vecs = {0: bath_like, 1: bedroom_like}

    # Stored labels name ordinal 0 -> the bathroom-like vector is scored
    for r, lab in zip(rows, [1, 0]):
        r["is_bathroom"] = lab
    scored_stored = _finalize(env, lid, rows, None, new_vecs)
    # Stored labels name ordinal 1 -> the bedroom-like vector is scored instead.
    # A centroid-only path cannot produce this order (it always picks bath_like).
    for r, lab in zip(rows, [0, 1]):
        r["is_bathroom"] = lab
    scored_swapped = _finalize(env, lid, rows, None, new_vecs)
    assert scored_stored > scored_swapped, (
        f"stored labels were not authoritative: {scored_stored:.3f} vs {scored_swapped:.3f}; "
        "detection=None must not fall back to the centroid when every photo is labelled")

    # A PARTIAL label set (some NULL) is unknown coverage -> the centroid stays eligible
    for r, lab in zip(rows, [1, None]):
        r["is_bathroom"] = lab
    assert _finalize(env, lid, rows, None, new_vecs) > 0.0
