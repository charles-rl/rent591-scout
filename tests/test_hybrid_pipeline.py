"""Hybrid orchestration (main.run_incoming): offline queue+alert vs live drain+vision+notify."""

import io
import json

import pytest
from PIL import Image

import main
from src import database

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
        "raw_search": {"id": int(lid), "title": f"Listing {lid}", "price": "18000",
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


def test_extract_text_warnings_rules():
    w = main.extract_text_warnings({
        "floor": "5樓/12樓(頂樓)", "description": "整租，水電另計", "deposit": "半年",
    })
    assert any("頂樓" in x for x in w)
    assert any("水電" in x for x in w)
    assert "要求半年付" in w
    assert main.extract_text_warnings({"floor": "3樓", "description": "", "deposit": "押一"}) == []


def test_offline_stores_pending_and_alerts_once(env):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: False)
    _write_payload(env["incoming"], "9001", ["https://img2.591.com.tw/h/a.jpg"])

    assert main.run_incoming(env["incoming"], -1, do_notify=False) == 0
    row = _row(env, "9001")
    assert row["image_status"] == "pending"
    assert bool(row["text_only_notified"]) is True
    warnings = json.loads(row["qwen_warnings"])
    assert any("頂樓" in x for x in warnings) and any("水電" in x for x in warnings)
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


def test_online_drains_pending_then_scores(env, tmp_path):
    env["monkeypatch"].setattr(main.proxy_check, "is_proxy_available", lambda *a, **k: True)
    out_dir = tmp_path / "dl"
    env["monkeypatch"].setattr(main.image_queue, "IMAGES_DIR", out_dir)

    class Resp:
        content: bytes
        status_code = 200
        headers = {"content-type": "image/jpeg"}

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
