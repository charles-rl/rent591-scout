"""Pending-image queue drain: mocked proxy session, temp DB + temp image dir."""

import io

import pytest
import requests
from PIL import Image

from src import database
from src.utils import image_queue

IMG_URLS = [
    "https://img2.591.com.tw/house/2026/01/01/aaa.jpg",
    "https://img1.591.com.tw/house/2026/01/01/bbb.jpg",
]


def _image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (120, 60, 200)).save(buf, "JPEG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers = {"content-type": "image/webp" if status_code == 200 else "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """url -> status: 200 serves an image, HTTPError(404)=persistent, ConnectionError=tunnel down."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.requested: list[str] = []

    def get(self, url, timeout=None):
        self.requested.append(url)
        if self.mode == "down":
            raise requests.ConnectionError("tunnel down")
        if self.mode == "ratelimit":
            raise requests.exceptions.RetryError("too many 502 error responses")
        if self.mode == "notfound":
            return _Resp(b"", status_code=404)
        if self.mode == "partial" and url.endswith("bbb.jpg" + image_queue.PROXY_IMAGE_SUFFIX):
            return _Resp(b"", status_code=404)
        return _Resp(_image_bytes())


@pytest.fixture
def queue_db(tmp_path, monkeypatch):
    monkeypatch.setattr(image_queue, "IMAGES_DIR", tmp_path / "images")
    conn = database.connect(tmp_path / "t.db")
    yield conn
    conn.close()


def _seed(conn, lid, urls):
    database.upsert_listing(conn, {
        "listing_id": lid, "title": lid, "is_active": True,
        "image_urls": urls, "image_status": "pending",
    })


def test_pending_drains_to_completed_with_suffix(queue_db):
    _seed(queue_db, "9001", IMG_URLS)
    session = FakeSession("ok")
    results = _run(queue_db, session)
    assert [lid for lid, _ in results] == ["9001"]
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9001'").fetchone()
    assert row["image_status"] == "completed"
    # bare CDN originals must be requested as the !fit resize variant (403 otherwise)
    assert all(u.endswith(image_queue.PROXY_IMAGE_SUFFIX) for u in session.requested)
    _lid, rows = results[0]
    assert len(rows) == 2 and all(r["image_path"] for r in rows)
    for r in rows:
        img = Image.open(r["image_path"])
        assert img.format == "WEBP"
    stored = database.get_all_images(queue_db)["9001"]
    assert len(stored) == 2


def test_all_failures_mark_failed(queue_db):
    _seed(queue_db, "9002", IMG_URLS)
    results = _run(queue_db, FakeSession("notfound"))
    assert results == []
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9002'").fetchone()
    assert row["image_status"] == "failed"


def test_partial_failures_still_complete(queue_db):
    _seed(queue_db, "9003", IMG_URLS)
    results = _run(queue_db, FakeSession("partial"))
    assert [lid for lid, _ in results] == ["9003"]
    rows = results[0][1]
    assert sum(1 for r in rows if r["image_path"]) == 1
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9003'").fetchone()
    assert row["image_status"] == "completed"


def test_tunnel_down_leaves_pending(queue_db):
    _seed(queue_db, "9004", IMG_URLS)
    results = _run(queue_db, FakeSession("down"))
    assert results == []
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9004'").fetchone()
    assert row["image_status"] == "pending"


def test_rate_limit_leaves_pending_and_continues_next_listing(queue_db, monkeypatch):
    """502 exhaustion (proxy alive but throttling) must NOT be treated as PC offline."""
    monkeypatch.setattr(image_queue, "RATE_LIMIT_BACKOFF", 0)
    _seed(queue_db, "9007", IMG_URLS)   # 502 mid-drain -> stays pending
    _seed(queue_db, "9008", [])          # still processed (skipped) -> proves loop continued
    results = _run(queue_db, FakeSession("ratelimit"))
    assert results == [("9008", [])]
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9007'").fetchone()
    assert row["image_status"] == "pending"


def test_no_urls_skipped_for_text_only_vision(queue_db):
    _seed(queue_db, "9005", [])
    results = _run(queue_db, FakeSession("ok"))
    assert [lid for lid, _ in results] == ["9005"]
    assert results[0][1] == []
    row = queue_db.execute("SELECT image_status FROM listings WHERE listing_id='9005'").fetchone()
    assert row["image_status"] == "skipped"


def test_inactive_listings_are_not_drained(queue_db):
    database.upsert_listing(queue_db, {
        "listing_id": "9006", "title": "x", "is_active": False,
        "image_urls": IMG_URLS, "image_status": "pending",
    })
    results = _run(queue_db, FakeSession("ok"))
    assert results == []


def _run(conn, session):
    orig = image_queue._make_session
    image_queue._make_session = lambda proxy_url: session
    try:
        return image_queue.process_pending_images(conn, "http://127.0.0.1:8999")
    finally:
        image_queue._make_session = orig
