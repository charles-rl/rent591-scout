"""Offline vision_llm tests: Ollama endpoint fully mocked (no network)."""

import json

import pytest
from PIL import Image

from src import database, vision_llm


@pytest.fixture
def webp(tmp_path):
    p = tmp_path / "img.webp"
    Image.new("RGB", (64, 64), (10, 120, 200)).save(p, "WEBP")
    return str(p)


def fake_endpoint(responses):
    """Capture messages per call; replay canned assistant replies in order."""
    calls = []

    def _ask(messages, timeout=600):
        calls.append(messages)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return _ask, calls


VALID = {
    "qwen_warnings": ["4th floor walk-up"],
    "vision_flags": {
        "has_bathroom_img": True,
        "shower_sink_combo": False,
        "drainage_risk": False,
        "has_kitchen_sink": True,
        "has_exterior_window": True,
    },
    "qwen_direct_score": 3.5,
}


def test_valid_json_passthrough(monkeypatch, webp):
    ask, _ = fake_endpoint([json.dumps(VALID)])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    out = vision_llm.analyze_listing({"title": "t"}, [{"image_path": webp}], None)
    assert out == VALID


def test_fenced_json_with_prose(monkeypatch, webp):
    raw = "Here is my analysis:\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps!"
    ask, _ = fake_endpoint([raw])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    assert vision_llm.analyze_listing({}, [{"image_path": webp}], None) == VALID


def test_retry_loop_recovers(monkeypatch, webp):
    ask, calls = fake_endpoint(["Sorry, I cannot parse this at all", json.dumps(VALID)])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    out = vision_llm.analyze_listing({}, [{"image_path": webp}], None)
    assert out == VALID
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "user" and vision_llm._RETRY_NOTE in calls[1][-1]["content"]


def test_all_malformed_then_text_only_fallback(monkeypatch, webp):
    ask, calls = fake_endpoint(["garbage"] * (vision_llm.VLM_ATTEMPTS + 1))
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    out = vision_llm.analyze_listing({}, [{"image_path": webp}], None)
    assert out is None
    final = calls[-1]
    assert "images" not in final[1], "final fallback must be text-only"
    assert len(calls) == vision_llm.VLM_ATTEMPTS + 1


def test_endpoint_error_never_raises(monkeypatch):
    def boom(messages, timeout=600):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(vision_llm, "ask_ollama", boom)
    assert vision_llm.analyze_listing({}, [], None) is None


def test_legacy_predicted_score_key(monkeypatch):
    legacy = dict(VALID, predicted_score=4.0)
    legacy.pop("qwen_direct_score")
    ask, _ = fake_endpoint([json.dumps(legacy)])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    out = vision_llm.analyze_listing({}, [], None)
    assert out["qwen_direct_score"] == 4.0


def test_score_clamp_defaults_and_coercion(monkeypatch):
    sloppy = {
        "qwen_warnings": "single string not a list",
        "vision_flags": {"shower_sink_combo": 1, "bogus_key": True},
        "qwen_direct_score": 9.7,
    }
    ask, _ = fake_endpoint([json.dumps(sloppy)])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    out = vision_llm.analyze_listing({}, [], None)
    assert out["qwen_direct_score"] == 5.0
    assert out["vision_flags"]["shower_sink_combo"] is True
    assert out["vision_flags"]["has_bathroom_img"] is False  # missing key -> default
    assert "bogus_key" not in out["vision_flags"]
    assert out["qwen_warnings"] == ["single string not a list"]


def test_validate_rejects_unusable():
    assert vision_llm.validate_analysis({}) is None
    assert vision_llm.validate_analysis({"qwen_direct_score": "abc"}) is None
    assert vision_llm.validate_analysis({"qwen_direct_score": 3, "vision_flags": "nope"}) is None
    assert vision_llm.validate_analysis(None) is None
    assert vision_llm.validate_analysis({"qwen_direct_score": True}) is None  # bool is not a score
    assert vision_llm.validate_analysis({"qwen_direct_score": float("nan")}) is None
    assert vision_llm.validate_analysis({"qwen_direct_score": float("inf")}) is None


def test_nan_score_token_rejected_end_to_end(monkeypatch):
    # json.loads accepts the non-standard NaN token; it must not pass validation as 5.0.
    ask, _ = fake_endpoint(['{"qwen_warnings": [], "vision_flags": {}, "qwen_direct_score": NaN}'])
    monkeypatch.setattr(vision_llm, "ask_ollama", ask)
    assert vision_llm.analyze_listing({}, [], None) is None


def test_build_messages_cap_and_prompt(monkeypatch, tmp_path):
    paths = []
    for i in range(12):
        p = tmp_path / f"{i}.webp"
        Image.new("RGB", (2048, 128), (i * 10 % 255, 0, 0)).save(p, "WEBP")
        paths.append(str(p))
    msgs = vision_llm.build_messages({"title": "雙北市"}, paths, "- Prefer balcony.")
    assert msgs[0]["role"] == "system"
    assert vision_llm.BASE_SYSTEM_PROMPT in msgs[0]["content"]
    assert "- Prefer balcony." in msgs[0]["content"]
    assert len(msgs[1]["images"]) == vision_llm.VLM_MAX_IMAGES
    assert "雙北市" in msgs[1]["content"]


def test_downscaled_encode(tmp_path):
    import base64
    import io
    big = tmp_path / "big.webp"
    Image.new("RGB", (2048, 2048), (5, 100, 150)).save(big, "WEBP")
    b64 = vision_llm._image_b64(str(big))
    with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
        assert img.format == "JPEG"
        assert max(img.size) == vision_llm.VLM_IMAGE_MAX_SIDE


def test_sqlite_persistence_roundtrip(tmp_path):
    conn = database.connect(tmp_path / "t.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)")}
    assert "is_duplicate" in cols
    listing = {
        "listing_id": "T1", "title": "test", "is_duplicate": False,
        "qwen_warnings": VALID["qwen_warnings"],
        "qwen_vision_flags": VALID["vision_flags"],
        "qwen_direct_score": VALID["qwen_direct_score"],
    }
    database.upsert_listing(conn, listing)
    row = conn.execute(
        "SELECT qwen_warnings, qwen_vision_flags, qwen_direct_score, is_duplicate FROM listings WHERE listing_id='T1'"
    ).fetchone()
    assert json.loads(row["qwen_warnings"]) == VALID["qwen_warnings"]
    assert json.loads(row["qwen_vision_flags"]) == VALID["vision_flags"]
    assert row["qwen_direct_score"] == 3.5
    assert row["is_duplicate"] == 0
    listing["is_duplicate"] = True
    database.upsert_listing(conn, listing)
    assert conn.execute("SELECT is_duplicate FROM listings WHERE listing_id='T1'").fetchone()[0] == 1
    # Omitted key on a fresh insert must store FALSE, never NULL (SQL DEFAULT doesn't
    # fire for explicitly-bound NULLs in the INSERT column list).
    database.upsert_listing(conn, {"listing_id": "T2", "title": "no flag key"})
    assert conn.execute("SELECT is_duplicate FROM listings WHERE listing_id='T2'").fetchone()[0] == 0
    # Omitted key on re-upsert (e.g. inactive path) preserves the stored flag.
    database.upsert_listing(conn, {"listing_id": "T1", "title": "inactive re-run, no flag key"})
    assert conn.execute("SELECT is_duplicate FROM listings WHERE listing_id='T1'").fetchone()[0] == 1
    conn.close()
