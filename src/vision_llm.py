"""Qwen 27B (Ollama) vision + text analysis wrapper.

Two-tier system prompt: immutable BASE_SYSTEM_PROMPT (structured JSON) + dynamic
preference bullets. Uses the non-uncensored local model
hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL via Ollama /api/chat (vision-capable).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL", "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"
)
VLM_ATTEMPTS = max(1, int(os.environ.get("VLM_ATTEMPTS", "3")))
VLM_IMAGE_MAX_SIDE = max(256, int(os.environ.get("VLM_IMAGE_MAX_SIDE", "1024")))
VLM_MAX_IMAGES = max(1, int(os.environ.get("VLM_MAX_IMAGES", "8")))

BASE_SYSTEM_PROMPT = """\
You are an expert real estate auditor analyzing a Rent591 apartment listing.
You must extract facts from Chinese listing text and analyze all provided images.

Return ONLY a valid JSON object matching this schema (no prose, no code fences):
{
  "qwen_warnings": ["string warning highlights, e.g. 4th floor walk-up, shared meter"],
  "vision_flags": {
    "has_bathroom_img": bool,
    "shower_sink_combo": bool,
    "drainage_risk": bool,
    "has_kitchen_sink": bool,
    "has_exterior_window": bool
  },
  "qwen_direct_score": float
}
qwen_direct_score is 1.0 to 5.0 based on the user context rules and overall condition.
If no images are available, set all vision_flags to false and base the score on text only.

Verification rules (soft constraints — warn, never silently reject):
- 分租套房/共居 listing: determine whether the bathroom is 獨立衛浴 (private, inside the
  rented unit) or 共用 (shared hallway/floor). If text and photos cannot confirm it, do NOT
  guess — add warning "衛浴獨立性未確認" and let the human reviewer decide.
- Windowless spaces: if the room (or bathroom) has no exterior window/balcony access,
  set has_exterior_window=false and add a 無窗/採光不足 warning.
- 頂樓加蓋 (illegal rooftop addition): metal sheet roofing, exterior staircases, or
  cramped top-floor construction → add warning "頂樓加蓋疑慮".
- Cooking: note any 可開伙 / 電可開伙 / 禁用明火 mentions in warnings.
- Uncertainty policy: when evidence is missing or ambiguous, surface a specific warning
  instead of changing the score; only certain defects should lower qwen_direct_score.
"""

DEFAULT_BULLETS = (
    "- Prioritize dry/wet separation in bathroom.\n"
    "- Flag shower-sink combo faucet setups.\n"
    "- 分租套房必須確認獨立衛浴；不確定時標示『衛浴獨立性未確認』交由人工判斷。\n"
    "- 無窗、頂樓加蓋、共用卫浴等疑慮一律標入 qwen_warnings。"
)

_RETRY_NOTE = (
    "Your previous response was not a valid JSON object for the required schema. "
    "Respond again with ONLY the JSON object described in the system prompt — "
    "no prose, no code fences, all keys present."
)

_DEFAULT_FLAGS = {
    "has_bathroom_img": False,
    "shower_sink_combo": False,
    "drainage_risk": False,
    "has_kitchen_sink": False,
    "has_exterior_window": False,
}


def construct_full_prompt(bullets: str | None) -> str:
    dynamic = (bullets or "").strip() or DEFAULT_BULLETS
    return f"{BASE_SYSTEM_PROMPT}\n\n### User Context & Evolving Preferences ###\n{dynamic}"


def _image_b64(path: str) -> str | None:
    """Downscale before encode: full-size PNG base64 bloats every Ollama call.

    Returns None for missing/corrupt files so one dead photo never fails the
    whole vision pass (DB rows can outlive data/images/ cleanups).
    """
    from PIL import Image
    try:
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        try:
            w, h = img.size
            m = max(w, h)
            if m > VLM_IMAGE_MAX_SIDE:
                scale = VLM_IMAGE_MAX_SIDE / m
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
        finally:
            img.close()
    except Exception as e:
        logger.warning("image %s skipped for vision: %s", path, e)
        return None
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(listing: dict, image_paths: list[str], bullets: str | None) -> list[dict]:
    facts = {
        "title": listing.get("title", ""),
        "description": listing.get("description", ""),
        "layout": listing.get("layout", ""),
        "area": listing.get("area"),
        "floor": listing.get("floor", ""),
        "shape": listing.get("shape", ""),
        "tags": listing.get("tags", []),
        "contain_cost": listing.get("contain_cost", []),
        "facilities": listing.get("facilities", []),
        "social_house": listing.get("social_house"),
        "deposit": listing.get("deposit", ""),
        "rent_per": listing.get("rent_per"),
    }
    text = (
        "Analyze this Rent591 rental listing. Return the JSON object.\n"
        f"Listing facts: {json.dumps(facts, ensure_ascii=False)}\n"
        "Images are attached below (each corresponds to one photo of the property)."
    )
    user_msg: dict = {"role": "user", "content": text}
    images = [b for b in (_image_b64(p) for p in image_paths[:VLM_MAX_IMAGES] if p) if b]
    if images:
        user_msg["images"] = images
    return [
        {"role": "system", "content": construct_full_prompt(bullets)},
        user_msg,
    ]


def ask_ollama(messages: list[dict], timeout: int = 600) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def parse_json(text: str) -> dict:
    """Robust JSON extraction: strip fences/code blocks, pull balanced {…} block."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        return {}
    try:
        return json.loads(text[start:end])
    except Exception:
        return {}


def validate_analysis(data) -> dict | None:
    """Strict schema validation: qwen_warnings list, 5 boolean vision_flags, score 1.0-5.0.

    Returns a normalized {qwen_warnings, vision_flags, qwen_direct_score} dict,
    or None if the payload is unusable (missing/non-numeric/non-finite score, bad types).
    """
    if not isinstance(data, dict):
        return None
    score = data.get("qwen_direct_score", data.get("predicted_score"))  # accept legacy key
    if score is None or isinstance(score, (bool, str)):
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):  # json.loads accepts bare NaN/Infinity tokens
        return None
    score = max(1.0, min(5.0, score))
    flags_raw = data.get("vision_flags")
    if flags_raw is None:
        flags_raw = {}
    if not isinstance(flags_raw, dict):
        return None
    warnings = data.get("qwen_warnings")
    if warnings is None:
        warnings = []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    flags = dict(_DEFAULT_FLAGS)
    for k, v in flags_raw.items():
        if k in _DEFAULT_FLAGS:
            flags[k] = bool(v)
    return {
        "qwen_warnings": [str(w) for w in warnings],
        "vision_flags": flags,
        "qwen_direct_score": score,
    }


def analyze_listing(listing: dict, image_rows: list[dict], bullets: str | None) -> dict | None:
    """Returns {qwen_warnings, vision_flags, qwen_direct_score} or None on failure.

    Retry fallback loop: up to VLM_ATTEMPTS exchanges with a correction message on
    malformed/invalid JSON, then a final text-only attempt (no images) as fallback.
    """
    try:
        paths = [r["image_path"] for r in image_rows if r.get("image_path")]
        messages = build_messages(listing, paths, bullets)
        for attempt in range(1, VLM_ATTEMPTS + 1):
            try:
                raw = ask_ollama(messages)
            except Exception as e:
                logger.warning("qwen attempt %d/%d endpoint error: %s", attempt, VLM_ATTEMPTS, e)
                raw = ""
            data = validate_analysis(parse_json(raw))
            if data is not None:
                if attempt > 1:
                    logger.info("qwen JSON recovered on attempt %d", attempt)
                return data
            messages = messages + [
                {"role": "assistant", "content": (raw or "")[:2000]},
                {"role": "user", "content": _RETRY_NOTE},
            ]
        if paths:
            logger.warning("qwen malformed after %d attempts -> text-only fallback", VLM_ATTEMPTS)
            data = validate_analysis(parse_json(ask_ollama(build_messages(listing, [], bullets))))
            if data is not None:
                return data
        logger.warning("qwen analysis failed: no valid JSON after retries")
        return None
    except Exception as e:
        logger.warning("analyze failed: %s", e)
        return None


def consolidate_preferences(current_bullets: str | None, feedback: str) -> str:
    """Text-only Qwen call merging current preference bullets with new feedback (max 7)."""
    prompt = f"""\
Current user preferences:
{current_bullets or '(none)'}

New feedback received: "{feedback}"

Task: Update the bulleted list of user preferences (max 7 items).
If the feedback introduces a new rule, add or update a bullet.
If it is redundant or invalid, keep the list unchanged.
Return ONLY the bulleted list, one bullet per line starting with "-".
"""
    try:
        messages = [
            {"role": "system", "content": "You summarize rental preferences concisely."},
            {"role": "user", "content": prompt},
        ]
        raw = ask_ollama(messages, timeout=300)
        lines = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip().startswith(("-", "•"))]
        if not lines:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()][:7]
        return "\n".join(f"- {ln}" for ln in lines[:7]) if lines else (current_bullets or "")
    except Exception as e:
        logger.warning("consolidate failed: %s", e)
        return current_bullets or ""
