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
import os
import re

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL", "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"
)

BASE_SYSTEM_PROMPT = """\
You are an expert real estate auditor analyzing a Rent591 apartment listing.
You must extract facts from Chinese listing text and analyze all provided images.

Return ONLY a valid JSON object matching this schema:
{
  "qwen_warnings": ["string warning highlights, e.g. 4th floor walk-up, shared meter"],
  "vision_flags": {
    "has_bathroom_img": bool,
    "shower_sink_combo": bool,
    "drainage_risk": bool,
    "has_kitchen_sink": bool,
    "has_exterior_window": bool
  },
  "predicted_score": float
}
predicted_score is 1.0 to 5.0 based on the user context rules and overall condition.
If no images are available, set all vision_flags to false and base the score on text only.
"""

_DEFAULT_FLAGS = {
    "has_bathroom_img": False,
    "shower_sink_combo": False,
    "drainage_risk": False,
    "has_kitchen_sink": False,
    "has_exterior_window": False,
}


def construct_full_prompt(bullets: str | None) -> str:
    dynamic = bullets or (
        "- Prioritize dry/wet separation in bathroom.\n"
        "- Flag shower-sink combo faucet setups."
    )
    return f"{BASE_SYSTEM_PROMPT}\n\n### User Context & Evolving Preferences ###\n{dynamic}"


def _image_b64(path: str) -> str:
    from PIL import Image
    with Image.open(path) as raw:
        img = raw.convert("RGB")
    buf = io.BytesIO()
    try:
        img.save(buf, "PNG")
    finally:
        img.close()
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
    images = [_image_b64(p) for p in image_paths[:8]]
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


def _clean_score(raw) -> float:
    try:
        s = float(raw)
        return max(1.0, min(5.0, s))
    except Exception:
        return 3.0


def analyze_listing(listing: dict, image_rows: list[dict], bullets: str | None) -> dict | None:
    """Returns {qwen_warnings, vision_flags, qwen_direct_score} or None on failure."""
    try:
        paths = [r["image_path"] for r in image_rows if r.get("image_path")]
        messages = build_messages(listing, paths, bullets)
        raw = ask_ollama(messages)
        data = parse_json(raw)
        if not data:
            return None
        flags = dict(_DEFAULT_FLAGS)
        flags.update(data.get("vision_flags") or {})
        for k in _DEFAULT_FLAGS:
            flags[k] = bool(flags[k])
        warnings = data.get("qwen_warnings") or []
        if not isinstance(warnings, list):
            warnings = [str(warnings)]
        return {
            "qwen_warnings": [str(w) for w in warnings],
            "vision_flags": flags,
            "qwen_direct_score": _clean_score(data.get("predicted_score", 3.0)),
        }
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
