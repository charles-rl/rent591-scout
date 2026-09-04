"""Qwen photo classifier: which photos of a listing show the bathroom.

One cheap JSON-only call per chunk of photos (no listing text, no preference
bullets). Powers scripts/backfill_bathrooms.py and the live pipeline, which
persists results to listing_images.is_bathroom (1/0, NULL = unknown) so the
bathroom probe can train/refresh on real labels instead of self-selected ones.
"""

from __future__ import annotations

import logging
import os

from . import vision_llm

logger = logging.getLogger(__name__)

BATH_DETECT_CHUNK = max(1, int(os.environ.get("BATH_DETECT_CHUNK", "8")))
BATH_DETECT_TIMEOUT = int(os.environ.get("BATH_DETECT_TIMEOUT", "300"))

_SYSTEM = "You are a precise photo classifier. Reply with JSON only."
_PROMPT_T = (
    "{n} photos are attached, in order. Return only JSON "
    '{{"bathroom_indices":[...]}} where the array lists the 0-based positions of every '
    "photo that clearly shows a bathroom interior: toilet, shower, bathtub, washbasin or "
    "shower floor drain. Exclude bedrooms, kitchens, hallways, building exteriors and "
    "photos where a bathroom is only glimpsed through a doorway. Return [] if none."
)


def detect_chunk(paths: list[str]) -> set[int]:
    """Indices (within paths) that are bathroom photos; empty set when model says none."""
    images = [b for b in (vision_llm._image_b64(p) for p in paths) if b]
    if not images:
        return set()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _PROMPT_T.format(n=len(images)), "images": images},
    ]
    data = vision_llm.parse_json(vision_llm.ask_ollama(messages, timeout=BATH_DETECT_TIMEOUT))
    out: set[int] = set()
    if isinstance(data, dict):
        for i in data.get("bathroom_indices") or []:
            try:
                n = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= n < len(paths):
                out.add(n)
    return out


def detect_flags(image_paths: list[str | None], chunk: int | None = None) -> dict[int, int] | None:
    """{index: 1|0} over positions in image_paths; None when every chunk failed.

    Chunks are independent: a failed chunk only leaves its photos unlabeled (absent
    from the dict), never tainting the rest. Callers persist what they get.
    """
    chunk = chunk or BATH_DETECT_CHUNK
    valid = [(i, p) for i, p in enumerate(image_paths) if p]
    flags: dict[int, int] = {}
    ok = False
    for s in range(0, len(valid), chunk):
        part = valid[s:s + chunk]
        try:
            hits = detect_chunk([p for _, p in part])
        except Exception as e:
            logger.warning("bathroom detection chunk failed (%d photos): %s", len(part), e)
            continue
        ok = True
        for j, (i, _) in enumerate(part):
            flags[i] = 1 if j in hits else 0
    return flags if ok else None
