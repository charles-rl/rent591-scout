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


def detect_chunk(paths: list[str]) -> tuple[set[int], set[int]]:
    """(hits, shown): indices (within paths) of bathroom photos and of photos the model saw.

    `shown` excludes photos whose b64 could not be read inside an otherwise-live
    chunk -- the model never saw them, so the caller must leave them unlabelled
    rather than record an authoritative 0.

    Raises when NO image in the chunk could be read, and when the model reply does
    not carry a `bathroom_indices` list: both are infrastructure/model failures,
    not a verdict. Returning an empty set there would let the caller record
    `is_bathroom=0` for every photo -- an authoritative "Qwen says there is no
    bathroom" -- which then persists (backfill revisits only NULLs) and, under the
    authority rule in `main.py`, forces bath_model_score to 0.0.
    """
    kept = [(i, b) for i, b in ((i, vision_llm._image_b64(p)) for i, p in enumerate(paths)) if b]
    if not kept:
        raise ValueError(f"none of {len(paths)} chunk photos could be read as images")
    images = [b for _, b in kept]
    # The model is shown `images` only, so its indices are positions within `kept`,
    # NOT within `paths`. Map them back through the filter or a dead leading photo
    # shifts every label onto the wrong ordinal.
    kept_positions = [i for i, _ in kept]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _PROMPT_T.format(n=len(images)), "images": images},
    ]
    data = vision_llm.parse_json(vision_llm.ask_ollama(messages, timeout=BATH_DETECT_TIMEOUT))
    if not isinstance(data, dict) or not isinstance(data.get("bathroom_indices"), list):
        raise ValueError(  # noqa: TRY004 - the "type" is the LLM reply's JSON shape, not a caller argument
            f"model reply did not contain a bathroom_indices list: {data!r}")
    out: set[int] = set()
    for i in data["bathroom_indices"]:
        try:
            n = int(i)
        except (TypeError, ValueError):
            continue
        if 0 <= n < len(kept_positions):
            out.add(kept_positions[n])
    return out, set(kept_positions)


def detect_flags(image_paths: list[str | None], chunk: int | None = None) -> dict[int, int] | None:
    """{index: 1|0} over positions in image_paths; None when every chunk failed.

    Chunks are independent: a failed chunk leaves its photos ABSENT from the dict
    (unknown -- "detection never ran on these"), never tainting the rest. Photos
    whose b64 failed inside a live chunk are absent for the same reason (the model
    never saw them). Callers must treat an absent key as unknown rather than as 0;
    `main.py` derives that from which positions the dict covers.
    """
    chunk = chunk or BATH_DETECT_CHUNK
    valid = [(i, p) for i, p in enumerate(image_paths) if p]
    flags: dict[int, int] = {}
    ok = False
    for s in range(0, len(valid), chunk):
        part = valid[s:s + chunk]
        try:
            hits, shown = detect_chunk([p for _, p in part])
        except Exception as e:
            logger.warning("bathroom detection chunk failed (%d photos): %s", len(part), e)
            continue
        ok = True
        for j, (i, _) in enumerate(part):
            # Photos the model never saw (b64 failure inside a live chunk) stay
            # absent = unknown; writing 0 here made them permanent "no bathroom".
            if j in shown:
                flags[i] = 1 if j in hits else 0
    return flags if ok else None
