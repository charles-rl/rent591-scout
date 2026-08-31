"""DINOv2 feature extraction + multi-image group cosine deduplication.

Model: facebook/dinov2-base (768-dim CLS embeddings), cached under models/dinov2_cache.
Group similarity follows the outline: mean over new vectors of max cosine over stored set.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_MODEL = None
_PROCESSOR = None


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    import torch
    from transformers import AutoImageProcessor, AutoModel

    cache_dir = __import__("os").environ.get("DINOV2_CACHE", "models/dinov2_cache")
    _PROCESSOR = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
    _MODEL = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
    _MODEL.eval()
    if torch.cuda.is_available():
        _MODEL = _MODEL.to("cuda")
    return _MODEL, _PROCESSOR


def embed_image(path: str) -> np.ndarray | None:
    """Return L2-normalized 768-dim CLS embedding, or None on failure."""
    import torch

    try:
        model, processor = _load()
        img = Image.open(path).convert("RGB")
        inputs = processor(images=[img], return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0]
        vec = out.cpu().numpy().astype(np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None
    except Exception as e:
        print(f"[dedup] embed {path} failed: {e}")
        return None


def embed_image_rows(image_rows: list[dict]) -> dict[int, np.ndarray]:
    """Embed a listing's image rows; returns {ordinal: vec} for successful ones."""
    out: dict[int, np.ndarray] = {}
    for row in image_rows:
        path = row.get("image_path")
        if not path:
            continue
        vec = embed_image(path)
        if vec is not None:
            out[row["ordinal"]] = vec
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both L2-normalized


def group_similarity(new_vecs: list[np.ndarray], stored_vecs: list[np.ndarray]) -> float:
    if not new_vecs or not stored_vecs:
        return 0.0
    scores = []
    for a in new_vecs:
        scores.append(max(cosine(a, b) for b in stored_vecs))
    return float(np.mean(scores))


def find_duplicate(new_vecs: list[np.ndarray], baseline: dict[str, list[np.ndarray]],
                   threshold: float = 0.95) -> tuple[bool, str | None]:
    """Compare new listing's vectors against stored listings' vector sets.

    baseline: {listing_id: [vec, ...]}. Returns (is_dup, matched_listing_id).
    """
    for listing_id, stored in baseline.items():
        if not stored:
            continue
        if group_similarity(new_vecs, stored) >= threshold:
            return True, listing_id
    return False, None


def aggregate_embedding(vecs: list[np.ndarray]) -> bytes | None:
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0).astype(np.float32).tobytes()
