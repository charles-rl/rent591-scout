"""DINOv3 feature extraction + multi-image group cosine deduplication.

Model: DINOv3 ViT-B/16 ("dinov3-vit-base", 768-dim CLS embeddings), Meta's
facebook/dinov3-vitb16-pretrain-lvd1689m weights, cached under models/dinov3_cache.
Group similarity follows the outline: mean over new vectors of max cosine over stored set.

DINOv3 upgrades over DINOv2 (facebook/dinov2-base):
- RoPE positional embeddings replace learnable absolute positions -> resolution-independent,
  higher spatial detail recognition for layout photos.
- Register tokens + improved dense patch features -> better preservation of fine bathroom/
  fixture details used by the group-cosine dedup signal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "models" / "dinov3_cache"
LOCAL_MODEL_DIRNAME = "facebook_dinov3-vit-base"  # local canonical name for dinov3-vit-base
HUB_FALLBACK_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

_MODEL = None
_PROCESSOR = None


def _model_source() -> str:
    """Resolve where to load the DINOv3 checkpoint from (offline-first)."""
    override = os.environ.get("DINOV3_MODEL_PATH")
    if override and Path(override).is_dir():
        return override
    cache_dir = Path(os.environ.get("DINOV3_CACHE", str(DEFAULT_CACHE)))
    local = cache_dir / LOCAL_MODEL_DIRNAME
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)  # offline: pre-staged weights under models/dinov3_cache/
    return HUB_FALLBACK_REPO_ID  # online: pulled from HF into the same cache dir


def _load():
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    import torch
    from transformers import AutoImageProcessor, AutoModel

    source = _model_source()
    load_kwargs = {} if (Path(source).is_dir()) else {"cache_dir": str(Path(os.environ.get("DINOV3_CACHE", str(DEFAULT_CACHE))))}
    _PROCESSOR = AutoImageProcessor.from_pretrained(source, **load_kwargs)
    _MODEL = AutoModel.from_pretrained(source, **load_kwargs)
    _MODEL.eval()
    if torch.cuda.is_available():
        _MODEL = _MODEL.to("cuda")
    logger.info("DINOv3 loaded from %s", source)
    return _MODEL, _PROCESSOR


def embed_image(path: str) -> np.ndarray | None:
    """Return L2-normalized 768-dim float32 CLS embedding, or None on failure."""
    import torch

    try:
        model, processor = _load()
        with Image.open(path) as raw:
            img = raw.convert("RGB")
        inputs = processor(images=[img], return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).last_hidden_state[:, 0]
        vec = out.cpu().numpy().astype(np.float32).reshape(-1)
        if vec.shape[0] != 768:
            logger.warning("embed %s returned %d dims, expected 768", path, vec.shape[0])
            return None
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None
    except Exception as e:
        logger.warning("embed %s failed: %s", path, e)
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
