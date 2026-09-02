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
EMBED_DIM = 768
DINO_BATCH = max(1, int(os.environ.get("DINO_BATCH", "8")))

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


def release_gpu_memory() -> None:
    """Return cached CUDA blocks after batch inference (no-op on CPU-only runs)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # pragma: no cover - defensive, never breaks the pipeline
        logger.debug("cuda cache release skipped: %s", e)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def embed_paths_batch(paths: list[str]) -> list[np.ndarray | None]:
    """Batched DINOv3 CLS embeddings: one forward pass per DINO_BATCH images.

    Returns per-path L2-normalized 768-dim float32 vectors (None per failed image).
    """
    if not paths:
        return []
    import torch

    results: list[np.ndarray | None] = []
    try:
        model, processor = _load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for i in range(0, len(paths), DINO_BATCH):
            chunk = paths[i : i + DINO_BATCH]
            imgs: list = []
            idxs: list[int] = []
            chunk_out: list[np.ndarray | None] = [None] * len(chunk)
            for j, p in enumerate(chunk):
                try:
                    with Image.open(p) as raw:
                        imgs.append(raw.convert("RGB"))
                    idxs.append(j)
                except Exception as e:
                    logger.warning("open %s failed: %s", p, e)
            if not imgs:
                results.extend(chunk_out)
                continue
            inputs = processor(images=imgs, return_tensors="pt")
            for im in imgs:
                im.close()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                cls = model(**inputs).last_hidden_state[:, 0]
            raw = cls.float().cpu().numpy().reshape(len(imgs), -1)
            if raw.shape[1] != EMBED_DIM:
                logger.warning("embed batch returned %d dims, expected %d", raw.shape[1], EMBED_DIM)
                results.extend(chunk_out)
                continue
            for j, vec in zip(idxs, _normalize_rows(raw)):
                if np.isfinite(vec).all():  # never persist NaN/Inf vectors into SQLite
                    chunk_out[j] = vec
            results.extend(chunk_out)
    except Exception as e:
        logger.warning("batch embed failed (%d paths): %s", len(paths), e)
        results.extend([None] * (len(paths) - len(results)))
    finally:
        release_gpu_memory()
    return results


def embed_image(path: str) -> np.ndarray | None:
    """Return L2-normalized 768-dim float32 CLS embedding, or None on failure."""
    return embed_paths_batch([path])[0]


def embed_image_rows(image_rows: list[dict]) -> dict[int, np.ndarray]:
    """Embed a listing's image rows in batches; returns {ordinal: vec} for successful ones."""
    rows = [r for r in image_rows if r.get("image_path")]
    if not rows:
        return {}
    vecs = embed_paths_batch([r["image_path"] for r in rows])
    return {r["ordinal"]: v for r, v in zip(rows, vecs) if v is not None}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both L2-normalized


def group_similarity(new_vecs: list[np.ndarray], stored_vecs: list[np.ndarray]) -> float:
    """Group Similarity(A,B) = mean over a in A of max over b in B of cos(a, b).

    Vectorized as (N @ S^T).max(axis=1).mean() — inputs are already L2-normalized,
    so the matmul is the cosine matrix.
    """
    if not new_vecs or not stored_vecs:
        return 0.0
    n = np.stack(new_vecs)
    s = np.stack(stored_vecs)
    return float(np.clip((n @ s.T).max(axis=1).mean(), 0.0, 1.0))


def find_duplicate(new_vecs: list[np.ndarray], baseline: dict[str, list[np.ndarray]],
                   threshold: float = 0.95) -> tuple[bool, str | None]:
    """Compare new listing's vectors against stored listings' vector sets.

    baseline: {listing_id: [vec, ...]}. Returns (is_dup, matched_listing_id).
    """
    if not new_vecs:
        return False, None
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
