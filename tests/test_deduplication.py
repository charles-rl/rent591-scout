"""Offline DINOv3 dedup tests: cached weights only (HF_HUB_OFFLINE), local images."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src import deduplication

ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("torch")
pytest.importorskip("transformers")
_source = deduplication._model_source()
if not (Path(_source).is_dir() and (Path(_source) / "model.safetensors").is_file()):
    pytest.skip("DINOv3 staged weights absent and HF offline — skipping model tests",
                allow_module_level=True)


def _synth(tmp_dir: Path, seed: int) -> str:
    """Structured deterministic image: DINOv3 CLS collapses on textureless pure noise
    (cos ~0.99 between unrelated noise images), so fixtures need real shapes/regions."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (256, 256), (235, 230, 220))
    dr = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = (int(v) for v in rng.integers(0, 190, 2))
        x1, y1 = x0 + int(rng.integers(30, 70)), y0 + int(rng.integers(30, 70))
        dr.rectangle([x0, y0, x1, y1], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    p = tmp_dir / f"synth_{seed}.webp"
    img.save(p, "WEBP")
    return str(p)


@pytest.fixture(scope="module")
def webp_set(tmp_path_factory):
    """{ref, dup_of_ref, unrelated}: real WebPs when data/images has >=2 listings, else synthetic."""
    d = tmp_path_factory.mktemp("webp")
    by_listing: dict[str, list[Path]] = {}
    for p in sorted((ROOT / "data" / "images").glob("*/*.webp")):
        by_listing.setdefault(p.parent.name, []).append(p)
    if len(by_listing) >= 2:
        (_lid_a, paths_a), (_lid_b, paths_b) = sorted(by_listing.items())[:2]
        ref = d / "ref.webp"
        dup = d / "dup.webp"
        ref.write_bytes(paths_a[0].read_bytes())
        dup.write_bytes(paths_a[0].read_bytes())
        unr = d / "unrelated.webp"
        unr.write_bytes(paths_b[0].read_bytes())
        return {"ref": str(ref), "dup": str(dup), "unrelated": str(unr), "source": "real"}
    return {
        "ref": _synth(d, 1),
        "dup": _synth(d, 1),  # same seed -> byte-identical image
        "unrelated": _synth(d, 99),
        "source": "synthetic",
    }


def test_embed_dims_norm_dtype(webp_set):
    vec = deduplication.embed_image(webp_set["ref"])
    assert vec is not None
    assert vec.shape == (768,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_embed_failure_returns_none(tmp_path):
    assert deduplication.embed_image(str(tmp_path / "missing.webp")) is None


def test_batch_alignment_with_bad_path(webp_set, tmp_path):
    out = deduplication.embed_paths_batch([webp_set["ref"], str(tmp_path / "bad.webp"), webp_set["unrelated"]])
    assert len(out) == 3
    assert out[0] is not None and out[1] is None and out[2] is not None


def test_group_similarity_bounds_and_dup(webp_set):
    ref = deduplication.embed_image(webp_set["ref"])
    dup = deduplication.embed_image(webp_set["dup"])
    unr = deduplication.embed_image(webp_set["unrelated"])
    dup_sim = deduplication.group_similarity([ref], [dup])
    unr_sim = deduplication.group_similarity([ref], [unr])
    assert 0.0 <= unr_sim <= dup_sim <= 1.0
    assert dup_sim >= 0.95  # identical image must trip the default threshold
    assert deduplication.group_similarity([], [ref]) == 0.0
    assert deduplication.group_similarity([ref], []) == 0.0


def test_find_duplicate(webp_set):
    ref = deduplication.embed_image(webp_set["ref"])
    dup = deduplication.embed_image(webp_set["dup"])
    unr = deduplication.embed_image(webp_set["unrelated"])
    baseline = {"OLD-1": [dup], "OLD-2": [unr]}
    is_dup, matched = deduplication.find_duplicate([ref], baseline)
    assert is_dup is True and matched == "OLD-1"
    is_dup, matched = deduplication.find_duplicate([unr], {"OLD-1": [ref]})
    assert is_dup is False and matched is None
    assert deduplication.find_duplicate([], baseline) == (False, None)


def test_vectorized_group_similarity_matches_naive():
    rng = np.random.default_rng(11)
    rows = rng.normal(size=(6, 768)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    new, stored = list(rows[:2]), list(rows[2:])
    naive = float(np.mean([max(float(np.dot(a, b)) for b in stored) for a in new]))
    assert deduplication.group_similarity(new, stored) == pytest.approx(naive, abs=1e-5)


def test_aggregate_embedding_roundtrip(webp_set):
    vecs = [v for v in (deduplication.embed_image(webp_set["ref"]), deduplication.embed_image(webp_set["unrelated"])) if v is not None]
    blob = deduplication.aggregate_embedding(vecs)
    assert len(blob) == 768 * 4
    agg = np.frombuffer(blob, dtype=np.float32)
    assert np.allclose(agg, np.mean(np.stack(vecs), axis=0))
    assert deduplication.aggregate_embedding([]) is None


def test_release_gpu_memory_no_exception():
    deduplication.release_gpu_memory()


def test_real_images_present(webp_set):
    assert Path(webp_set["ref"]).is_file()
