import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def hermetic_model_paths(tmp_path, monkeypatch):
    """Point every model artifact at tmp_path for the whole session.

    `scoring.train_and_save` calls `bathroom_probe.train_and_save`, which WRITES
    PROBE_PATH via np.savez. Fixtures that pinned only `visual_preference.PROBE_PATH`
    left the real `models/bathroom_probe.npz` one fixture change away from being
    overwritten with test-shaped weights.
    """
    from src import bathroom_probe, scoring, visual_preference

    monkeypatch.setattr(scoring, "MODEL_PATH", tmp_path / "hermetic" / "xgboost_head.json")
    monkeypatch.setattr(visual_preference, "PROBE_PATH", tmp_path / "hermetic" / "dino_probe.npz")
    monkeypatch.setattr(bathroom_probe, "PROBE_PATH", tmp_path / "hermetic" / "bathroom_probe.npz")
