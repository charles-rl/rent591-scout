"""The three trained model artifacts must be bundled by BOTH backup paths.

`scripts/backup.py` (local tarball) and `src/utils/gdrive_backup.py` (Drive zip,
what scripts/bootstrap.py restores from) drift independently; an omitted
`bathroom_probe.npz` silently means a restored machine serves bath_model_score=0.0
for every listing while the restored head was fit on real values.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_model_artifact_lists_agree_across_backup_paths():
    backup = _load("scripts_backup", "scripts/backup.py")
    from src.utils import gdrive_backup

    local = {Path(n).name for n in backup.MODEL_FILES}
    drive = {Path(n).name for n in gdrive_backup.COMPANION_FILES if n.startswith("models/")}
    assert local == drive, f"scripts/backup.py {sorted(local)} != gdrive_backup {sorted(drive)}"
    assert {"xgboost_head.json", "dino_probe.npz", "bathroom_probe.npz"} <= local
