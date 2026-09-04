#!/usr/bin/env python
"""Backup the irreplaceable state: SQLite DB (+ WAL) and trained model artifacts.

User ratings, dynamic preferences and stored DINOv3 embeddings cannot be
re-derived once 591 delists a photo, and scripts/purge_noncompliant.py does
unguarded bulk DELETEs. Snapshot BEFORE destructive maintenance.

Usage:
    .venv/bin/python scripts/backup.py             # defaults, keep 10 newest
    .venv/bin/python scripts/backup.py --keep 5
    .venv/bin/python scripts/backup.py --out /some/other/disk

Backups land in data/backups/ (gitignored via data/*): rent591-<ts>.db plus
models-<ts>.tar.gz (xgboost head + dino probe; NOT models/dinov3_cache/, which
is re-downloadable and multi-GB).
"""

from __future__ import annotations

import argparse
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "apartments.db"
DEFAULT_OUT = ROOT / "data" / "backups"
MODEL_FILES = ("xgboost_head.json", "dino_probe.npz")


def backup_db(db_path: Path, out_dir: Path, stamp: str) -> Path:
    # sqlite3 online-backup API: consistent snapshot even in WAL mode, no VACUUM needed.
    dest = out_dir / f"rent591-{stamp}.db"
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
        ok = dst.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        dst.close()
        src.close()
    if not ok:
        dest.unlink()
        raise SystemExit(f"backup failed integrity quick_check: {dest}")
    return dest


def backup_models(out_dir: Path, stamp: str) -> Path | None:
    present = [p for name in MODEL_FILES if (p := ROOT / "models" / name).is_file()]
    if not present:
        return None
    dest = out_dir / f"models-{stamp}.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        for p in present:
            tar.add(p, arcname=f"models/{p.name}")
    return dest


def prune(out_dir: Path, keep: int) -> None:
    for pattern in ("rent591-*.db", "models-*.tar.gz"):
        old = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)[keep:]
        for p in old:
            p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--keep", type=int, default=10, help="snapshots to keep per artifact type")
    args = ap.parse_args()

    if not args.db.is_file():
        raise SystemExit(f"DB not found: {args.db}")
    args.out.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    while (args.out / f"rent591-{stamp}.db").exists():  # two backups within one second
        stamp += "b"
    db_dest = backup_db(args.db, args.out, stamp)
    models_dest = backup_models(args.out, stamp)
    prune(args.out, args.keep)

    print(f"DB backup:   {db_dest} ({db_dest.stat().st_size / 1e6:.1f} MB)")
    if models_dest:
        print(f"Models:      {models_dest}")
    else:
        print("Models:      none present, skipped")


if __name__ == "__main__":
    main()
