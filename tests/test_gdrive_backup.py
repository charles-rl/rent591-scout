"""Offline tests for the native Google Drive backup module.

Nothing here touches the network. The google-* stack is imported lazily inside
the module, so this file (and CI, which installs only the light deps) can run
the full local archive/snapshot logic plus the Drive-side helpers against
duck-typed fake services. The real network path is validated by hand.
"""

import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest

from src.utils import gdrive_backup as gb


# --------------------------------------------------------------------------- #
# snapshot_db
# --------------------------------------------------------------------------- #
def _make_wal_db(path: Path, rows: int = 5) -> None:
    c = sqlite3.connect(str(path))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.executemany("INSERT INTO t (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    c.commit()
    c.close()


def _open_and_check(db: Path):
    c = sqlite3.connect(str(db))
    ok = c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    n = c.execute("SELECT count(*) FROM t").fetchone()[0]
    c.close()
    return ok, n


def test_snapshot_db_consistent_copy(tmp_path):
    src = tmp_path / "live.db"
    _make_wal_db(src, rows=7)
    dest = tmp_path / "snap.db"
    gb.snapshot_db(src, dest)

    assert dest.is_file()
    ok, n = _open_and_check(dest)
    assert ok is True
    assert n == 7


def test_snapshot_db_missing_source_raises(tmp_path):
    with pytest.raises(SystemExit):
        gb.snapshot_db(tmp_path / "nope.db", tmp_path / "out.db")


def test_snapshot_db_corrupt_dest_fails(tmp_path, monkeypatch):
    """A snapshot that fails quick_check is removed and aborts the run."""
    src = tmp_path / "live.db"
    _make_wal_db(src)
    dest = tmp_path / "snap.db"
    dest.touch()  # pre-exist so we can assert it is cleaned up

    class _Cur:
        def fetchone(self):
            return ("corrupt",)

    class _FakeConn:
        def execute(self, sql, *a):
            return _Cur()

        def backup(self, target, *a, **k):
            pass

        def close(self):
            pass

    class _FakeSqlite3:
        @staticmethod
        def connect(_path):
            return _FakeConn()

    monkeypatch.setattr(gb, "sqlite3", _FakeSqlite3)
    with pytest.raises(SystemExit):
        gb.snapshot_db(src, dest)
    assert not dest.exists()


# --------------------------------------------------------------------------- #
# _unique_stem / _parse_proxy
# --------------------------------------------------------------------------- #
def test_unique_stem_avoids_collision(tmp_path):
    first = gb._unique_stem(tmp_path, "20260101_000000", "db")
    first.touch()
    second = gb._unique_stem(tmp_path, "20260101_000000", "db")
    assert first.name == "591_scout_backup_db_20260101_000000.zip"
    assert second.name == "591_scout_backup_db_20260101_000000_1.zip"
    assert first != second


def test_parse_proxy_variants():
    assert gb._parse_proxy("http://127.0.0.1:8999") == ("127.0.0.1", 8999, "http")
    host, port, scheme = gb._parse_proxy("127.0.0.1:8999")
    assert (host, port) == ("127.0.0.1", 8999)
    assert scheme == "http"


# --------------------------------------------------------------------------- #
# build_zip
# --------------------------------------------------------------------------- #
def _fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "models").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "scripts" / "systemd").mkdir(parents=True)
    (root / "models" / "xgboost_head.json").write_text("{}", encoding="utf-8")
    (root / "docs" / "591research.md").write_text("# research", encoding="utf-8")
    (root / "README.md").write_text("# readme", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (root / "AGENTS.md").write_text("# agents", encoding="utf-8")
    (root / "scripts" / "systemd" / "rent591-incoming.timer").write_text("[Timer]", encoding="utf-8")
    return root


def test_build_zip_no_images(tmp_path):
    db = tmp_path / "apartments.db"
    _make_wal_db(db, rows=3)
    staging = tmp_path / "staging"
    root = _fake_root(tmp_path)

    zp = gb.build_zip(db, tmp_path / "images", staging, include_images=False, root=root, stamp="20260101_000000")
    assert zp.is_file() and zp.name == "591_scout_backup_db_20260101_000000.zip"

    names = zipfile.ZipFile(zp).namelist()
    assert "apartments.db" in names
    assert "docs/591research.md" in names
    assert "models/xgboost_head.json" in names
    assert not any(n.startswith("images/") for n in names)

    tmp = tempfile.mkdtemp()
    try:
        zipfile.ZipFile(zp).extract("apartments.db", tmp)
        c = sqlite3.connect(os.path.join(tmp, "apartments.db"))
        assert c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert c.execute("SELECT count(*) FROM t").fetchone()[0] == 3
        c.close()
    finally:
        shutil.rmtree(tmp)


def test_build_zip_with_images(tmp_path):
    db = tmp_path / "apartments.db"
    _make_wal_db(db, rows=1)
    images = tmp_path / "images"
    # mimic the real layout: data/images/<listing_id>/NN.webp
    for lid, n in (("111", 2), ("222", 1)):
        (images / lid).mkdir(parents=True)
        for i in range(n):
            (images / lid / f"{i:02d}.webp").write_bytes(b"RIFF-webp-fake-bytes")
    staging = tmp_path / "staging"
    root = _fake_root(tmp_path)

    zp = gb.build_zip(db, images, staging, include_images=True, root=root, stamp="20260101_000000")
    zf = zipfile.ZipFile(zp)
    infos = zf.infolist()

    img_names = sorted(i.filename for i in infos if i.filename.startswith("images/"))
    assert img_names == ["images/111/00.webp", "images/111/01.webp", "images/222/00.webp"]

    by_name = {i.filename: i.compress_type for i in infos}
    assert all(by_name[n] == 0 for n in img_names)  # webp stored (already compressed)
    assert by_name["apartments.db"] == 8  # deflated
    assert zf.testzip() is None  # every CRC valid


def test_build_zip_name_collision_suffix(tmp_path):
    db = tmp_path / "apartments.db"
    _make_wal_db(db, rows=1)
    staging = tmp_path / "staging"
    root = _fake_root(tmp_path)
    first = gb.build_zip(db, staging / "img", staging, include_images=False, root=root, stamp="S")
    second = gb.build_zip(db, staging / "img", staging, include_images=False, root=root, stamp="S")
    assert first.name == "591_scout_backup_db_S.zip"
    assert second.name == "591_scout_backup_db_S_1.zip"


def test_build_zip_missing_db_raises(tmp_path):
    with pytest.raises(SystemExit):
        gb.build_zip(tmp_path / "missing.db", tmp_path / "img", tmp_path / "staging", include_images=False)


# --------------------------------------------------------------------------- #
# Drive-side helpers against duck-typed fakes (no network)
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, data):
        self._d = data

    def execute(self):
        return self._d


class _FakeFiles:
    def __init__(self, store):
        self.store = store

    def list(self, **kwargs):
        return _Resp(self.store["list_result"])

    def create(self, body=None, **kwargs):
        self.store["created"] = body
        return _Resp(self.store["create_result"])

    def delete(self, fileId, **kwargs):
        self.store["deleted"].append(fileId)
        return _Resp({})


class _FakeService:
    def __init__(self, store):
        self.store = store

    def files(self):
        return _FakeFiles(self.store)


def _store(**kw):
    base = {"list_result": {"files": []}, "created": None, "create_result": {"id": "NEWF"}, "deleted": []}
    base.update(kw)
    return base


def test_find_folder_returns_existing():
    store = _store(list_result={"files": [{"id": "F1", "name": "591_Scout_Backups"}]})
    fid = gb._find_folder(_FakeService(store), "591_Scout_Backups")
    assert fid == "F1"
    assert store["created"] is None  # did not create a duplicate


def test_find_folder_creates_when_absent():
    store = _store(list_result={"files": []}, create_result={"id": "NEWF"})
    fid = gb._find_folder(_FakeService(store), "591_Scout_Backups")
    assert fid == "NEWF"
    assert store["created"]["mimeType"] == "application/vnd.google-apps.folder"
    assert store["created"]["name"] == "591_Scout_Backups"


def test_upload_to_drive_calls_create_with_folder(tmp_path, monkeypatch):
    zip_path = tmp_path / "591_scout_backup_X.zip"
    zip_path.write_bytes(b"PKzipbytes")

    captured = {}

    class _Chunked:
        def next_chunk(self):
            return (None, {"id": "FILEID", "name": zip_path.name, "size": "8"})

    class _Files:
        def create(self, body=None, media_body=None, fields=None, **kw):
            captured.update(body=body, media=media_body, fields=fields)
            return _Chunked()

    class _Svc:
        def files(self):
            return _Files()

    class _Media:
        def __init__(self, *a, **k):
            captured["media_kwargs"] = k

    monkeypatch.setattr(gb, "_load_or_auth", lambda *a, **k: "CREDS")
    monkeypatch.setattr(gb, "_make_authorized_http", lambda creds, proxy: "HTTP")
    monkeypatch.setattr(gb, "_build_service", lambda http, build: _Svc())
    monkeypatch.setattr(gb, "_find_folder", lambda service, folder: "FOLDER123")
    monkeypatch.setattr(gb, "_gdrive_imports", lambda: (None, None, None, None, _Media))

    fid = gb.upload_to_drive(zip_path, tmp_path / "c.json", tmp_path / "t.json", folder_name="591_Scout_Backups")
    assert fid == "FILEID"
    assert captured["body"] == {"name": zip_path.name, "parents": ["FOLDER123"]}
    assert captured["fields"] == "id, name, size"
    assert captured["media_kwargs"].get("resumable") is True


class _PruneStore:
    deleted: ClassVar[list] = []
    files: ClassVar[list] = [
        # three DB-tier snapshots (frequent) + one full (daily). A db-tier prune
        # must delete old db snapshots but never the full backup.
        {"id": "A", "name": "591_scout_backup_db_20260101_000000.zip", "createdTime": "2026-01-01T00:00:00Z"},
        {"id": "B", "name": "591_scout_backup_db_20260102_000000.zip", "createdTime": "2026-01-02T00:00:00Z"},
        {"id": "C", "name": "591_scout_backup_db_20260103_000000.zip", "createdTime": "2026-01-03T00:00:00Z"},
        {"id": "F", "name": "591_scout_backup_full_20260101_000000.zip", "createdTime": "2026-01-01T06:00:00Z"},
    ]


class _PruneFiles:
    def list(self, **kwargs):
        return _Resp({"files": _PruneStore.files})

    def delete(self, fileId, **kwargs):
        _PruneStore.deleted.append(fileId)
        return _Resp({})


class _PruneService:
    def files(self):
        return _PruneFiles()


def test_prune_drive_deletes_oldest_in_tier_only(tmp_path, monkeypatch):
    _PruneStore.deleted = []
    monkeypatch.setattr(gb, "_load_or_auth", lambda *a, **k: "CREDS")
    monkeypatch.setattr(gb, "_make_authorized_http", lambda creds, proxy: "HTTP")
    monkeypatch.setattr(gb, "_build_service", lambda http, build: _PruneService())
    monkeypatch.setattr(gb, "_find_folder", lambda service, folder: "FOLDER123")
    monkeypatch.setattr(gb, "_gdrive_imports", lambda: (None, None, None, None, None))

    # db tier, keep 2 of the 3 db snapshots -> delete the oldest db (A),
    # but NEVER the full backup F.
    pruned = gb.prune_drive(
        tmp_path / "c.json", tmp_path / "t.json", folder_name="591_Scout_Backups", keep=2, tier="db"
    )
    assert pruned == 1
    assert sorted(_PruneStore.deleted) == ["A"]
    assert "F" not in _PruneStore.deleted


def test_prune_drive_keep_zero_is_noop(tmp_path, monkeypatch):
    # keep=0 disables pruning before any Google code runs.
    monkeypatch.setattr(gb, "_gdrive_imports", lambda: pytest.fail("should not reach the google stack"))
    assert gb.prune_drive(tmp_path / "c.json", tmp_path / "t.json", keep=0) == 0


def test_tier_of_parses_names():
    assert gb._tier_of("591_scout_backup_db_20260101_000000.zip") == "db"
    assert gb._tier_of("591_scout_backup_full_20260101_000000.zip") == "full"
    assert gb._tier_of("591_scout_backup_20260101_000000.zip") is None  # legacy/un-tagged


def test_prune_full_tier_never_touches_db(tmp_path, monkeypatch):
    _PruneStore.deleted = []
    # give the store more full-tier files than db so a full prune has something to cut
    _PruneStore.files = [
        {"id": "F1", "name": "591_scout_backup_full_20260101_000000.zip", "createdTime": "2026-01-01T00:00:00Z"},
        {"id": "F2", "name": "591_scout_backup_full_20260102_000000.zip", "createdTime": "2026-01-02T00:00:00Z"},
        {"id": "F3", "name": "591_scout_backup_full_20260103_000000.zip", "createdTime": "2026-01-03T00:00:00Z"},
        {"id": "D1", "name": "591_scout_backup_db_20260103_000000.zip", "createdTime": "2026-01-03T00:30:00Z"},
    ]
    monkeypatch.setattr(gb, "_load_or_auth", lambda *a, **k: "CREDS")
    monkeypatch.setattr(gb, "_make_authorized_http", lambda creds, proxy: "HTTP")
    monkeypatch.setattr(gb, "_build_service", lambda http, build: _PruneService())
    monkeypatch.setattr(gb, "_find_folder", lambda service, folder: "FOLDER123")
    monkeypatch.setattr(gb, "_gdrive_imports", lambda: (None, None, None, None, None))

    pruned = gb.prune_drive(
        tmp_path / "c.json", tmp_path / "t.json", folder_name="591_Scout_Backups", keep=1, tier="full"
    )
    # keep newest full (F3); delete F1, F2; never the db snapshot D1.
    assert pruned == 2
    assert sorted(_PruneStore.deleted) == ["F1", "F2"]
    assert "D1" not in _PruneStore.deleted
