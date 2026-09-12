"""Offline tests for scripts/bootstrap.py restore + clean logic.

The restore path (install_from_zip / _rewrite_image_path) is fully parameterized
by data_dir, so it is exercised against a temp dir and a synthetic backup zip —
no network, no touching the live repo data. The clean SQL is checked in dry-run
against a temp DB.
"""

import json
import sqlite3
import zipfile

from scripts import bootstrap as boot


def _make_source_zip(tmp_path):
    """Build a synthetic backup zip whose DB image paths point at a *different*
    (old) machine, mimicking what a backup taken elsewhere looks like."""
    old_root = "/opt/oldmachine/rent591-scout"  # absolute, wrong machine
    data = tmp_path / "src"
    (data / "images" / "111").mkdir(parents=True)
    (data / "images" / "222").mkdir(parents=True)
    (data / "images" / "111" / "00.webp").write_bytes(b"RIFF-111-00")
    (data / "images" / "111" / "01.webp").write_bytes(b"RIFF-111-01")
    (data / "images" / "222" / "00.webp").write_bytes(b"RIFF-222-00")

    db = data / "apartments.db"
    c = sqlite3.connect(str(db))
    c.execute(
        "CREATE TABLE listings (listing_id TEXT PRIMARY KEY, is_duplicate BOOLEAN, "
        "image_status TEXT, image_urls JSON, image_paths JSON)"
    )
    c.execute(
        "CREATE TABLE listing_images (id INTEGER PRIMARY KEY, listing_id TEXT, "
        "ordinal INTEGER, image_path TEXT, UNIQUE(listing_id, ordinal))"
    )
    c.execute(
        "INSERT INTO listings VALUES (?, 0, 'completed', ?, ?)",
        ("111", json.dumps(["http://cdn/111.jpg"]),
         json.dumps([f"{old_root}/data/images/111/00.webp", f"{old_root}/data/images/111/01.webp"])),
    )
    c.execute(
        "INSERT INTO listings VALUES (?, 1, 'completed', ?, ?)",
        ("222", json.dumps(["http://cdn/222.jpg"]),
         json.dumps([f"{old_root}/data/images/222/00.webp"])),
    )
    c.execute("INSERT INTO listing_images (listing_id, ordinal, image_path) VALUES (?,?,?)",
              ("111", 0, f"{old_root}/data/images/111/00.webp"))
    c.execute("INSERT INTO listing_images (listing_id, ordinal, image_path) VALUES (?,?,?)",
              ("111", 1, f"{old_root}/data/images/111/01.webp"))
    c.execute("INSERT INTO listing_images (listing_id, ordinal, image_path) VALUES (?,?,?)",
              ("222", 0, f"{old_root}/data/images/222/00.webp"))
    c.commit()
    c.close()

    zip_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(db, "apartments.db")
        for rel in ("images/111/00.webp", "images/111/01.webp", "images/222/00.webp"):
            zf.write(data / rel, rel)
    return zip_path, old_root


def test_install_from_zip_extracts_and_rewrites_paths(tmp_path):
    zip_path, old_root = _make_source_zip(tmp_path)
    data_dir = tmp_path / "freshdata"

    n_listings, n_images, n_rewritten = boot.install_from_zip(zip_path, data_dir)

    assert n_listings == 2
    assert n_images == 3
    # DB installed + valid.
    db = data_dir / "apartments.db"
    c = sqlite3.connect(str(db))
    assert c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    # Image files extracted under the NEW data dir.
    assert (data_dir / "images" / "111" / "00.webp").is_file()
    assert (data_dir / "images" / "222" / "00.webp").is_file()
    # listing_images.image_path now points at the new machine, never the old one.
    for lid, ordinal, ipath in c.execute("SELECT listing_id, ordinal, image_path FROM listing_images"):
        assert ipath.startswith(str(data_dir)), ipath
        assert old_root not in ipath, ipath
    # listings.image_paths JSON array also rewritten.
    for lid, paths_json in c.execute("SELECT listing_id, image_paths FROM listings"):
        arr = json.loads(paths_json)
        assert all(p.startswith(str(data_dir)) and old_root not in p for p in arr)
    c.close()
    assert n_rewritten == 3  # all three listing_images rows were on the old machine


def test_rewrite_image_path_idempotent_and_variants(tmp_path):
    data_dir = tmp_path / "d"
    # already-correct path (idempotent re-run) is left alone.
    ok = f"{data_dir}/images/9/00.webp"
    assert boot._rewrite_image_path(ok, data_dir) == ok
    # absolute path on another machine -> remapped under data_dir.
    assert boot._rewrite_image_path("/x/data/images/9/00.webp", data_dir) == ok
    # relative path with images/ anchor -> remapped.
    assert boot._rewrite_image_path("images/9/00.webp", data_dir) == ok
    # None passes through.
    assert boot._rewrite_image_path(None, data_dir) is None


def test_doomed_sql_selects_dupes_and_imageless(tmp_path):
    db = tmp_path / "db.sqlite"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE listings (listing_id TEXT PRIMARY KEY, is_duplicate BOOLEAN, "
              "image_status TEXT, image_urls JSON, image_paths JSON)")
    c.execute("CREATE TABLE listing_images (listing_id TEXT, ordinal INTEGER)")
    # good: has images, not a dup -> keep
    c.execute("INSERT INTO listings VALUES ('keep', 0, 'completed', '[\"u\"]', '[\"p\"]')")
    c.execute("INSERT INTO listing_images VALUES ('keep', 0)")
    # duplicate -> doomed
    c.execute("INSERT INTO listings VALUES ('dup', 1, 'completed', '[\"u\"]', '[\"p\"]')")
    c.execute("INSERT INTO listing_images VALUES ('dup', 0)")
    # imageless: no rows, no urls, completed -> doomed
    c.execute("INSERT INTO listings VALUES ('noimg', 0, 'completed', '[]', NULL)")
    # imageless but still pending (in-flight) -> keep
    c.execute("INSERT INTO listings VALUES ('pend', 0, 'pending', '[]', NULL)")
    # imageless but has CDN urls (re-downloadable) -> keep
    c.execute("INSERT INTO listings VALUES ('urls', 0, 'skipped', '[\"http://cdn/x\"]', NULL)")
    c.commit()

    doomed = set(boot._doomed_ids(c))
    assert doomed == {"dup", "noimg"}, doomed
    c.close()
