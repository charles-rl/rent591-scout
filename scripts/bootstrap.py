#!/usr/bin/env python3
"""Bootstrap a fresh Rent591 machine from the Google Drive backup.

A fresh clone has the tracked relay payloads (data/incoming/) but NO
data/apartments.db and NO data/images/. This tool rebuilds both from the
newest *full* (image-bearing) backup in the 591_Scout_Backups Drive folder:

  1. restore  — download the newest full zip, install apartments.db +
                data/images/, and rewrite the DB's ABSOLUTE image paths to
                THIS machine's location (they were stored absolute on the
                machine that made the backup).
  2. clean    — delete duplicate listings (is_duplicate=1) and imageless
                listings (no listing_images rows and no CDN source urls and not
                pending), plus their orphaned image files on disk.
  3. ingest   — (optional, --ingest) run `main.py --incoming` so listings not
                in the backup are pulled from the relay payloads, and (while
                the proxy is live) their images are drained from the internet.

Default (no flags) does restore + clean. Use --clean-only to skip restore,
--ingest to also ingest, --dry-run to preview the clean without deleting.

Run ONCE per fresh machine, after cloning, installing deps, and the one-time
`src.utils.gdrive_backup --auth-only` (which writes token.json). See
README_BACKUP.md "Fresh machine bootstrap".
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import gdrive_backup as gb

logger = logging.getLogger("bootstrap")
DATA = ROOT / "data"
DB = DATA / "apartments.db"
IMAGES_DIR = DATA / "images"


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #
def _latest_full_zip_id(service) -> str | None:
    """Drive id of the newest full-tier backup in the folder, or None."""
    folder_id = gb._find_folder(service, "591_Scout_Backups")
    res = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and mimeType='application/zip' and trashed=false",
            fields="files(id, name, createdTime, size)",
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    fulls = [f for f in res.get("files", []) if gb._tier_of(f.get("name", "")) == "full"]
    if not fulls:
        return None
    fulls.sort(key=lambda f: (f.get("createdTime", ""), f.get("name", "")), reverse=True)
    return fulls[0]["id"]


def _download_zip(service, zip_id: str, dest: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=zip_id, supportsAllDrives=True)
    with io.FileIO(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            status, done = dl.next_chunk()
            if status is not None:
                logger.info("  download %.0f%%", status.progress() * 100)


def _rewrite_image_path(p: str | None, data_dir: Path) -> str | None:
    """Rewrite a stored image path (possibly absolute on another machine) under data_dir."""
    if not p:
        return p
    # Keep already-correct paths (idempotent re-runs).
    if p.startswith(str(data_dir)):
        return p
    idx = p.rfind("images/")
    if idx == -1:
        return str(data_dir / p) if not Path(p).is_absolute() else p
    tail = p[idx:]  # 'images/<listing_id>/<nn>.webp'
    return str(data_dir / tail)


def _rewrite_db_paths(db_path: Path, data_dir: Path) -> tuple[int, int]:
    """Rewrite listing_images.image_path + listings.image_paths JSON under data_dir."""
    conn = sqlite3.connect(str(db_path))
    try:
        n = 0
        for lid, ordinal, ipath in conn.execute(
            "SELECT listing_id, ordinal, image_path FROM listing_images WHERE image_path IS NOT NULL"
        ).fetchall():
            new = _rewrite_image_path(ipath, data_dir)
            if new != ipath:
                if ordinal is None:
                    conn.execute(
                        "UPDATE listing_images SET image_path=? WHERE listing_id=? AND ordinal IS NULL",
                        (new, lid),
                    )
                else:
                    conn.execute(
                        "UPDATE listing_images SET image_path=? WHERE listing_id=? AND ordinal=?",
                        (new, lid, ordinal),
                    )
                n += 1
        # image_paths is a JSON array on listings.
        m = 0
        for lid, paths_json in conn.execute(
            "SELECT listing_id, image_paths FROM listings WHERE image_paths IS NOT NULL"
        ).fetchall():
            try:
                arr = json.loads(paths_json) if paths_json else []
            except (TypeError, ValueError):
                arr = []
            new_arr = [_rewrite_image_path(p, data_dir) for p in arr]
            if new_arr != arr:
                conn.execute(
                    "UPDATE listings SET image_paths=? WHERE listing_id=?",
                    (json.dumps(new_arr, ensure_ascii=False), lid),
                )
                m += 1
        conn.commit()
    finally:
        conn.close()
    return n, m


def install_from_zip(zip_path: Path, data_dir: Path) -> tuple[int, int, int]:
    """Extract apartments.db + images/** from a backup zip into data_dir and rewrite paths.

    Returns (listings, images_extracted, db_image_rows_rewritten). data_dir is
    created if missing; apartments.db is written at data_dir/apartments.db.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    db_dest = data_dir / "apartments.db"
    db_extracted = False
    images_extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name == "apartments.db":
                with zf.open(name) as src, open(db_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                db_extracted = True
            elif name.startswith("images/") and not name.endswith("/"):
                dest = data_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                images_extracted += 1
    if not db_extracted:
        raise SystemExit("backup zip has no apartments.db")

    n_img, _n_list = _rewrite_db_paths(db_dest, data_dir)
    c = sqlite3.connect(str(db_dest))
    ok = c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    n_listings = c.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
    c.close()
    if not ok:
        raise SystemExit(f"restored DB failed PRAGMA quick_check: {db_dest}")
    return n_listings, images_extracted, n_img


def cmd_restore(force: bool, proxy: str | None) -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if DB.exists() and not force:
        logger.error(
            "refusing to restore over existing DB %s (not a fresh machine?). "
            "Use --force to overwrite it.", DB
        )
        return 1

    _Request, Credentials, InstalledAppFlow, build, _Media = gb._gdrive_imports()
    creds = gb._load_or_auth(gb.DEFAULT_CREDENTIALS, gb.DEFAULT_TOKEN, Credentials, InstalledAppFlow, proxy)
    http = gb._make_authorized_http(creds, proxy)
    service = gb._build_service(http, build)

    zip_id = _latest_full_zip_id(service)
    if not zip_id:
        logger.error("no full-tier backup found on Drive (run `src.utils.gdrive_backup` first)")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="gdb_restore_"))
    zip_path = tmp / "backup.zip"
    logger.info("downloading newest full backup ...")
    try:
        _download_zip(service, zip_id, zip_path)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"download failed: {e}")
    logger.info("downloaded %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1e6)

    try:
        n_listings, images_extracted, n_img = install_from_zip(zip_path, DATA)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    logger.info(
        "restore OK: %d listings, %d image files, rewrote %d image rows",
        n_listings, images_extracted, n_img,
    )
    return 0


# --------------------------------------------------------------------------- #
# clean
# --------------------------------------------------------------------------- #
# A listing is "imageless" when it has no listing_images rows AND no CDN source
# urls to re-download from AND is not currently being drained (pending).
_DOOMED_SQL = """
SELECT l.listing_id
FROM listings l
WHERE l.is_duplicate = 1
   OR (
        NOT EXISTS (SELECT 1 FROM listing_images i WHERE i.listing_id = l.listing_id)
        AND IFNULL(l.image_status, '') NOT IN ('pending')
        AND (l.image_urls IS NULL OR l.image_urls IN ('', '[]'))
      )
"""


def _doomed_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(_DOOMED_SQL).fetchall()]


def cmd_clean(dry_run: bool) -> int:
    if not DB.exists():
        logger.error("no database at %s (run restore first, or --clean-only after restore)", DB)
        return 1
    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        before = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        doomed = _doomed_ids(conn)
        dup = [
            r[0]
            for r in conn.execute("SELECT listing_id FROM listings WHERE is_duplicate = 1")
        ]
        noimg = [lid for lid in doomed if lid not in set(dup)]
        logger.info(
            "clean: %d doomed (%d duplicates, %d imageless) of %d total",
            len(doomed), len(dup), len(noimg), before,
        )
        if dry_run:
            for lid in doomed[:20]:
                print(f"  would delete {lid}")
            if len(doomed) > 20:
                print(f"  ... and {len(doomed) - 20} more")
            logger.info("dry-run: nothing deleted")
            return 0

        if doomed:
            q = ",".join("?" * len(doomed))
            conn.execute(f"DELETE FROM listing_images WHERE listing_id IN ({q})", doomed)
            conn.execute(f"DELETE FROM relay_state WHERE listing_id IN ({q})", doomed)
            conn.execute(f"DELETE FROM listings WHERE listing_id IN ({q})", doomed)
            conn.commit()

        # Remove orphaned image files on disk (per-listing dirs).
        removed_dirs = 0
        for lid in doomed:
            d = IMAGES_DIR / lid
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed_dirs += 1

        after = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        logger.info(
            "clean OK: removed %d listings (%d duplicates, %d imageless), %d image dirs; %d remain",
            len(doomed), len(dup), len(noimg), removed_dirs, after,
        )
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# ingest (new listings from the internet / relay)
# --------------------------------------------------------------------------- #
def cmd_ingest() -> int:
    """Run the hybrid incoming pipeline: pulls relay payloads, and (while the
    proxy is live) drains pending images from the internet + scores them."""
    logger.info("running `main.py --incoming` to fetch new listings ...")
    proc = subprocess.run([sys.executable, str(ROOT / "main.py"), "--incoming"], check=False)
    return proc.returncode


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--clean-only", action="store_true", help="skip restore (clean an existing DB)")
    ap.add_argument("--ingest", action="store_true", help="after restore+clean, run main.py --incoming for new listings")
    ap.add_argument("--force", action="store_true", help="overwrite an existing data/apartments.db during restore")
    ap.add_argument("--dry-run", action="store_true", help="preview the clean deletions without deleting")
    ap.add_argument(
        "--proxy", default=gb._default_proxy(),
        help="Route Google traffic through this proxy (e.g. http://127.0.0.1:8999); defaults to PROXY_URL",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    rc = 0
    if not args.clean_only:
        rc = cmd_restore(args.force, args.proxy)
        if rc != 0:
            return rc
    rc = cmd_clean(args.dry_run)
    if rc != 0:
        return rc
    if args.ingest:
        rc = cmd_ingest()
    return rc


if __name__ == "__main__":
    sys.exit(main())
