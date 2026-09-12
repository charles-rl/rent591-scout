"""Native Google Drive backup for the Rent591 scout state.

Packages the irreplaceable state (SQLite DB via the online-backup API, all
scraped listing images, trained model artifacts, docs and config) into a single
timestamped zip and uploads it to a dedicated ``591_Scout_Backups`` folder on
Google Drive. One zip per run avoids hitting Drive per-file API quotas with
thousands of individual images.

Standard libraries only for the local side (sqlite3, zipfile); Google-side
uses google-api-python-client + google-auth-oauthlib (lazy-imported so the
module imports — and offline tests run — without them installed).

Egress: works direct, or routed through the PC devtunnel proxy (PROXY_URL /
--proxy) on hosts whose firewall blocks Google. The devtunnel MITMs TLS, so
certificate validation is disabled on the proxied path.

Two backup tiers (one Drive folder, pruned independently by tier tag in the
filename — see scripts/systemd/rent591-gdb-*):
    db    --no-images: DB + models + docs only. Small (~20 MB), run every 30
              min (listings change constantly). Retention --keep-db (48 h).
    full  images included. Large (~300-550 MB), run daily. Retention
              --keep-full (14 days).

Usage:
    .venv/bin/python -m src.utils.gdrive_backup --auth-only   # once, in a browser host
    .venv/bin/python -m src.utils.gdrive_backup               # full backup (images) + upload
    .venv/bin/python -m src.utils.gdrive_backup --no-images   # DB-only backup (small/fast)
    .venv/bin/python -m src.utils.gdrive_backup --keep-db 96 --keep-full 14
    .venv/bin/python -m src.utils.gdrive_backup --proxy http://127.0.0.1:8999
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import urllib.parse
import zipfile
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths are resolved from the repo root (two levels up from src/utils/).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "apartments.db"
DEFAULT_IMAGES = ROOT / "data" / "images"
DEFAULT_STAGING = ROOT / "data" / "backups"
DEFAULT_CREDENTIALS = ROOT / "credentials.json"
DEFAULT_TOKEN = ROOT / "token.json"
DEFAULT_FOLDER = "591_Scout_Backups"
DEFAULT_KEEP = 10
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Backup tiers. Two cadences share one Drive folder but must prune
# independently: frequent small DB-only snapshots (listings change ~every 30
# min) and rarer full image-included zips. The tier tag in the filename lets
# prune_drive scope retention per tier so one cadence never eats the other's.
TIER_DB = "db"        # --no-images: DB + models + docs (small, frequent)
TIER_FULL = "full"    # images included (large, less frequent)
DEFAULT_KEEP_DB = 96      # 96 x 30 min = 48 h of DB-only snapshots
DEFAULT_KEEP_FULL = 14    # 14 daily full zips

# Small companion files bundled into the zip (relative to ROOT). models/ holds
# the retrainable heads; the multi-GB dinov3_cache is deliberately excluded
# (re-downloadable) but the two trained artifacts are included.
COMPANION_FILES = (
    "models/xgboost_head.json",
    "models/dino_probe.npz",
    "docs/591research.md",
    "README.md",
    "pyproject.toml",
    "scripts/systemd/rent591-incoming.service",
    "scripts/systemd/rent591-incoming.timer",
    "AGENTS.md",
)

# Files that are already compressed store raw in the zip (deflating wastes CPU
# and shrinks nothing); everything else is deflated.
STORED_SUFFIXES = {".webp", ".jpg", ".jpeg", ".png", ".gz", ".npz"}


# --------------------------------------------------------------------------- #
# Local snapshot + packaging (no Google deps)
# --------------------------------------------------------------------------- #
def snapshot_db(db_path: Path, dest: Path) -> Path:
    """Copy the live WAL-mode DB to `dest` via the online-backup API.

    Consistent and non-blocking (no VACUUM, no lock on the running scraper).
    Fails the run if the snapshot does not pass PRAGMA quick_check.
    """
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
        ok = dst.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        dst.close()
        src.close()
    if not ok:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"DB snapshot failed PRAGMA quick_check: {dest}")
    return dest


def _unique_stem(base: Path, stamp: str, tier: str) -> Path:
    """Avoid clobbering a zip made within the same second."""
    candidate = base / f"591_scout_backup_{tier}_{stamp}.zip"
    i = 0
    while candidate.exists():
        i += 1
        candidate = base / f"591_scout_backup_{tier}_{stamp}_{i}.zip"
    return candidate


def _add_stored(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    zf.write(path, arcname, compress_type=zipfile.ZIP_STORED)


def _add_deflated(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    zf.write(path, arcname, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def build_zip(
    db_path: Path,
    images_dir: Path,
    staging: Path,
    include_images: bool = True,
    root: Path = ROOT,
    stamp: str | None = None,
    tier: str | None = None,
) -> Path:
    """Snapshot the DB and bundle it (plus images/models/docs) into a zip.

    Returns the path of the created zip. Raises SystemExit on a bad DB
    snapshot; missing optional companions are skipped, a missing DB or an
    empty image dir is a hard error. `tier` tags the filename (db/full) so
    the two cadences can be pruned independently; defaults to the tier that
    matches `include_images`.
    """
    if not db_path.is_file():
        raise SystemExit(f"DB not found: {db_path}")
    tier = tier or (TIER_FULL if include_images else TIER_DB)

    staging.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_path = _unique_stem(staging, stamp, tier)

    # Snapshot the DB into the staging area first (never zip the live file).
    db_snap = staging / f".db_snapshot_{stamp}.db"
    snapshot_db(db_path, db_snap)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            _add_deflated(zf, db_snap, "apartments.db")

            if include_images:
                if images_dir.is_dir():
                    img_files = sorted(p for p in images_dir.rglob("*") if p.is_file())
                else:
                    logger.warning("images dir not found: %s", images_dir)
                    img_files = []
                for img in img_files:
                    arc = Path("images") / img.relative_to(images_dir)
                    if img.suffix.lower() in STORED_SUFFIXES:
                        _add_stored(zf, img, str(arc))
                    else:
                        _add_deflated(zf, img, str(arc))
                logger.info("packed %d images", len(img_files))

            for rel in COMPANION_FILES:
                p = root / rel
                if p.is_file():
                    _add_deflated(zf, p, rel)
                else:
                    logger.debug("companion file absent, skipping: %s", rel)
    finally:
        db_snap.unlink(missing_ok=True)

    size_mb = zip_path.stat().st_size / 1e6
    logger.info("archive ready: %s (%.1f MB)", zip_path.name, size_mb)
    return zip_path


# --------------------------------------------------------------------------- #
# Google Drive (lazy imports)
# --------------------------------------------------------------------------- #
def _gdrive_imports():
    """Import the Google stack; raise a clear error if not installed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as e:  # pragma: no cover - depends on env
        raise SystemExit(
            "Google Drive dependencies are missing. Install with:\n"
            "  uv pip install --python .venv/bin/python "
            "google-api-python-client google-auth-oauthlib google-auth-httplib2"
        ) from e
    return Request, Credentials, InstalledAppFlow, build, MediaFileUpload


def _build_service(http, build):
    return build("drive", "v3", http=http, cache_discovery=False)


def _make_auth_request(proxy: str | None):
    """A google.auth Request for token refresh, optionally through the proxy.

    The devtunnel MITMs TLS, so verification is disabled on the proxied path
    (same convention as src.utils.proxy_check).
    """
    from google.auth.transport.requests import Request

    if not proxy:
        return Request()
    import requests

    session = requests.Session()
    session.proxies = {"http": proxy, "https": proxy}
    session.verify = False
    return Request(session=session)


def _load_or_auth(
    credentials_path: Path,
    token_path: Path,
    Credentials,
    InstalledAppFlow,
    proxy: str | None = None,
):
    """Load token.json (refreshing if stale) or run the browser flow."""
    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds and not creds.expired:
            return creds
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(_make_auth_request(proxy))
                token_path.write_text(creds.to_json(), encoding="utf-8")
                return creds
            except Exception as e:  # fail-soft: fall through to re-auth
                logger.warning("token refresh failed (%s); re-authorizing", e)
                creds = None
    if not credentials_path.is_file():
        raise SystemExit(
            f"credentials.json not found at {credentials_path}. "
            "Create an OAuth 2.0 Desktop client in Google Cloud and save it there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
    flow.run_local_server(port=0)
    token_path.write_text(flow.credentials.to_json(), encoding="utf-8")
    logger.info("saved token to %s", token_path)
    return flow.credentials


def _make_authorized_http(creds, proxy: str | None):
    """Wrap creds in an AuthorizedHttp, optionally proxied through devtunnel."""
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp

    if proxy:
        host, port, _scheme = _parse_proxy(proxy)
        # socks.PROXY_TYPE_HTTP == 3; use the int to avoid a hard PySocks import
        # here (PySocks is still required for httplib2 to honor the proxy).
        proxy_info = httplib2.ProxyInfo(3, host, port)
        http = httplib2.Http(
            proxy_info=proxy_info, disable_ssl_certificate_validation=True
        )
    else:
        http = httplib2.Http()
    # Let googleapiclient handle resumable-upload status codes. An incomplete
    # chunk returns HTTP 308 (Resume Incomplete) with no Location header; httplib2
    # force-follows 303/308 and would otherwise raise RedirectMissingLocation
    # before MediaFileUpload can interpret the 308. Discovery + list return
    # plain 200, so disabling auto-redirect here is safe.
    http.follow_redirects = False
    return AuthorizedHttp(creds, http)


def _parse_proxy(proxy_url: str) -> tuple[str, int, str]:
    p = urllib.parse.urlparse(proxy_url if "://" in proxy_url else f"http://{proxy_url}")
    return p.hostname, p.port or 8999, p.scheme


def _find_folder(service, folder_name: str) -> str:
    """Return the id of the named root folder, creating it if absent."""
    res = (
        service.files()
        .list(
            spaces="drive",
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id, name, createdTime)",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = res.get("files", [])
    for f in files:
        if f["name"] == folder_name:
            return f["id"]
    created = (
        service.files()
        .create(
            body={
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    logger.info("created folder %r (id=%s)", folder_name, created["id"])
    return created["id"]


def upload_to_drive(
    zip_path: Path,
    credentials_path: Path,
    token_path: Path,
    folder_name: str = DEFAULT_FOLDER,
    proxy: str | None = None,
) -> str:
    """Upload `zip_path` into the (auto-created) Drive folder. Returns file id."""
    _Request, Credentials, InstalledAppFlow, build, MediaFileUpload = _gdrive_imports()
    creds = _load_or_auth(credentials_path, token_path, Credentials, InstalledAppFlow, proxy)
    http = _make_authorized_http(creds, proxy)
    service = _build_service(http, build)
    folder_id = _find_folder(service, folder_name)

    media = MediaFileUpload(str(zip_path), mimetype="application/zip", resumable=True, chunksize=8 * 1024 * 1024)
    body = {"name": zip_path.name, "parents": [folder_id]}
    request = service.files().create(
        body=body,
        media_body=media,
        fields="id, name, size",
        supportsAllDrives=True,
    )
    logger.info("uploading %s to %r ...", zip_path.name, folder_name)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status is not None:
            logger.info("  upload %.0f%%", status.progress() * 100)
    logger.info("uploaded: id=%s size=%s", response.get("id"), response.get("size"))
    return response.get("id")


def _tier_of(name: str) -> str | None:
    """Tier tag embedded in a backup filename, or None for untagged (legacy)."""
    for t in (TIER_DB, TIER_FULL):
        if f"_{t}_" in name:
            return t
    return None


def prune_drive(
    credentials_path: Path,
    token_path: Path,
    folder_name: str = DEFAULT_FOLDER,
    keep: int = DEFAULT_KEEP,
    tier: str | None = None,
    proxy: str | None = None,
) -> int:
    """Delete the oldest backups beyond the newest `keep`, scoped by `tier`.

    Each cadence only prunes its own tier (plus untagged legacy zips), so the
    frequent DB-only runs never delete full image backups, and vice versa.
    """
    if keep <= 0:
        return 0
    _Request, Credentials, InstalledAppFlow, build, _Media = _gdrive_imports()
    creds = _load_or_auth(credentials_path, token_path, Credentials, InstalledAppFlow, proxy)
    http = _make_authorized_http(creds, proxy)
    service = _build_service(http, build)
    folder_id = _find_folder(service, folder_name)

    res = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and mimeType='application/zip' and trashed=false",
            fields="files(id, name, createdTime, modifiedTime, size)",
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    owned = [f for f in res.get("files", []) if tier is None or _tier_of(f.get("name", "")) in (tier, None)]
    files = sorted(
        owned,
        key=lambda f: (f.get("createdTime", ""), f.get("name", "")),
        reverse=True,
    )
    to_delete = files[keep:]
    for f in to_delete:
        try:
            service.files().delete(
                fileId=f["id"], supportsAllDrives=True
            ).execute()
            logger.info("pruned old backup: %s", f["name"])
        except Exception as e:  # fail-soft: keep going, don't lose the run
            logger.warning("could not prune %s: %s", f["name"], e)
    return len(to_delete)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run_auth_only(credentials_path: Path, token_path: Path) -> None:
    _Request, Credentials, InstalledAppFlow, _build, _Media = _gdrive_imports()
    _load_or_auth(credentials_path, token_path, Credentials, InstalledAppFlow)
    print(f"Authorization complete. Token saved to: {token_path}")


def _default_proxy() -> str | None:
    # Mirror the pipeline convention: PROXY_URL set (non-empty) => route through it.
    p = os.environ.get("PROXY_URL", "").strip()
    return p or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--auth-only",
        action="store_true",
        help="Run the OAuth browser flow once to create token.json, then exit.",
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB to snapshot")
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="Listing images dir")
    ap.add_argument("--staging", type=Path, default=DEFAULT_STAGING, help="Where to write the zip")
    ap.add_argument(
        "--credentials", type=Path, default=DEFAULT_CREDENTIALS, help="OAuth client secrets"
    )
    ap.add_argument("--token", type=Path, default=DEFAULT_TOKEN, help="OAuth token cache")
    ap.add_argument("--folder", default=DEFAULT_FOLDER, help="Drive destination folder name")
    ap.add_argument(
        "--keep-db", type=int, default=DEFAULT_KEEP_DB,
        help=f"DB-only snapshots to keep on Drive (default {DEFAULT_KEEP_DB} = 48h at 30min cadence; 0 = keep all)",
    )
    ap.add_argument(
        "--keep-full", type=int, default=DEFAULT_KEEP_FULL,
        help=f"Full (image) backups to keep on Drive (default {DEFAULT_KEEP_FULL} = 14 days; 0 = keep all)",
    )
    ap.add_argument(
        "--tier", choices=[TIER_DB, TIER_FULL], default=None,
        help="Override the backup tier (defaults to 'db' when --no-images, else 'full')",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Skip images (DB + models + docs only) for a small, fast backup",
    )
    ap.add_argument(
        "--proxy",
        default=_default_proxy(),
        help="Route Google traffic through this proxy (e.g. http://127.0.0.1:8999); defaults to PROXY_URL",
    )
    ap.add_argument("--no-upload", action="store_true", help="Build the zip but do not upload (local-only test)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.auth_only:
        _run_auth_only(args.credentials, args.token)
        return 0

    include_images = not args.no_images
    tier = args.tier or (TIER_FULL if include_images else TIER_DB)
    keep = args.keep_full if tier == TIER_FULL else args.keep_db
    zip_path = build_zip(
        db_path=args.db,
        images_dir=args.images,
        staging=args.staging,
        include_images=include_images,
        root=ROOT,
        tier=tier,
    )
    print(f"Archive: {zip_path}")

    if args.no_upload:
        print("--no-upload set; leaving zip in place. Done.")
        return 0

    try:
        file_id = upload_to_drive(
            zip_path, args.credentials, args.token, folder_name=args.folder, proxy=args.proxy
        )
    except Exception as e:  # keep the local zip for retry on upload failure
        logger.error("upload failed, keeping local zip %s: %s", zip_path.name, e)
        return 1

    # Only delete the local temp zip after a confirmed upload.
    try:
        zip_path.unlink()
        print(f"Deleted local temp zip after successful upload (file_id={file_id}).")
    except OSError as e:
        logger.warning("could not delete local zip %s: %s", zip_path, e)

    pruned = prune_drive(
        args.credentials, args.token, folder_name=args.folder, keep=keep, tier=tier, proxy=args.proxy
    )
    print(f"Done. Upload OK (tier={tier}); pruned {pruned} old backup(s) (keeping {keep}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
