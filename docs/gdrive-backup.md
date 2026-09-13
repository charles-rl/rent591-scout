# Google Drive Backup

Native (no `rclone`) backup of the Rent591 scout state to Google Drive.
Packages the irreplaceable data into one timestamped `.zip` and uploads it to a
dedicated `591_Scout_Backups` folder. One zip per run keeps you under Drive's
per-file API quotas (5,000+ individual images would otherwise be thousands of
upload requests).

Everything is standard Python on the local side (`sqlite3`, `zipfile`); the
Google side uses `google-api-python-client` + `google-auth-oauthlib`.

## What gets backed up

| Item | Source | Notes |
|---|---|---|
| SQLite DB | `data/apartments.db` | Consistent, non-blocking copy via the SQLite **online-backup API** — the running scraper is never locked or interrupted. Validated with `PRAGMA quick_check`. |
| Listing images | `data/images/**` | All scraped `.webp` (stored uncompressed in the zip). Disable with `--no-images`. |
| Model weights | `models/xgboost_head.json`, `models/dino_probe.npz` | Included when present. The multi-GB `models/dinov3_cache/` is excluded (re-downloadable). |
| Docs & config | `docs/591research.md`, `README.md`, `AGENTS.md`, `pyproject.toml`, `scripts/systemd/*` | Config is env-var driven, so the systemd units + docs are bundled to capture it. |

The resulting archive is named `591_scout_backup_YYYYMMDD_HHMMSS.zip` (UTC).

## Setup

### 1. Dependencies

Already declared in `pyproject.toml`. On a fresh clone:

```bash
uv pip install --python .venv/bin/python google-api-python-client google-auth-oauthlib google-auth-httplib2 PySocks
```

`PySocks` is only needed if you route through the devtunnel proxy (`--proxy`);
direct egress works without it.

### 2. OAuth client (`credentials.json`)

You need an OAuth **2.0 client ID** of type **Desktop app** (an installed app
with redirect URI `http://localhost`). It is already present in this repo as
`credentials.json` (gitignored). To create a fresh one:

1. Go to <https://console.cloud.google.com/apis/credentials> (project: `rent-591-scout-backup`).
2. **Credentials → Create credentials → OAuth client ID → Desktop app**.
3. Download the JSON and save it as `credentials.json` in the repo root.
   - The OAuth consent screen for the **Drive API** must be authorized (Test
     mode is fine — tokens last ~7 days in Test mode, ~1 year in Published).
4. Enable the **Drive API** for the project if not already.

### 3. Authorize once (`token.json`)

Run the interactive browser flow **once**, on a machine with a browser:

```bash
.venv/bin/python -m src.utils.gdrive_backup --auth-only
```

It opens a Google sign-in page; on success it writes `token.json` (gitignored)
to the repo root. Copy that `token.json` to any other host that runs backups
(e.g., the GPU box). Subsequent runs reuse and auto-refresh it.

> Headless note: `--auth-only` uses the browser (local-server) flow. Run it on
> the PC and copy `token.json` over — that is the supported path.

## Two backup tiers

Two cadences share the single `591_Scout_Backups` folder but are pruned
independently — the tier is encoded in the filename (`591_scout_backup_<tier>_<ts>.zip`)
so one cadence never deletes the other's backups.

| Tier | Contents | Size | Cadence | Retention flag (default) |
|---|---|---|---|---|
| `db` | DB + models + docs (no images) | ~20 MB | every 30 min (listings change constantly) | `--keep-db` (96 ≈ 48 h) |
| `full` | DB + **all images** + models + docs | ~300–550 MB | daily | `--keep-full` (14 days) |

Images are re-downloadable from 591 and accumulate ~200/day, so the full tier
runs less often; the always-changing DB is captured frequently and cheaply.

## Usage

```bash
# Full backup (DB + all images + models + docs) -> upload -> prune old
.venv/bin/python -m src.utils.gdrive_backup

# Small/fast backup: DB + models + docs only (the frequent tier)
.venv/bin/python -m src.utils.gdrive_backup --no-images

# Adjust retention per tier (0 disables pruning for that tier)
.venv/bin/python -m src.utils.gdrive_backup --no-images --keep-db 48
.venv/bin/python -m src.utils.gdrive_backup --keep-full 7

# Build the zip locally without uploading (test the packaging only)
.venv/bin/python -m src.utils.gdrive_backup --no-upload

# Route Google traffic through the PC devtunnel (firewalled hosts)
.venv/bin/python -m src.utils.gdrive_backup --proxy http://127.0.0.1:8999
```

Common flags:

| Flag | Default | Purpose |
|---|---|---|
| `--auth-only` | off | Run the browser flow to create `token.json`, then exit. |
| `--no-images` | off | DB tier: skip `data/images` for a small, fast backup. |
| `--keep-db` | `96` | DB-only snapshots to keep on Drive (48 h at 30 min cadence). `0` = keep all. |
| `--keep-full` | `14` | Full (image) backups to keep on Drive (14 days). `0` = keep all. |
| `--tier {db,full}` | from `--no-images` | Force a tier tag regardless of `--no-images`. |
| `--no-upload` | off | Build the zip, do not upload. |
| `--folder NAME` | `591_Scout_Backups` | Destination folder (auto-created at Drive root). |
| `--proxy URL` | `$PROXY_URL` or direct | HTTP(S) proxy for all Google traffic (tunnel MITMs TLS → cert validation is disabled on this path). |
| `--db`, `--images`, `--staging` | `data/apartments.db`, `data/images`, `data/backups` | Override paths. |
| `-v` | off | Debug logging. |

### Behavior & safety

- The local temp zip is written to `data/backups/` and **deleted only after a
  confirmed upload**. If the upload fails the zip is kept (for retry) and the
  process exits non-zero.
- **WAL-safe DB snapshot.** The live DB runs in WAL mode with `-wal`/`-shm`
  sidecars that change continuously. The backup uses SQLite's **online-backup
  API** (`sqlite3` `src.backup(dst)`), not a file copy, so the snapshot is a
  single self-contained file that **includes the contents of the WAL** (no
  `-wal` file needed to open it) and passes `PRAGMA quick_check`. The scraper
  is never locked, paused, or interrupted. (A naive `cp` of the `.db` file
  would silently drop uncommitted WAL — the online-backup API does not.)
- The `591_Scout_Backups` folder is created automatically if missing.
- Only `drive.file` scope is used — the token can touch only files this app
  created, not the rest of your Drive.
- Uploads use Google's **resumable** protocol (8 MB chunks), so a dropped
  connection resumes rather than restarting a 500 MB transfer.

## Run on a schedule

Two systemd unit pairs are provided under `scripts/systemd/` (mirroring the
existing `rent591-incoming.*` convention):

| Units | Tier | Cadence | Runs |
|---|---|---|---|
| `rent591-gdb-db.{service,timer}` | db | every 30 min | `--no-images` (keeps 48 h) |
| `rent591-gdb-full.{service,timer}` | full | daily | full (keeps 14 days) |

Install and enable both (adjust the `WorkingDirectory`/`ExecStart` path if the
repo lives elsewhere):

```bash
sudo cp scripts/systemd/rent591-gdb-*.service scripts/systemd/rent591-gdb-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rent591-gdb-db.timer rent591-gdb-full.timer
systemctl list-timers | grep rent591   # confirm both are scheduled
```

`Persistent=true` means a missed run (host was off) fires shortly after boot.
The `db` timer starts ~6 min after boot so the first run doesn't race the
incoming-ingest timer for the same DB (both are read-safe, but this spreads
peak load). Because a full zip is ~300–550 MB, it runs daily; the always-changing
DB is captured every 30 min at ~20 MB.

## Fresh machine bootstrap

A fresh clone has the tracked relay payloads (`data/incoming/`) but no
`data/apartments.db` and no `data/images/`. `scripts/bootstrap.py` rebuilds both
from the newest **full** backup on Drive, then cleans, then (optionally) pulls
new listings from the internet:

```bash
# 0) one-time auth (on a machine with a browser), then copy token.json over
.venv/bin/python -m src.utils.gdrive_backup --auth-only

# 1) restore DB + images from the newest full backup, rewrite image paths to
#    THIS machine, delete duplicate + imageless listings, then ingest new ones
.venv/bin/python scripts/bootstrap.py --ingest

# Or step by step:
.venv/bin/python scripts/bootstrap.py                 # restore + clean (no ingest)
.venv/bin/python scripts/bootstrap.py --clean-only    # clean an existing DB
.venv/bin/python scripts/bootstrap.py --dry-run       # preview the clean (no deletes)
.venv/bin/python scripts/bootstrap.py --force         # overwrite an existing DB
```

What each phase does:
- **restore** — downloads the newest full (image) zip, installs
  `data/apartments.db` + `data/images/`, and rewrites every stored image path
  (stored absolute on the machine that made the backup) to this machine's
  location. Refuses to clobber an existing DB unless `--force`.
- **clean** — deletes duplicate listings (`is_duplicate=1`) and imageless
  listings (no `listing_images` rows, no CDN source urls, and not `pending`),
  plus their orphaned image files on disk.
- **ingest** (`--ingest`) — runs `main.py --incoming`: pulls any listings not in
  the backup from `data/incoming/`, and (while the proxy is live) drains their
  images from the internet and scores them.

The goal: a fresh machine ends up with the complete set of images for old
listings (from the backup) **and** new listings (from the internet), with
duplicates and imageless entries removed.

> Note: restore restores listings/images as of the newest full backup. Listings
> that were delisted from 591 before the backup was taken will not be re-fetched
> (591 only serves active listings) — they are only as current as the backup.

## Tests

```bash
.venv/bin/python -m pytest tests/test_gdrive_backup.py tests/test_bootstrap.py -q
```

Fully offline: DB snapshotting, zip packaging (arcnames, compression, CRCs),
name-collision handling, the Drive folder/upload/prune logic against
duck-typed fakes, and bootstrap restore (extract + path rewrite) / clean SQL
against a temp dir and a synthetic backup zip. The real network/upload path is
not exercised in tests.
