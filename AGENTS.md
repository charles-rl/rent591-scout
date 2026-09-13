# AGENTS.md

Rent591 rental scout: 591.com.tw ingestion → DINOv3 dedup → Qwen vision (Ollama) → XGBoost score → ntfy alerts → SQLite. Python 3.12, uv-managed `.venv`.

## Network topology (verified 2026-09-13: this host has direct egress)

- This host reaches `rent.591.com.tw` / `ntfy.sh` directly (curl 301/200). `scripts/run_incoming.sh` defaults `PROXY_URL=` (empty) → direct-egress mode in `src/utils/proxy_check.py`. Set `PROXY_URL=http://127.0.0.1:8999` for the PC devtunnel hybrid mode (images + ntfy via tunnel); proxy offline → listings stay `pending`, never crash.
- GitHub Actions relay (`.github/workflows/scrape_relay.yml`) still scrapes on cron (tiers every 15 min / hourly / 3 h, `docs/591research.md` §5) and commits raw payloads + WebP to `data/incoming/` (`auto(ingest): ... [skip ci]`, rebased before push). Never hand-edit `data/incoming/`. Repo var `RELAY_SKIP_IMAGES=true` = text-only relay (591 image CDN 403s datacenter runner IPs).
- Relay idempotency: per-listing `payload_sha256` + `relay_state` table; both sides skip unchanged.
- Hard filters live in `src/ingestion.py` `passes_hard_filters` (price 10000–17000 via `HARD_PRICE_MIN`/`HARD_PRICE_MAX`, ≥6 ping, kind 獨立套房/分租套房, cooking/gender exclusions) and are enforced on **both** relay side (`dump_relay_payloads`) and incoming side — change both or payloads mismatch. `ENFORCE_HARD_FILTERS=0` disables.

## Commands

```bash
.venv/bin/python -m pytest -q                 # full suite (~2 min locally incl. DINOv3; fully offline)
.venv/bin/python -m pytest tests/test_x.py -q # single file
.venv/bin/python main.py --fixtures --limit 3 # offline E2E (needs external/mcp-591/tests/fixtures)
.venv/bin/python main.py --incoming           # hybrid relay run (git payloads → local pipeline)
.venv/bin/python main.py --train              # retrain XGBoost head
.venv/bin/python rate.py --id N --score 4     # rate listing (auto-retrains past RATED_THRESHOLD=20)
.venv/bin/python -m ruff check .              # lint — clean at HEAD, keep it that way
.venv/bin/python scripts/backup.py            # snapshot DB+model heads → data/backups/ (run before purge_noncompliant.py)
.venv/bin/python scripts/backfill_vision.py   # resumable Qwen re-run after prompt changes (~200s/listing)
bash scripts/run_incoming.sh                  # git pull --ff-only + --incoming (what the systemd timer runs)
gh workflow run scrape_relay.yml              # manual relay trigger
```

Dedup test fixtures need structured shapes: DINOv3 CLS collapses on textureless synthetic noise (cos ~0.99 between unrelated noise images).

## Repo gotchas

- `external/`, `models/`, and `data/*` (except `data/incoming/`) are gitignored but required locally: vendored `external/mcp-591` (fixtures live in the former) + `external/591scraper`; `models/dinov3_cache/` (cached weights — DINOv3 tests self-skip without them). Model heads (`xgboost_head.json`, `dino_probe.npz`) auto-train from the DB.
- `requirements.txt` is **relay-runner-only** (requests + Pillow). Heavy deps (torch, transformers, xgboost, DrissionPage) are in `pyproject.toml` and must NOT be added to requirements.txt.
- DB is `data/apartments.db` (WAL). Schema migrations: `src/database.py` `_EXTRA_COLUMNS` (listings) and `_IMAGE_EXTRA_COLUMNS` (listing_images), idempotent `ALTER TABLE`.
- All pipeline stages fail-soft by design — broad `except Exception` is intentional (ruff `BLE001` ignored, line-length 110). Don't "fix" it.
- ntfy headers must be ASCII/latin-1: score rendered as `(x.xx/5)` (2 decimals), never `★` (breaks latin-1 encoding).
- Scoring: XGBoost's input is the compressed fusion vector `scoring.FEATURE_NAMES` (dino_visual_score + qwen_score + flags + tabular + bath_model_score), never the raw 768-d blob. Phase 1 (≤20 ratings) = Qwen direct; phase 2 (>20) = XGBoost. Changing the vector width auto-retrains `models/xgboost_head.json` + `models/dino_probe.npz`.
- Bathroom layer: `src/bathroom_detect.py` labels bathroom photos via Qwen → `listing_images.is_bathroom`; `src/bathroom_probe.py` ridge-maps pooled bathroom DINO embeddings to `listings.bath_model_score` (0.0 = no bathroom photo, distinct from a bad 1/5), which feeds the fusion vector AND the vision prompt (`vision_llm.build_messages`). Seed/retrain: `scripts/backfill_bathrooms.py` (incremental — only labels `is_bathroom IS NULL`).
- All user-facing alert text is English: keep new warning strings English or add them to `notifier._WARN_EN` (fixed translation table for legacy Chinese DB rows); the vision prompt demands English warnings; the consolidate-preferences prompt also forces English bullets.
- Proxy traffic uses `verify=False` (devtunnel MITMs TLS with a cert Python rejects; `PROXY_SSL_VERIFY=1` re-enables). 591 CDN 403s original photo URLs via the tunnel → fetch `!fit.1000x.water2.jpg` resize variants (`PROXY_IMAGE_SUFFIX`). CDN 502 storms = throttling (backoff, stay `pending`), not a dead proxy.
- Tests must stay offline: `HF_HUB_OFFLINE=1`, cached DINOv3 weights, fixture replay, mocked Ollama/ntfy/proxy. `conftest.py` inserts repo root into `sys.path`.
- CI (`ci.yml`) installs light deps only (pytest/ruff/numpy/xgboost/Pillow/requests) — never torch there: real-DINOv3 tests self-skip at module level when torch or the staged weights are missing.
- Local cadence is NOT automatic: `scripts/systemd/rent591-incoming.{service,timer}` (30-min cycle) runs `run_incoming.sh` = git pull --ff-only + `--incoming` (pull failure tolerated); no-systemd hosts use `scripts/run_scheduler.sh`.
- Only the DB holds non-re-derivable data (ratings/comments, `dynamic_preferences`, DINOv3 embeddings of soon-delisted photos); `models/` auto-retrains from it. Run `scripts/backup.py` before `scripts/purge_noncompliant.py` (deletes with no confirmation). GDrive backup + fresh-machine bootstrap: `docs/gdrive-backup.md`, `scripts/bootstrap.py`.
- Env config is read at import time in `main.py`/`src/` (module-level `os.environ`) — README "Setup" has the full var list (`X591_*`, `PROXY_*`, `NTFY_TOPIC`, `OLLAMA_*`, `SCORE_THRESHOLD`, `RATED_THRESHOLD`, `HARD_PRICE_*`).

## Deeper docs

`README.md` (relay + hybrid proxy details), `IMPLEMENTATION_OUTLINE.md`, `docs/591research.md` (polling tiers §5, penalty engine §4), `docs/feedback-triage.md` (rating comments: deterministic-code vs dynamic-prompt routing; triage happens in-chat, not in `rate.py`), `docs/rent591-network-access.md` (591 WAF/CDN findings).
