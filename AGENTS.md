# AGENTS.md

Rent591 rental scout: 591.com.tw ingestion → DINOv3 dedup → Qwen vision (Ollama) → XGBoost score → ntfy alerts → SQLite. Python 3.12, uv-managed `.venv`.

## Two-machine architecture (never assume direct network)

This GPU host's egress firewall blocks `rent.591.com.tw` / `bff-house.591.com.tw` / `ntfy.sh`; only `github.com` is whitelisted. Flow:

1. GitHub Actions (`.github/workflows/scrape_relay.yml`) scrapes on GitHub runners, commits raw payloads + WebP to `data/incoming/` in this repo (`auto(ingest): ... [skip ci]` commits). Never hand-edit `data/incoming/` or push competing commits without `git pull --rebase`.
2. `python main.py --incoming` ingests those payloads locally; images/vision/ntfy go through the PC devtunnel proxy at `127.0.0.1:8999` (`PROXY_URL`). Proxy offline → listings stay `pending`, never crash.
3. Relay idempotency: per-listing `payload_sha256` + `relay_state` table. Both sides skip unchanged.

Hard filters (price 10000–17000, ≥6 ping, kind 2/3 套房) live in `src/ingestion.py` (`passes_hard_filters`) and are enforced on **both** relay side (`dump_relay_payloads`) and incoming side — change both or payloads mismatch.

## Commands

```bash
.venv/bin/python -m pytest -q                 # full suite (~13s, fully offline)
.venv/bin/python -m pytest tests/test_x.py -q # single file
.venv/bin/python main.py --fixtures --limit 3 # offline E2E (needs external/mcp-591/tests/fixtures)
.venv/bin/python main.py --incoming           # hybrid relay run
.venv/bin/python main.py --train             # retrain XGBoost head
.venv/bin/python rate.py --id N --score 4     # rate listing (auto-retrains past RATED_THRESHOLD=20)
.venv/bin/python -m ruff check .              # lint — clean at HEAD, keep it that way
gh workflow run scrape_relay.yml              # manual relay trigger
```

Dedup tests need real image content: DINOv3 CLS collapses on textureless synthetic noise (cos ~0.99 between unrelated noise images) — test fixtures must draw structured shapes.

## Repo gotchas

- `external/` and `models/` are gitignored but required locally: vendored `external/mcp-591` + `external/591scraper` (fixtures live in the former) and `models/dinov3_cache/` + `models/xgboost_head.json`.
- `requirements.txt` is **relay-runner-only** (requests + Pillow). Heavy deps (torch, transformers, xgboost, DrissionPage) are in `pyproject.toml` and must NOT be added to requirements.txt.
- DB schema migrations go in `src/database.py` `_EXTRA_COLUMNS` (idempotent `ALTER TABLE`); DB is `data/apartments.db` (WAL).
- All pipeline stages fail-soft by design — broad `except Exception` is intentional (ruff `BLE001` ignored, line-length 110). Don't "fix" it.
- ntfy headers must be ASCII/latin-1: score rendered as `(x.xx/5)` (2 decimals), never `★` (breaks latin-1 encoding).
- Layer 1 visual preference lives in `src/visual_preference.py`: XGBoost's input is the
  compressed fusion vector `scoring.FEATURE_NAMES` (dino_visual_score + qwen_score + flags +
  tabular), never the raw 768-d blob; changing the vector width invalidates `models/xgboost_head.json`
  and `models/dino_probe.npz` (both auto-retrain).
- All user-facing text is English: vision prompt demands English warnings; `notifier._summary`/`_en`
  translate the fixed Chinese vocabulary (districts/kind/heuristics/legacy rows). Keep new warning
  strings English or add them to `notifier._WARN_EN`; the consolidate-preferences prompt also forces English bullets.
- Proxy traffic uses `verify=False` because devtunnel MITMs TLS with a cert Python rejects.
- 591 CDN 403s on original photo URLs via the tunnel → fetch `!fit.1000x.water2.jpg` resize variants (`PROXY_IMAGE_SUFFIX`). CDN 502 storms = throttling (backoff, stay `pending`), not a dead proxy.
- Tests must stay offline: `HF_HUB_OFFLINE=1`, cached DINOv3 weights, fixture replay, mocked Ollama/ntfy/proxy. `conftest.py` inserts repo root into `sys.path`.
- Env config is read at import time in `main.py`/`src/` — see README "Setup" for the full var list (`X591_*`, `PROXY_*`, `NTFY_TOPIC`, `OLLAMA_*`).

## Deeper docs

`README.md` (hybrid proxy mode details), `IMPLEMENTATION_OUTLINE.md`, `docs/591research.md` (polling tiers §5), `docs/BUILD_PLAN.md`.
