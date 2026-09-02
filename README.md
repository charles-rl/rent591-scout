# 591-mcp-vision

Automated, Active Learning-driven rental monitor for Rent591 (`rent.591.com.tw`).
Combines API ingestion (mcp-591), browser fallback scraping (591scraper), local
DINOv3 image embeddings for deduplication, a local Qwen 27B vision LLM (Ollama),
an XGBoost scoring head, and ntfy.sh push notifications.

## Architecture

```
Ingestion (mcp-591 API → 591scraper DOM fallback)
   → WebP images (quality=85)
    → DINOv3 group-cosine dedup (≥0.95, 768-dim ViT-B/16 CLS embeddings from models/dinov3_cache, offline-capable)
   → Qwen 27B vision + text JSON analysis (Base + Dynamic prompt)
   → Score (Qwen direct ≤20 ratings | XGBoost >20 ratings)
   → ntfy.sh push if score ≥ 3.5 (via PC devtunnel when online, see hybrid mode)
   → SQLite storage + CLI feedback → dynamic preference prompt retrain
```

### GitHub Actions cron relay (firewall workaround)

The GPU server's egress firewall blocks `rent.591.com.tw` / `bff-house.591.com.tw` /
`ntfy.sh` but whitelists `github.com`. Scraping is therefore offloaded to GitHub's
cloud runners (`.github/workflows/scrape_relay.yml`, every 30 min + manual dispatch):

```
GitHub runner (can reach 591)                    GPU server (591 blocked)
  python -m src.ingestion --output-dir data/incoming/
    raw JSON payloads + WebP images
  git commit + push  ────────────────────────────▶  git pull
                                                    python main.py --incoming
                                                      DINOv3 dedup (local)
                                                      Qwen vision (Ollama)
                                                      XGBoost score + SQLite
```

Payloads land in `data/incoming/listings/<id>.json` + `data/incoming/images/<id>/*.webp`
(shipped in-git; `.gitignore` un-ignores `data/incoming/`). Per-listing `payload_sha256`
plus a `relay_state` table make both sides idempotent: the runner skips unchanged
payloads (no commit churn), the server skips already-processed hashes. Text payloads
usually arrive **without** images (`images: []`) — photos are fetched by the GPU
server itself through the PC proxy bridge described next.

### Hybrid PC-proxy mode (devtunnel)

The personal PC runs a local HTTP proxy (port `8999`) bridged to the GPU server as
`127.0.0.1:8999` via `devtunnel host`. Each `python main.py --incoming` run:

1. **Text ingestion (always, zero egress):** payloads stored in SQLite with
   `image_status='pending'` + rule-based text warnings (floor / utilities / pricing).
2. **Proxy probe** (`src/utils/proxy_check.py`): GET `www.591.com.tw` through the
   tunnel — HTTP 200 means the PC is online.
3. **LIVE** → `src/utils/image_queue.py` drains all pending listings through the
   proxy (WebP q85 → `data/images/`), then each completed listing runs DINOv3
   dedup → Qwen vision → XGBoost; match alerts (score ≥ `SCORE_THRESHOLD`) push to
   ntfy **via the tunnel**.
4. **OFFLINE** → listings stay queued and one ntfy "connect PC proxy" alert is sent
   per batch (`text_only_notified` flag prevents spam; re-armed on payload change).

Details & gotchas:
- 591's CDN serves **403** for stripped original photo URLs through the tunnel; the
  queue requests resize variants instead (`PROXY_IMAGE_SUFFIX`, default
  `!fit.1000x.water2.jpg` ≈ 1000px, watermarked, served as WebP — ample for DINOv3/Qwen).
- devtunnel MITMs TLS with its own cert chain, which Python's OpenSSL rejects
  (`Missing Subject Key Identifier`) → proxy traffic uses `verify=False`
  (`PROXY_SSL_VERIFY=1` to re-enable once the tunnel CA is installed).
- CDN 502 storms are treated as **throttling**, not a dead PC: the listing stays
  `pending`, the drainer backs off (`PROXY_RATE_LIMIT_BACKOFF`) and resumes the
  next run; only true connection failures stop the drain.
- ntfy delivery is **tunnel-first with direct fallback** for every alert (match +
  proxy-request). When the PC is fully powered off there is no network path at
  that moment — the alert is best-effort by design; the queue and DB state persist.
- Listings stuck `completed` but unscored (Ollama was down) are retried every run.
- `listings.image_status` lifecycle: `pending → completed | failed | skipped`
  (skipped = inactive or no photos). The migration requeues legacy rows whose photo
  files are missing or solid-color placeholders from old `PLACEHOLDER_IMAGES=1` runs.

## Setup

```bash
uv venv --python 3.12 .venv
# Install deps (see docs/BUILD_PLAN.md; CUDA wheels via nexus proxy on this host)
uv pip install --python .venv/bin/python requests Pillow numpy xgboost scikit-learn DrissionPage transformers
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Config via env vars (defaults shown): `X591_REGION=台北市`, `X591_SECTION=`,
`X591_KIND=整層住家`, `X591_PRICE_STR=15000_25000`, `NTFY_TOPIC=rent591-scout`,
`OLLAMA_BASE_URL=http://localhost:11434`,
`OLLAMA_MODEL=hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`, `SCORE_THRESHOLD=3.5`.

Hybrid proxy mode (see above): `PROXY_URL=http://127.0.0.1:8999`,
`PROXY_PROBE_URL=https://www.591.com.tw/`, `PROXY_SSL_VERIFY=0`,
`PROXY_IMAGE_SUFFIX=!fit.1000x.water2.jpg`, `PROXY_DOWNLOAD_TIMEOUT=30`,
`PROXY_RATE_LIMIT_BACKOFF=20`, `PROXY_RATE_LIMIT_MAX_STREAK=3`,
`PLACEHOLDER_MAX_BYTES=4096`.

## Usage

```bash
python main.py                    # live run
python main.py --incoming         # hybrid run over relay payloads: text always; images +
                                  # vision + alerts via PC devtunnel proxy when online
python main.py --fixtures --limit 3   # offline test with captured fixtures
python main.py --train            # train XGBoost head from rated rows
python rate.py --id 21103645 --score 4 --bathroom 4 --comment "dry-wet separation preferred"

# manual relay trigger (needs gh / repo token):
gh workflow run scrape_relay.yml
```

## Notes

- Requires Chrome/Chromium for the 591scraper DOM fallback; it degrades gracefully.
- Images stored as WebP q85 (`data/images/{id}/{n}.webp`). Ollama cannot ingest WebP
  directly, so images are transcoded to PNG in-memory for the vision call.
- The 591 API and website are firewalled on this dev host; use `--fixtures` for
  end-to-end testing against captured responses in `external/mcp-591/tests/fixtures`.
- Live runs honor `HTTPS_PROXY`/`HTTP_PROXY` automatically (requests `trust_env=True`).
  On a restricted host, export a forward proxy that can reach 591 before running.
- All ntfy pushes are tunnel-first with direct fallback (`src/notifier.py`): the GPU
  server cannot reach `ntfy.sh` directly, but the PC devtunnel path returns 200 while
  the PC is online. The offline "connect the proxy" alert can only be delivered when
  the tunnel is partially alive — a fully-off PC means no egress path, by design.
- DrissionPage is non-commercial licensed. See `docs/` for full analysis.
- **DINOv3 dedup:** feature extraction uses Meta DINOv3 ViT-B/16 (`dinov3-vit-base`). Weights are cached under
  `models/dinov3_cache/` (offline after initial pull; set `HF_HUB_OFFLINE=1` to force). The extractor emits
  768-dim float32 L2-normalized CLS embeddings, keeping the XGBoost concat and SQLite BLOB schema unchanged.
- Verified: full pipeline passes end-to-end in fixtures mode on GPU (DINOv3 + Qwen
  vision + cold-start/XGBoost scoring); ntfy payload validated via local capture
  (Title header is ASCII `(x.x/5)` — a `★` in the header breaks latin-1 encoding).
