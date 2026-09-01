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
   → ntfy.sh push if score ≥ 3.5
   → SQLite storage + CLI feedback → dynamic preference prompt retrain
```

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

## Usage

```bash
python main.py                    # live run
python main.py --fixtures --limit 3   # offline test with captured fixtures
python main.py --train            # train XGBoost head from rated rows
python rate.py --id 21103645 --score 4 --bathroom 4 --comment "dry-wet separation preferred"
```

## Notes

- Requires Chrome/Chromium for the 591scraper DOM fallback; it degrades gracefully.
- Images stored as WebP q85 (`data/images/{id}/{n}.webp`). Ollama cannot ingest WebP
  directly, so images are transcoded to PNG in-memory for the vision call.
- The 591 API and website are firewalled on this dev host; use `--fixtures` for
  end-to-end testing against captured responses in `external/mcp-591/tests/fixtures`.
- Live runs honor `HTTPS_PROXY`/`HTTP_PROXY` automatically (requests `trust_env=True`).
  On a restricted host, export a forward proxy that can reach 591 before running.
- DrissionPage is non-commercial licensed. See `docs/` for full analysis.
- **DINOv3 dedup:** feature extraction uses Meta DINOv3 ViT-B/16 (`dinov3-vit-base`). Weights are cached under
  `models/dinov3_cache/` (offline after initial pull; set `HF_HUB_OFFLINE=1` to force). The extractor emits
  768-dim float32 L2-normalized CLS embeddings, keeping the XGBoost concat and SQLite BLOB schema unchanged.
- Verified: full pipeline passes end-to-end in fixtures mode on GPU (DINOv3 + Qwen
  vision + cold-start/XGBoost scoring); ntfy payload validated via local capture
  (Title header is ASCII `(x.x/5)` — a `★` in the header breaks latin-1 encoding).
