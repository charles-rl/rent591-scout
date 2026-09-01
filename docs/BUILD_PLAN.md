# BUILD_PLAN — `591-mcp-vision`

Execution spec for the Build agent. All paths relative to repo root `/root/rent591-scout`.

## STATUS: COMPLETE (all 8 tasks implemented, tested, lint-clean)

### Execution deviations & findings (post-implementation)
1. **Vendored modules renamed** to valid Python identifiers: `client591.py`, `constants591.py` (Python can't `import .591_client` — module names may not start with a digit).
2. **CUDA torch pinned** to `2.8.0+cu128` — the default `2.13.0+cu130` won't load on the driver (CUDA 12.8). Install via the nexus pypi proxy; nvidia-* deps are auto-routed to the blocked `pypi.nvidia.com`, so use `pip --index-url nexus-pypi-proxy --extra-index-url nexus-pypi-nvidia-proxy`.
3. **scikit-learn added** — `xgboost.XGBRegressor` is a sklearn wrapper and raises `ImportError` without it.
4. **Ollama cannot decode WebP** (returns 400 "Failed to load image or audio file"). WebP stays on disk (q85); `vision_llm._image_b64` transcodes to PNG in-memory for the `/api/chat` images array.
5. **ntfy `Title` header bug fixed** — `★` (U+2605) is not latin-1 and requests refuses to set the header. Replaced with `(x.x/5)`. Verified payload via a local capture server.
6. **Sandbox network reality**: 591.com.tw and ntfy.sh are firewall-blocked here (IP-level DROP; only GitHub/PyPI/HF/PyTorch/DockerHub are whitelisted). The nexus proxy is package-registry-only (no general egress). All end-to-end testing uses `--fixtures` (captured 591 responses) + `PLACEHOLDER_IMAGES=1` + a local `NTFY_URL` capture server.
7. **Live runs honor `HTTPS_PROXY`** automatically (requests `trust_env=True`); pass a real forward proxy to run live from a restricted host.
8. **DINOv2 → DINOv3 upgrade:** dedup feature extractor moved from `facebook/dinov2-base` to Meta DINOv3 ViT-B/16 (`dinov3-vit-base`). The official HF checkpoint is gated, so the exact-weights conversion of Meta's released `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (verified: 187/187 tensors mapped, RoPE θ=100 frequency grid matches) is staged under `models/dinov3_cache/facebook_dinov3-vit-base/`. Output contract unchanged: 768-dim float32 CLS embeddings; group-cosine threshold stays 0.95. Verified offline on sample WebP images: 32/32 embedded, all vectors 768-dim float32 L2-normalized; self re-ingestion group sim = 1.0000 (dup detected at 0.95), cross-listing sim in [0, 1); CUDA memory stable across repeated passes.

## Environment (provisioned)
- Python 3.12 venv: `.venv/` (uv). Installed: `requests, Pillow, numpy, xgboost, DrissionPage, torch==2.8.0+cu128, torchvision==0.23.0, transformers`.
- GPU: NVIDIA H100 80GB (CUDA 12.8). `torch.cuda.is_available() == True`.
- Ollama: model `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL` (vision-capable, non-uncensored) at `http://localhost:11434`.
- 591.com.tw is firewalled in this sandbox → implement `--fixtures` offline mode using `external/mcp-591/tests/fixtures/*.json` for end-to-end testing.
- Vendored deps: copy `external/mcp-591/mcp_591/client.py` → `src/client591.py`, `constants.py` → `src/constants591.py`.
- `docs/`: `mcp-591-analysis.md`, `591scraper-analysis.md`, `data-maxification.md` (schema reference).

## Config (env vars, defaults in code)
- `X591_REGION` (default `台北市`), `X591_SECTION` (default `` = all), `X591_KIND` (default `整層住家`), `X591_PRICE_STR` (default `15000_25000`), `X591_FIRST_PAGES` (default 1)
- `NTFY_TOPIC` (default `rent591-scout`)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_MODEL` (default `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL`)
- `SCORE_THRESHOLD` (default 3.5), `RATED_THRESHOLD` (default 20), `DEDUP_THRESHOLD` (default 0.95)
- `DB_PATH` (default `data/apartments.db`), `IMAGES_DIR` (default `data/images`)
- `FIXTURES_DIR` (default `external/mcp-591/tests/fixtures`) — offline mode source
- `CHROME_BINARY` (optional path override for DrissionPage)
- `DINOV3_CACHE` (default `models/dinov3_cache`), `DINOV3_MODEL_PATH` (optional local checkpoint dir; default staged `models/dinov3_cache/facebook_dinov3-vit-base`)

---

## Task 1 — `src/database.py`
- `connect(db_path=None) -> sqlite3.Connection`: WAL mode, `PRAGMA foreign_keys=ON`, run DDL from `data-maxification.md`.
- `upsert_listing(conn, listing: dict) -> None` (INSERT … ON CONFLICT(listing_id) DO UPDATE).
- `replace_images(conn, listing_id, images: list[dict]) -> None` (clear + insert `listing_images`).
- `get_all_images(conn) -> list[dict]` (for dedup baseline).
- `get_rating_count(conn) -> int`, `get_rated_samples(conn) -> list[tuple]` (emb blob, flags json, target).
- `get_latest_preferences(conn) -> str | None`.
- `save_preferences(conn, bullets: str) -> None`.
- Keep functions small; every write commits.

## Task 2 — `src/ingestion.py`
- Copy vendored `client591.py`/`constants591.py` into `src/`.
- `fetch_raw_listings(fixtures: bool = False) -> list[dict]`:
  - **Live**: `Client591().search_rent(region_id, section_ids, kind, price_str)` → paginate via `data.firstRow` up to `X591_FIRST_PAGES`; for each item call `get_rent_detail(id)`.
  - **Fixtures**: read `search_rent.json` + `rent_detail.json` from `FIXTURES_DIR`, replay `data.items[]` and `data` with same key mapping.
  - Return unified list of `{raw_search: item, raw_metadata: data}` (empty `raw_metadata` for delisted).
- `normalize_listing(item, data) -> dict`: derive canonical fields per `data-maxification.md` rules; `url = https://rent.591.com.tw/{id}`; price parsed from `price` (strip commas).
- `fetch_image_urls(item) -> list[str]`: `photoList` (strip `!…` suffix), fallback `cover`, `meta.ogimage`, `favData.thumb`.
- `download_images(listing_id, urls, images_dir) -> list[dict]`: requests GET (timeout 30, verify=False), Pillow open→RGB→`save(…, 'WEBP', quality=85)` to `images_dir/{listing_id}/{ordinal:02d}.webp`; return `[{ordinal, image_url, image_path}]`; tolerate per-image failure.
- `scraper_fallback(listing_id) -> dict | None`: DrissionPage detail scrape (selectors from `docs/591scraper-analysis.md`), guarded import; return DOM dict or `None` on any failure. (Optional in fixtures mode → `None`.)

## Task 3 — `src/deduplication.py` (DINOv3 upgrade)
- Model: **Meta DINOv3 ViT-B/16** (`dinov3-vit-base`; HF repo `facebook/dinov3-vitb16-pretrain-lvd1689m`, checkpoint hash `73cec8be`). Loaded with `transformers` `AutoModel` + `AutoImageProcessor`; moved to cuda; CLS token → L2-normalized.
- **Setup & dependencies:** no new packages beyond the existing `torch==2.8.0+cu128`, `transformers` (needs `dinov3_vit` model support, present in installed build), `Pillow`, `numpy`. The official HF repo is gated → stage converted weights locally at `models/dinov3_cache/facebook_dinov3-vit-base/` (`config.json` + `model.safetensors` + `preprocessor_config.json`) so the GPU server runs **offline** after the initial pull.
- **Caching directory:** `models/dinov3_cache/`. Offline-first resolution: use staged local dir if present, else pull from HF into the same cache (`cache_dir=models/dinov3_cache`). Env overrides: `DINOV3_CACHE`, `DINOV3_MODEL_PATH` (dir path); `HF_HUB_OFFLINE=1` forces offline.
- `embed_image(path) -> np.ndarray | None`: open webp → processor resize 224 → embed; returns **768-dim float32** L2-normalized `[CLS]`; dimension guard rejects non-768 outputs; failures logged and return `None`.
- `embed_image_rows(rows: list[dict]) -> dict[ordinal, vec]`: missing files skipped.
- `cosine(a, b)`, `group_similarity(new_vecs, stored_vecs) -> float` (mean over new of max-over-stored).
- `find_duplicate(new_vecs, baseline: dict[listing_id, list[vec]], threshold) -> (bool, listing_id)`.
- `aggregate_embedding(vecs) -> bytes` (mean → float32 BLOB).

## Task 4 — `src/vision_llm.py`
- `BASE_SYSTEM_PROMPT` + `build_messages(listing, images, dynamic_bullets)`: user content = structured JSON schema + Chinese text (`title`, `description`, `layout`, `area`, `floor`, `tags`, `contain_cost`) + images as base64 (`images: [b64...]`).
- `ask_ollama(messages) -> str`: POST `{OLLAMA_BASE_URL}/api/chat`, `stream: false`, `model: OLLAMA_MODEL`.
- `parse_json(text) -> dict`: strip markdown fences, regex-extract `{...}` balanced, `json.loads`; fallback defaults.
- `analyze_listing(listing, image_rows, dynamic_bullets) -> dict`: returns `{qwen_warnings, vision_flags, qwen_direct_score}`; never raises (returns None on failure).

## Task 5 — `src/scoring.py`
- `predict_score(conn, dino_vec, flags, qwen_direct_score) -> (float, str)`:
  - `rated_count <= RATED_THRESHOLD` → `(qwen_direct_score, 'qwen')`.
  - else train `XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)` on `[dino_embedding; flag values] → user_score`; predict; return `(clip(pred,1,5), 'xgboost')`.
- `train_model(conn) -> xgb.XGBRegressor` (saved to `models/xgboost_head.json` for `--train` mode).
- Flag vector = fixed order of `vision_flags` values + `qwen_warnings` binary presence.

## Task 6 — `src/notifier.py`
- `send_ntfy_alert(listing, threshold)` per outline Component 5; `POST https://ntfy.sh/{NTFY_TOPIC}`; headers Title `Apartment Match (x.x/5)`, Click=url, Tags `house,bathroom`; body = `NT$price - title\n⚠️ warnings`. Non-fatal on failure.

## Task 7 — `rate.py` + `src/dynamic_prompt.py`
- `rate.py`: argparse `--id --score --bathroom --comment` → `UPDATE listings SET user_rated=TRUE, user_score=?, bathroom_score=?, user_comments=? WHERE listing_id=?`; if comment → call `dynamic_prompt.consolidate`.
- `dynamic_prompt.consolidate(conn, new_feedback) -> str`: Qwen text call (Ollama `/api/chat`, no images) merging current bullets + feedback into ≤7 bullets; `save_preferences`.

## Task 8 — `main.py`
- `argparse`: `--fixtures`, `--limit N`, `--train`, `--notify` (default True).
- Loop (per outline §6):
  1. `ingestion.fetch_raw_listings(fixtures=...)` → `normalize` → download images.
  2. Dead-link filter: `raw_metadata` empty → `is_active=FALSE`, skip.
  3. Dedup: embed images → compare vs DB baseline → skip if dup.
  4. `vision_llm.analyze_listing` (skip if no images → warnings from text only).
  5. `scoring.predict_score`.
  6. `notifier.send_ntfy_alert` if `predicted_score >= threshold`.
  7. `database.upsert_listing` + `replace_images` (store `dino_embedding`, `score_source`).
- `--train`: train + persist xgboost model, then exit.
- Graceful: each listing wrapped so one failure never aborts the run.

## Acceptance
- `python main.py --fixtures --limit 3 --notify=false` completes without exception; rows + images in `data/apartments.db`.
- `python rate.py --id <fixture_id> --score 4 --comment "..."` updates DB + writes new dynamic_preferences row.
- `python main.py --fixtures --train` trains model from rated rows.
- Lint: `python -m pyflakes src/ main.py rate.py` clean (pyflakes installed via nexus proxy).
