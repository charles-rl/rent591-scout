# Code Review — Sessions 1 & 2 (Foundation, Schema, Ingestion, WebP)

**Date:** 2026-08-31 (audit) / 2026-09-01 (fixes applied)
**Method:** Parallel `@explore` subagent audit (Agent A: schema/storage, Agent B: ingestion/scraper, Agent C: images) + independent cross-module verification by the main agent, including a live offline-fixture run and a live `apartments.db` schema dump.

**Status legend:** ✅ FIXED (verified) · ⚠️ DEFERRED (open, documented)

---

## Section 1 — Verified Features (production-ready)

### Schema & storage (`src/database.py`) — **MATCH**
- Live `sqlite3` dump of `data/apartments.db` confirms `listings`, `listing_images`, `dynamic_preferences` match `IMPLEMENTATION_OUTLINE.md` §3.
- `PRAGMA foreign_keys=ON`, `journal_mode=WAL`, `sqlite3.Row` on every connection (`database.py:63-66`). Single contract index `idx_listing_images_lid` present.
- JSON (`ensure_ascii=False`) and BLOB↔float32 DINO round-trips symmetric; all statements bound-parameterized.
- ✅ New canonical columns `social_house BOOLEAN` + `facilities JSON` added via idempotent migration `_ensure_columns` (verified live: cols 44-45).

### External isolation
- `external/mcp-591/` and `external/591scraper/` present; no nested `.git` trees. `.gitignore` covers `external/`, `data/`, `models/`, `.venv/`, `__pycache__/`, `browser_profile/`, `.uv/`.

### Image pipeline (`src/ingestion.py:download_images`)
- WebP + `quality=85` at the single save site; `timeout=30` + `raise_for_status`; ✅ content-type check + 20 MB payload cap + `img.close()`; ✅ cache integrity check (`_valid_webp`) re-downloads truncated files; ✅ failed downloads record `image_path=None` rows to keep URL/ordinal alignment.
- Placeholder mode verified: stored `.webp` are genuine WebP, 1-color 800×600 (fixture mode; 591 CDN firewalled in sandbox). Browser session closed in `finally`.

### General hygiene
- ✅ Root `pyproject.toml` added (deps pinned). ✅ `logging` module replaces ad-hoc prints across `main.py`, `src/ingestion.py`, `deduplication.py`, `vision_llm.py`, `scoring.py`, `notifier.py`. No hardcoded credentials; consistent synchronous code.

### Visual dedup layer — DINOv2 → DINOv3 upgrade (`src/deduplication.py`)
- Feature extractor moved from `facebook/dinov2-base` to **Meta DINOv3 ViT-B/16** (`dinov3-vit-base`, checkpoint hash `73cec8be`). The official HF repo is gated (manual access), so the exact-weights conversion of Meta's released `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` is staged locally at `models/dinov3_cache/facebook_dinov3-vit-base/` (`config.json` + `model.safetensors` + `preprocessor_config.json`) → GPU server runs **offline** after the initial pull (verified with `HF_HUB_OFFLINE=1`).
- Weight integrity verified before staging: all 187 parameter tensors mapped from the official checkpoint (zero random-init leakage); fused QKV split into q/k/v; K-bias verified zero in released weights; RoPE grid check — transformers' `θ=100` inv-freq sequence exactly equals Meta's stored `rope_embed.periods` (`head_dim=64`, 12 heads).
- Contract unchanged downstream: CLS output is **768-dim float32**, L2-normalized (dimension guard added in `embed_image`); XGBoost concat (768 + 6 flags = 774) and SQLite BLOB schema unaffected; group-cosine threshold stays `0.95`.
- Offline verification on sample WebP fixtures (`data/images/`, 2 listings, 32 images): 32/32 embedded; all vectors `(768,)` float32 finite with ‖v‖=1.000000; self re-ingestion group similarity = **1.0000** (duplicate detected at ≥0.95); cross-listing group similarity = **0.9341** (valid float, `< 0.95` → not a duplicate); CUDA memory stable over 4 repeated passes (0.376 GB before/after, peak 0.384 GB — no leak growth).

---

## Section 2 — Bugs & Code Smells (status)

| # | Severity | File:Line | Issue | Status |
|---|---|---|---|---|
| B1 | HIGH | `database.py:88-96` | Partial-dict upsert NULL-wiped stored data on degraded runs | ✅ FIXED via `COALESCE(excluded.c, listings.c)` in the conflict SET — live-verified (lat/lng/tags/facilities/qwen_score preserved on a None-filled degraded update; genuine delist still writes `is_active=0`). |
| B2 | HIGH | `main.py` | Self-duplicate: own embeddings in baseline caused `21103645 duplicate of 21103645` | ✅ FIXED — `own_baseline` excludes current `listing_id`; live run now scores 21103645 + 21074190. |
| B3 | HIGH | `client591.py:98/143/153/166` | No `timeout=` / retry on API calls | ✅ FIXED — `REQUEST_TIMEOUT` (30s) on all 4 calls + `Retry` adapter (429/5xx, backoff 1.0). |
| B4 | HIGH | `main.py` | No top-level exception handling; conn leak; exit always 0 | ✅ FIXED — try/finally closes conn; per-listing try/except continues; fatal → exit 1; any failure → non-zero exit. |
| B5 | HIGH | `ingestion.py:99-100, 178-179` | Silent swallow; false delisting on rate-limit/timeout | ✅ FIXED — logged warning; `detail_failed` flag; `is_active=True` on unknown failure (no delist), `status='closed'` only on genuine empty-detail. |
| B6 | HIGH | `ingestion.py:107-120` | Fixture mode broke for `--limit>1` (false delist) | ✅ FIXED — non-first items marked `detail_failed` → processed, not delisted. Verified: 3 listings fetched, 2 scored. |
| B7 | MED | `ingestion.py:193` | `rent_per` truncated (`"1,125.9"`→1125) | ✅ FIXED — `_float_price`; verified 1125.9 stored. |
| B8 | MED | `ingestion.py:250-251` | Failed images dropped rows → URL/ordinal misalignment | ✅ FIXED — failed rows appended with `image_path=None`. |
| B9 | MED | `ingestion.py:239` | Stale/corrupt `.webp` never re-downloaded | ✅ FIXED — `_valid_webp` integrity gate. |
| B10 | MED | `ingestion.py:168-170` | Detail tags masked by search tags | ✅ FIXED — union + dedup + strip (`dict.fromkeys`). |
| B11 | MED | `ingestion.py` | No DOM discovery fallback (`collect_list.py` not wired) | ⚠️ DEFERRED — per-listing scraper fallback only; outline-stated search-tier fallback unimplemented. |
| B12 | MED | `client591.py:35`, `ingestion.py:245` | `verify=False` on live TLS | ✅ FIXED — `RENT591_SSL_VERIFY` env gate, default `verify=True`; disabled only when explicitly set. |
| B13 | MED | `ingestion.py:245-247` | No content-type / size cap before decode | ✅ FIXED — `content-type` must contain `image`; `MAX_IMAGE_BYTES` (20 MB) cap. |
| B14 | MED | `ingestion.py:92` | Unthrottled detail loop (anti-bot risk) | ✅ FIXED — jittered `sleep(random.uniform(0.5, 2.0))` between detail calls (live mode). |
| B15 | LOW | `ingestion.py:147` | `.lstrip("R")` strips char class | ✅ FIXED — `removeprefix("R")` + explicit id handling. |
| B16 | LOW | `database.py:100-108` | `replace_images` churns AUTOINCREMENT ids | ⚠️ DEFERRED — cosmetic; delete+reinsert keeps `UNIQUE(listing_id, ordinal)`; acceptable at scale. |
| B17 | LOW | `deduplication.py:23` | CWD-relative models path | ✅ FIXED — ROOT-anchored `ROOT / models / dinov3_cache` (was `dinov2_cache`; moved with the DINOv3 upgrade). |
| B18 | LOW | `ingestion.py:154-155` | Dead `data.get("lat")` branch | ✅ FIXED — fallback to `positionRound.lat/lng`. |
| B19 | LOW | `ingestion.py:294-296` | Pet "Yes" heuristic (absence of ban) | ⚠️ DEFERRED — upstream behavior kept; surfaced into `facilities`. |
| B20 | LOW | repo root | No root `pyproject.toml` | ✅ FIXED — added with pinned deps. |
| B21 | LOW | `src/` | No `logging`, ad-hoc prints | ✅ FIXED — `logging` everywhere; only vendored CLI block prints remain (`client591.py __main__`). |
| B22 | LOW | `ingestion.py:241,247`; `main.py:88` | Inline `import`/`__import__` | ✅ FIXED — moved to module top. |
| B23 | LOW | `ingestion.py:206-215` | Duplicate thumbnails (dedup before strip) | ✅ FIXED — dedup after `_strip_suffix`. |
| B24 | LOW | `ingestion.py:87` | Pagination guard silently stops on string `firstRow` | ✅ FIXED — `int()` coercion + warning. |

---

## Section 3 — Data Maximization Audit (status)

| Field | Source | Outcome | Status |
|---|---|---|---|
| `social_house` (社宅) | item + `favData.socialHouse` | New `social_house BOOLEAN` column, populated | ✅ FIXED |
| Facilities (冰箱/洗衣機/冷氣/…) | detail `service.facility[]` (active) | New `facilities JSON` column, populated (verified on 21103645) | ✅ FIXED |
| Pet policy + 提供設備 + 租金含 | 591scraper DOM | Merged into `facilities` / `contain_cost` in `apply_scraper`; `scraper_raw` retained | ✅ FIXED |
| Detail tags union (可養寵物/租金補貼/可入籍) | mcp-591 detail | Union + dedup + strip into `tags` | ✅ FIXED |
| `rent_per` precision (`"1,125.9"`) | search item | Full float stored in REAL column | ✅ FIXED |
| Qwen prompt context | all | Now includes `facilities`, `social_house`, `deposit`, `rent_per` (`vision_llm.py:build_messages`) | ✅ FIXED |
| `video` (m3u8), agent PII (`linkInfo` phone), surrounding POIs, precise `buildArea`, orientation, publish epoch | mcp-591 | Raw-only in `raw_search`/`raw_metadata` | ⚠️ DEFERRED — retained verbatim; no canonical columns |
| `price_adjusted` (agent fee + 管理費 + carport) | 591scraper | Not reproduced (no canonical price-adjusted field) | ⚠️ DEFERRED |

---

## Section 4 — Remaining Action Items (for Session 3)

### P1
1. **B11 — DOM discovery fallback:** wire `external/591scraper/collect_list.py` as a search-tier fallback when the API is rate-limited, per IMPLEMENTATION_OUTLINE "Secondary: 591scraper".
2. **Verification matrix for live mode:** a real network run cannot be validated in this sandbox (591.com.tw firewalled). Before live deployment: run with `RENT591_SSL_VERIFY=0` only on test networks; confirm retry/backoff behavior against 429s.

### P2
3. **B16 — `replace_images`:** consider UPSERT-only updates to stop AUTOINCREMENT churn.
4. **Canonical columns for:** `video`, orientation, precise `buildArea`, publish epoch (`publish.updateTime`), and `price_adjusted` — if downstream analytics need them outside raw JSON.
5. **B19 — pet heuristic:** add explicit "可養寵物" DOM evidence check before labeling.

### Follow-up verification (next session)
- Live run smoke test: pagination + detail loop throttling + scraper fallback with `CHROME_BINARY` set.
- Re-run `--fixtures --limit 3` after any DDL change to confirm the `_ensure_columns` migration is idempotent on an existing DB.
- **DINOv3 dedup verification (completed this session):** offline fixture run confirms 768-dim float32 embeddings from `models/dinov3_cache`, group-cosine values in `[0, 1]` with self-match = 1.0000 ≥ 0.95 threshold and cross-listing separation at 0.9341; no CUDA memory growth across repeated embedding passes. Remaining: compare DINOv2 vs DINOv3 duplicate-detection recall on the historical fixture set if the old `dinov2_cache` is still available for A/B (old embeddings in `apartments.db` are DINOv2-space and should be re-embedded once, or tolerate mixed spaces during transition since cross-model cosine is not comparable).
