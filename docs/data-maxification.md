# Data Maximization: Unified Schema Specification

Goal: **zero data points missed** when combining mcp-591 API metadata (primary) with
591scraper DOM fallback (secondary). Strategy: store the **full raw JSON** of every
payload plus a small set of denormalized canonical columns for fast querying.

## Side-by-side field comparison

| Semantic field | mcp-591 search item | mcp-591 rent detail | 591scraper DOM |
|---|---|---|---|
| Listing ID | `id` | (in URL) | `id` |
| Title | `title` | `title` | `title` |
| Price | `price`, `price_unit`, `price_per`, `price_per_unit`, `price_has_carport` | `price`, `priceUnit`, `containCost[]`, `cost[]` | `price`, `price_adjusted` |
| URL | `url` | — | `link` |
| Address | `address` | `address.data`, `address.value` | `addr` |
| lat/lng | — | `address.lat/lng`, `positionRound.lat/lng` | — |
| Region/Section | `regionid`/`sectionid` + `region_name`/`section_name` (via gtm) | `regionId`, `sectionId`, `gtm_detail_data.region_name/section_name` | — |
| Community | `community_name`, `community_id` | `positionRound.communityName/communityId` | `社區` |
| Layout | `layoutStr` | `info[layout]`, `headInfo` | (unavailable) |
| Area 坪 | `area` (float), `area_name` | `info[area]`, `houseInfo[buildArea]` | (unavailable) |
| Floor | `floor_name` | `info[floor]` | (unavailable) |
| Shape | (via `kind_name`/`ding_kind_*`) | `info[shape]` | (unavailable) |
| Kind | `kind`, `kind_name` | `kind`, `gtm_detail_data.kind_name` | — |
| Description | — | `remark.content` | `desc` |
| Facilities | — | `service.facility[]` (structured keys) | `提供設備` (text) |
| Pet | — | `houseInfo[pet]` | `養寵物` (Yes/No) |
| Cook | — | `houseInfo[cook]` | — |
| Lease/move-in | — | `houseInfo[leaseTime/comeDate]` | — |
| Deposit | — | `deposit`, `cost[deposit]` | — |
| Fees included | — | `containCost[]` | `租金含` |
| 管理費/車位費 | — | `cost[]` | `管理費`, `車位費` |
| Tags | `tags[]` | `tags[{id,value}]` | — |
| Images | `photoList[]`, `cover` | `meta.ogimage`, `favData.thumb` | **none** |
| Video | `video.video_url` (m3u8) | — | — |
| browse count | `browse_count` | — | — |
| refresh time | `refresh_time`, `refresh_tag_visible` | `publish.postTime` | `fetched` |
| Agent | `role_name` | `linkInfo.{name,roleName,mobile,phone}` | `poster` |
| Surrounding POIs | `surrounding{}` | `surround.data[]`, `positionRound.mapData[]` (with distance) | — |
| Social housing | `social_house` | — | `mark` (社宅) |
| Fee-adjusted price | — | — | `price_adjusted` (agent fee 1/24 + fees) |

## Rules guaranteeing zero loss
1. **Ingest through `Client591` directly** (not the MCP server's lossy `_filter_*`), storing the complete response JSON.
2. Keep **three raw captures** per listing:
   - `raw_search` → full search item (images, video, tags, surrounding, browse_count)
   - `raw_metadata` → full rent-detail `data{}` (address, cost, service, houseInfo, linkInfo, positionRound, gtm_detail_data)
   - `scraper_raw` → 591scraper DOM dict (desc, poster, 租金含/車位費/管理費, 提供設備, 養寵物, mark, price_adjusted)
3. `description` = `remark.content` (API) with fallback to scraper `desc`.
4. Images: primary `photoList` (strip `!resize` suffix → original), fallback `cover`/`meta.ogimage`/`favData.thumb`. **No images from scraper**.
5. Canonical denormalized columns (price, layout, area, floor, address, lat/lng, community, tags, etc.) derived API-first, scraper-fallback where the API lacks them.
6. All image derivatives (WebP paths, per-image DINOv2 vectors) live in the `listing_images` table keyed by `listing_id`.

## Unified schema

```sql
CREATE TABLE IF NOT EXISTS listings (
    listing_id        TEXT PRIMARY KEY,            -- post_id
    title             TEXT,
    price             INTEGER,                     -- NT$/month (int)
    price_unit        TEXT,
    url               TEXT,
    status            TEXT,
    is_active         BOOLEAN DEFAULT TRUE,
    region            TEXT, section TEXT,
    address           TEXT,
    lat REAL, lng REAL,
    community_name TEXT, community_id INTEGER,
    layout TEXT, area REAL, floor TEXT, shape TEXT, kind_name TEXT,
    deposit TEXT, rent_per REAL, rent_per_unit TEXT,
    browse_count INTEGER, refresh_time TEXT,
    tags JSON, contain_cost JSON,
    raw_search JSON, raw_metadata JSON, scraper_raw JSON,
    description TEXT,
    image_urls JSON, image_paths JSON,
    qwen_warnings JSON, qwen_vision_flags JSON, qwen_direct_score REAL,
    dino_embedding BLOB,
    predicted_score REAL, score_source TEXT,
    user_rated BOOLEAN DEFAULT FALSE,
    user_score REAL, bathroom_score REAL, user_comments TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS listing_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id TEXT NOT NULL REFERENCES listings(listing_id),
    ordinal INTEGER,
    image_url TEXT, image_path TEXT,
    dino_embedding BLOB,
    UNIQUE(listing_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_lid ON listing_images(listing_id);

CREATE TABLE IF NOT EXISTS dynamic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_bullet_list TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Image storage directive
- Download originals (resize suffix stripped).
- Re-encode to **WebP, quality=85** (Pillow) → `data/images/{listing_id}/{ordinal:02d}.webp`.
- Rationale: ~40-70% smaller than JPEG originals at equivalent fidelity; optimal for Qwen-VL tokenization and DINOv2 feature extraction.
- On download/encode failure: record empty `image_path`, do not abort the listing.

### Verified constraint (Ollama + WebP)
- **Ollama cannot ingest WebP** (400 "Failed to load image or audio file"). WebP stays on disk
  (disk savings + DINOv2 reads it fine via Pillow), but `src/vision_llm._image_b64` transcodes to
  **PNG in-memory** before base64-encoding for the `/api/chat` `images` array. No schema change needed
  (`image_path` remains `.webp`).
