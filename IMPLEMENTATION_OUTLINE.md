# System Architecture & Implementation Spec: `591-mcp-vision`

`591-mcp-vision` is an automated, Active Learning-driven rental monitor designed for Rent591 (`rent.591.com.tw`). It combines API querying, headless browser scraping, local Vision LLM analysis, multi-image semantic deduplication, and tabular machine learning to automatically evaluate rental properties against granular personal preferences (specifically focusing on bathroom conditions, kitchen amenities, hidden costs, and building rules).

---

## 1. Primary Repositories & Dependencies

This system integrates two open-source scraping components:

1. **`asgard-ai-platform/mcp-591`**
* **URL:** `[https://github.com/asgard-ai-platform/mcp-591](https://github.com/asgard-ai-platform/mcp-591)`
* **Role:** High-speed API-based ingestion. Extracts structured JSON metadata (rent price, room layout, floor, pet/cooking policies, address coordinates, and direct image links).


2. **`ceshine/591scraper`**
* **URL:** `[https://github.com/ceshine/591scraper](https://github.com/ceshine/591scraper)`
* **Role:** Browser-driven fallback scraper using `DrissionPage` (CDP protocol) with persistent profiles (`./browser_profile`). Captures full DOM descriptions, detail attributes, and images when API endpoints are rate-limited or return incomplete payloads.



---

## 2. End-to-End System Architecture

```
                       ┌───────────────────────────────┐
                       │   Scheduled Cron Ingestion    │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 1. Ingestion Layer                │
                     │    - mcp-591 API Fetch           │
                     │    - ceshine/591scraper Fallback  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                      │ 2. Deduplication & Active Check   │
                      │    - HTTP 200 / Active status     │
                      │    - DINOv3 Cosine Similarity     │
                      │      (Group Avg Match >= 0.95)   │
                     └─────────────────┬─────────────────┘
                                       │ (If unique & active)
                                       ▼
                     ┌───────────────────────────────────┐
                     │ 3. Qwen 27B Vision & Text Engine  │
                     │    - Base System Prompt (JSON)    │
                     │    - Dynamic Preference Prompt    │
                     │    - Generates Flags & Red Flags  │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                      ┌────────────────────────────────────┐
                      │ 4. Scoring Engine (3-Layer Fusion) │
                      │    - L1: DINO 768-d -> scalar      │
                      │      dino_visual_score             │
                      │      (liked centroid <=20 rated,   │
                      │       Ridge linear probe >20)      │
                      │    - L2: qwen_score (0.0-1.0)      │
                      │    - First 20 Samples: Qwen Score  │
                      │    - Post-20: XGBoost on fusion    │
                      │      (scalars + flags + tabular)   │
                      └─────────────────┬─────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
                     ▼                                   ▼
          [ High Likeliness ]                     [ Low Likeliness ]
                     │                                   │
                     ▼                                   ▼
       ┌──────────────────────────┐             ┌──────────────────┐
       │ 5. ntfy.sh Notification  │             │ Quietly Store in │
       │    - Direct 591 Link     │             │ SQLite DB        │
       │    - Key Warning Flags   │             │ (Unsupervised)   │
       └────────────┬─────────────┘             └──────────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ 6. User Feedback         │
       │    - CLI / JSON File     │
       │    - Partner & User      │
       │      Ratings + Comments  │
       └────────────┬─────────────┘
                    │
                    ▼
       ┌──────────────────────────┐
       │ 7. Retrain & Evolve      │
       │    - Retrain XGBoost Head│
       │    - Synthesize Dynamic  │
       │      Preference Prompt   │
       └──────────────────────────┘

```

---

## 3. SQLite Database Schema (`apartments.db`)

Create a single local SQLite database containing the `listings`, `listing_images`, and `dynamic_preferences` tables. **Revision (from analysis):** store full raw payloads (`raw_search`, `raw_metadata`, `scraper_raw`) to guarantee zero data loss, plus per-image rows (`listing_images`) for true multi-image group cosine deduplication. Full field rationale: `docs/data-maxification.md`.

```sql
CREATE TABLE IF NOT EXISTS listings (
    listing_id TEXT PRIMARY KEY,            -- post_id
    title TEXT,
    price INTEGER,                          -- NT$/month
    price_unit TEXT,
    url TEXT,
    status TEXT,                            -- open / closed
    is_active BOOLEAN DEFAULT TRUE,
    is_duplicate BOOLEAN DEFAULT FALSE,     -- DINOv3 group-cosine dup of a stored listing

    -- Denormalized canonical fields (API-first, scraper fallback)
    region TEXT, section TEXT, address TEXT,
    lat REAL, lng REAL,
    community_name TEXT, community_id INTEGER,
    layout TEXT, area REAL, floor TEXT, shape TEXT, kind_name TEXT,
    deposit TEXT, rent_per REAL, rent_per_unit TEXT,
    browse_count INTEGER, refresh_time TEXT,
    tags JSON, contain_cost JSON,
    social_house BOOLEAN,                   -- 社宅 (social housing) flag
    facilities JSON,                        -- fridge/washer/AC/pet policy/etc.

    -- Zero-loss raw captures
    raw_search JSON,                        -- full search item (photoList, video, surrounding, ...)
    raw_metadata JSON,                      -- full rent-detail data{}
    scraper_raw JSON,                       -- 591scraper DOM fallback
    description TEXT,                       -- remark.content (API) or scraper desc
    image_urls JSON,
    image_paths JSON,                       -- local .webp paths

    -- Qwen Extracted Features
    qwen_warnings JSON,                     -- e.g., ["4th floor walk-up", "Elec $6/kWh"]
    qwen_vision_flags JSON,                 -- e.g., {"has_bathroom_img": true, "shower_sink_combo": false, ...}
    qwen_direct_score REAL,                 -- 1.0 to 5.0 native Qwen score (cold-start score)
    qwen_score REAL,                        -- Layer 2 normalized: (qwen_direct_score - 1) / 4, 0.0-1.0
    dino_visual_score REAL,                 -- Layer 1 scalar 0.0-1.0 (centroid cosine / Ridge probe)

    -- Feature Vectors & scores
    dino_embedding BLOB,                    -- mean-aggregated float32 (768-dim); raw input to Layer 1
    predicted_score REAL,
    score_source TEXT,                      -- 'qwen' | 'xgboost'
    heuristic_score REAL,                   -- Stage-3 penalty engine, baseline 100 (591research.md §4)
    bath_model_score REAL,                  -- bathroom probe /5.0 (NULL = no labelled bathroom photo)

    -- Hybrid relay mode (README "Hybrid PC-proxy mode")
    image_status TEXT DEFAULT 'pending',    -- pending → completed | failed | skipped
    text_only_notified BOOLEAN DEFAULT FALSE,  -- anti-spam flag for the offline proxy alert

    -- Active Learning Feedback
    user_rated BOOLEAN DEFAULT FALSE,
    user_score REAL,                        -- Overall rating (1.0 to 5.0)
    bathroom_score REAL,                    -- Bathroom score (1.0 to 5.0)
    user_comments TEXT,                     -- User + partner raw notes

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS listing_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id TEXT NOT NULL REFERENCES listings(listing_id),
    ordinal INTEGER,                        -- 0-based photo order
    image_url TEXT, image_path TEXT,        -- path to .webp
    dino_embedding BLOB,                    -- per-image float32 (768-dim)
    is_bathroom INTEGER,                    -- Qwen label: 1 bathroom / 0 not / NULL unlabelled
    UNIQUE(listing_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_lid ON listing_images(listing_id);

CREATE TABLE IF NOT EXISTS dynamic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_bullet_list TEXT,                -- Current synthesized prompt bullet points
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relay_state (
    listing_id     TEXT PRIMARY KEY,        -- per-listing payload dedup gate
    payload_sha256 TEXT,                    -- unchanged payloads skipped on both sides
    processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Image Storage Pipeline (standardized)
- Fetch original-resolution images by stripping the 591 CDN resize suffix (e.g. `!510x400.jpg`) from `photoList`.
- Re-encode every downloaded image to **WebP, quality=85** (Pillow) and store at `data/images/{listing_id}/{ordinal:02d}.webp`.
- WebP q85 keeps ~visual fidelity for Qwen-VL analysis and DINOv3 feature extraction while cutting disk usage roughly 40–70% vs JPEG originals.
- Per-image download/encode failures are non-fatal; the listing is still stored with whatever images succeeded.

---

## 4. Subsystem Implementations

### Component 1: Scraping & Merging Layer

Integrate both repositories to ingest listings based on hard query bounds (e.g., Region: Taipei/New Taipei, Rent Range: NT$15,000 - $25,000).

```python
import json
import requests
from DrissionPage import ChromiumPage, ChromiumOptions

def fetch_raw_listings():
    listings = {}
    
    # Primary: mcp-591 API execution via Client call
    try:
        from mcp_591.client import Client591
        client = Client591()
        rent_results = client.search_rent(
            region="台北市", 
            price_str="15000_25000", 
            kind="整層住家"
        )
        for item in rent_results:
            pid = str(item["post_id"])
            detail = client.get_rent_detail(pid)
            listings[pid] = {
                "id": pid,
                "title": detail.get("title", ""),
                "price": detail.get("price", 0),
                "url": f"https://rent.591.com.tw/{pid}",
                "images": detail.get("facilities", {}).get("images", []),
                "description": detail.get("description", ""),
                "raw": detail
            }
    except Exception as e:
        print(f"[Ingestion] mcp-591 fetch fallback triggered: {e}")

    # Secondary: 591scraper DrissionPage fallback for DOM state
    co = ChromiumOptions()
    co.set_paths(user_data_path="./browser_profile")
    page = ChromiumPage(co)
    page.get("https://rent.591.com.tw/?kind=1&region=1&rentprice=15000,25000")
    
    # Merge items into unified dict by ID
    return listings

```

---

### Component 2: Active Check & Multi-Image DINOv3 Deduplication

Before executing expensive visual LLM passes, verify listing availability and perform set-based cosine similarity on image embeddings to skip re-uploaded or duplicate properties.

#### Feature Extractor: Meta DINOv3 (`dinov3-vit-base`)

The extractor is upgraded from `facebook/dinov2-base` (DINOv2) to **Meta DINOv3 ViT-B/16**
(`dinov3-vit-base`, self-supervised LVD-1689M pretraining). Loading via Hugging Face
`transformers` (`AutoModel` / `AutoImageProcessor`) keeps the downstream contract intact:
the CLS token is still a **768-dimensional float32** vector, so the SQLite BLOB schema is
unchanged. The raw vector no longer feeds XGBoost directly — Layer 1
(`src/visual_preference.py`) compresses it to the scalar `dino_visual_score` (0.0–1.0).

```python
import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "models" / "dinov3_cache"   # offline weight cache (GPU server)
HUB_REPO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # DINOv3 ViT-B/16

_MODEL = None
_PROCESSOR = None

def load_dinov3():
    """Offline-first: use staged weights under models/dinov3_cache/, else pull from HF."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    local = Path(os.environ.get("DINOV3_MODEL_PATH", str(DEFAULT_CACHE / "facebook_dinov3-vit-base")))
    source, kwargs = (str(local), {}) if (local / "config.json").is_file() else \
                     (HUB_REPO_ID, {"cache_dir": str(DEFAULT_CACHE)})
    _PROCESSOR = AutoImageProcessor.from_pretrained(source, **kwargs)
    _MODEL = AutoModel.from_pretrained(source, **kwargs)
    _MODEL.eval().cuda()  # fp32 on GPU; L2-normalized CLS in embed step

def embed_dinov3(image_path):
    """768-dim float32 L2-normalized CLS embedding for one image."""
    import torch, numpy as np
    from PIL import Image
    model, processor = load_dinov3()
    img = Image.open(image_path).convert("RGB")
    inputs = {k: v.cuda() for k, v in processor(images=[img], return_tensors="pt").items()}
    with torch.no_grad():
        vec = model(**inputs).last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else None
```

**Why DINOv3 over DINOv2:**

- **RoPE positional embeddings** replace learnable absolute positions — resolution-independent
  attention with higher spatial detail recognition (exact fixture/room placement matters when
  deciding whether two photos show the same bathroom).
- **Register tokens + improved dense patch features** — better preservation of fine texture and
  layout cues across bathroom/layout photos, which sharpens the per-image cosine signal that the
  group-average dedup formula consumes.
- **Same vector contract** — ViT-B/16 still emits a 768-dim CLS embedding; only feature quality
  improves, so no BLOB-schema changes are required (the XGBoost head consumes the Layer-1
  compressed `dino_visual_score`, not the raw embedding).

#### Group Cosine Similarity Formula

For listing image embedding set $A$ and existing database listing image set $B$:

$$\text{Group Similarity}(A, B) = \frac{1}{\vert{}A\vert{}} \sum_{a \in A} \max_{b \in B} \frac{\vec{a} \cdot \vec{b}}{\Vert{}\vec{a}\Vert{} \Vert{}\vec{b}\Vert{}}$$

```python
import numpy as np

def compute_cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def check_group_duplication(new_image_vectors, stored_listings_vectors_dict, threshold=0.95):
    """
    new_image_vectors: List of DINO feature vectors for the new listing
    stored_listings_vectors_dict: Dict[listing_id, List[DINO vectors]]
    """
    for listing_id, stored_vectors in stored_listings_vectors_dict.items():
        sim_scores = []
        for vec_a in new_image_vectors:
            max_sim = max([compute_cosine_similarity(vec_a, vec_b) for vec_b in stored_vectors])
            sim_scores.append(max_sim)
            
        group_sim = np.mean(sim_scores)
        if group_sim >= threshold:
            return True, listing_id  # Duplicate found
            
    return False, None

```

---

### Component 3: Vision & Text Parsing (Qwen 27B)

The LLM call uses a **Two-Tier System Prompt**: an immutable Base System Prompt (enforcing structured JSON) and a Dynamic Preference Prompt (updated from user feedback).

#### System Prompt Blueprint

```python
BASE_SYSTEM_PROMPT = """
You are an expert real estate auditor analyzing a Rent591 apartment listing.
You must extract facts from Chinese listing text and analyze all provided images.

Return ONLY a valid JSON object matching this schema:
{
  "qwen_warnings": ["string warning highlights, e.g. 4th floor walk-up, shared meter"],
  "vision_flags": {
    "has_bathroom_img": bool,
    "shower_sink_combo": bool,      // True if shower hose connects directly to sink faucet
    "drainage_risk": bool,          // True if open wet room with no curb/door
    "has_kitchen_sink": bool,       // True if dedicated kitchen/countertop sink exists
    "has_exterior_window": bool     // True if bathroom has direct exterior window
  },
  "qwen_direct_score": float        // 1.0 to 5.0 score based on user context rules
}
"""
// NOTE: the emitted field is qwen_direct_score (the legacy "predicted_score" key is still
// accepted by the parser for old payloads). "predicted_score" is reserved for the Layer-3
// XGBoost output; downstream, qwen_direct_score is normalized to qwen_score = (score-1)/4.

def construct_full_prompt(sqlite_conn):
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT prompt_bullet_list FROM dynamic_preferences ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    
    dynamic_rules = row[0] if row else "- Prioritize dry/wet separation in bathroom.\n- Flag shower-sink combo faucet setups."
    
    return f"{BASE_SYSTEM_PROMPT}\n\n### User Context & Evolving Preferences ###\n{dynamic_rules}"

```

---

### Component 4: Scoring Engine — 3-Layer Preference Fusion

**Layer 1 — Visual Preference Engine** (`src/visual_preference.py`): compresses the
768-dim DINOv3 embedding into a single normalized scalar `dino_visual_score` (0.0–1.0).

* **Phase 1 — Cold Start ($\le 20$ labeled listings):** cosine similarity between the
  listing embedding and the mean embedding ($\vec{V}_{\text{liked}}$) of liked listings
  (`user_score >= 4.0`), clipped to [0,1]. Avoids parameter overfitting while data is scarce.
* **Phase 2 — Matured Model ($> 20$ labeled listings):** automatic transition to a
  **closed-form Ridge linear probe** on the raw 768-dim vectors (regression target
  $(\text{user\_score}-1)/4$, output clipped to [0,1]), learning per-dimension weights that
  penalize dark/dated spaces and reward preferred aesthetics. Weights persist to
  `models/dino_probe.npz` (env `DINO_PROBE_PATH`).

**Layer 2 — Textual Preference Memory** (see Component 3): Qwen's `qwen_direct_score`
(1–5) is normalized to `qwen_score = (qwen_direct_score - 1) / 4` (0.0–1.0).

**Layer 3 — Feature Fusion & XGBoost Regressor** (`src/scoring.py`):

* **Cold-Start Phase ($\le 20$ rated):** `predicted_score = qwen_direct_score`
  (`score_source='qwen'`); Layer 1 still computes and stores `dino_visual_score` via centroid.
* **Supervised Phase ($> 20$ rated):** `XGBRegressor(max_depth=3, n_estimators=50, lr=0.1)`
  trained on `user_score` over the compressed fusion vector, persisted to
  `models/xgboost_head.json`, prediction clamped to [1.0, 5.0]:

```text
X = [dino_visual_score, qwen_score,
     has_bathroom_img, shower_sink_combo, drainage_risk, has_kitchen_sink,
     has_exterior_window, has_warnings,
     log1p(price), area_ping, price_per_ping, floor_num,
     HIGH_ELEC_FEE, MANUAL_TRASH]              # scoring.FEATURE_NAMES (14 dims)
```

The raw 768-dim embedding is never concatenated into the XGBoost vector; every stage
fails soft (probe/model load failures fall back to the liked centroid / Qwen score).
* **Unsupervised Feature Learning:** Store all unrated listing embeddings in SQLite for dimensionality reduction (PCA/UMAP) and cluster feature analysis.

---

### Component 5: Push Notification Payload (`ntfy.sh`)

When `predicted_score >= 3.5`, trigger an HTTP `POST` to `ntfy.sh`. Scores render with
3 significant figures (`{score:.2f}` on the 1–5 scale) in both the body and the Title.

```python
import requests

def send_ntfy_alert(topic_name, listing):
    warnings_str = " | ".join(listing["qwen_warnings"]) if listing["qwen_warnings"] else "No major issues"
    
    message = f"NT${listing['price']} - {listing['title']}\n⚠️ {warnings_str}"
    
    headers = {
        "Title": f"Apartment Match ({listing['predicted_score']:.2f}/5)",
        "Click": listing["url"],
        "Tags": "house,bathroom"
    }
    
    requests.post(f"https://ntfy.sh/{topic_name}", data=message.encode("utf-8"), headers=headers)

```

---

### Component 6: User Feedback & CLI Tool (`rate.py`)

Users enter partner ratings and text comments directly into the server database using a simple command-line interface.

```python
# rate.py
import argparse
import sqlite3

parser = argparse.ArgumentParser(description="Rate a 591 Apartment")
parser.add_argument("--id", required=True, help="591 Listing ID")
parser.add_argument("--score", type=float, required=True, help="Overall Score 1-5")
parser.add_argument("--bathroom", type=float, default=3.0, help="Bathroom Score 1-5")
parser.add_argument("--comment", type=str, default="", help="User + Partner feedback text")

args = parser.parse_args()

conn = sqlite3.connect("apartments.db")
cursor = conn.cursor()

# 1. Save user ratings
cursor.execute("""
    UPDATE listings 
    SET user_rated = TRUE, user_score = ?, bathroom_score = ?, user_comments = ?
    WHERE listing_id = ?
""", (args.score, args.bathroom, args.comment, args.id))
conn.commit()

print(f"✅ Rating saved for listing {args.id}.")

# 2. Trigger Dynamic Prompt Consolidation via Qwen
if args.comment:
    cursor.execute("SELECT prompt_bullet_list FROM dynamic_preferences ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    current_bullets = row[0] if row else ""
    
    summarizer_prompt = f"""
    Current user preferences:
    {current_bullets}
    
    New feedback received: "{args.comment}"
    
    Task: Update the bulleted list of user preferences (max 7 items). 
    If the feedback introduces a new rule, add or update a bullet.
    If it is redundant or invalid, keep the list unchanged.
    Return ONLY the bulleted list.
    """
    
    # Execute local Qwen text call & insert updated list into dynamic_preferences table
    # ...
    print("✅ Dynamic Preference Prompt updated.")

conn.close()

```

---

## 5. Repository Structure Layout

When creating the repository, follow this folder layout:

```text
591-mcp-vision/
├── browser_profile/             # Persistent DrissionPage CDP browser profile
├── data/
│   └── apartments.db            # SQLite primary database
├── models/
│   ├── xgboost_head.json        # Saved XGBoost weights
│   ├── dino_probe.npz           # Layer-1 Ridge probe weights (768-d -> dino_visual_score)
│   └── dinov3_cache/            # Local DINOv3 model weights (offline)
│       └── facebook_dinov3-vit-base/  # config.json + model.safetensors (ViT-B/16)
├── src/
│   ├── __init__.py
│   ├── database.py              # SQLite connection, DDL, upserts
│   ├── client591.py            # Vendored mcp-591 Client591 (MIT)
│   ├── constants591.py         # Vendored region/section/kind ID maps
│   ├── ingestion.py             # Ingestion combining mcp-591 & 591scraper
│   ├── deduplication.py         # Multi-image DINO group cosine similarity
│   ├── visual_preference.py     # Layer 1: liked centroid / Ridge probe -> dino_visual_score
│   ├── vision_llm.py            # Qwen 27B (Ollama) wrapper & JSON parser
│   ├── scoring.py               # Layer 3: fusion vector builder, router & XGBoost head
│   ├── notifier.py              # ntfy.sh HTTP webhook publisher
│   └── dynamic_prompt.py        # Preference summarizer & prompt manager
├── rate.py                      # CLI feedback script
├── main.py                      # Core execution loop / Cron runner
├── pyproject.toml
└── README.md

```

---

## 6. Execution Loop Order (`main.py`)

1. **Query:** Run `ingestion.fetch_raw_listings()` to collect recent active listings.
2. **Filter Dead Links:** Validate `HTTP 200` status and verify non-expired listing text.
3. **Deduplicate:** Generate DINOv3 (ViT-B/16) CLS vectors for listing photos from `models/dinov3_cache` (offline after initial pull); evaluate group cosine similarity against SQLite database ($>0.95$ threshold drops listing).
4. **LLM Inference:** Pass photos and Chinese text to local Qwen 27B Vision API with Base + Dynamic system prompt.
5. **Predict:** Layer 1 computes `dino_visual_score` (centroid cosine if $\le 20$ ratings exist, Ridge probe if $> 20$); Layer 3 predicts with Qwen direct rating (if $\le 20$ ratings exist) or the XGBoost regressor on the fusion vector (if $> 20$ ratings exist).
6. **Notify:** If predicted score $\ge 3.5$, trigger `ntfy.sh` push notification with listing URL and warning highlights (score shown to 2 decimals, 3 significant figures).
7. **Store:** Persist features (`dino_visual_score`, `qwen_score`), warnings, and scores into SQLite `listings` table.