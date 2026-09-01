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
                     ┌───────────────────────────────────┐
                     │ 4. Scoring Engine                 │
                     │    - First 20 Samples: Qwen Score │
                     │    - Post-20: XGBoost Model      │
                     │      (DINO Embedding + Qwen Flags)│
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

    -- Denormalized canonical fields (API-first, scraper fallback)
    region TEXT, section TEXT, address TEXT,
    lat REAL, lng REAL,
    community_name TEXT, community_id INTEGER,
    layout TEXT, area REAL, floor TEXT, shape TEXT, kind_name TEXT,
    deposit TEXT, rent_per REAL, rent_per_unit TEXT,
    browse_count INTEGER, refresh_time TEXT,
    tags JSON, contain_cost JSON,

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
    qwen_direct_score REAL,                 -- 1.0 to 5.0 cold-start score

    -- Feature Vectors
    dino_embedding BLOB,                    -- mean-aggregated float32 (768-dim)
    predicted_score REAL,
    score_source TEXT,                      -- 'qwen' | 'xgboost'

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
    UNIQUE(listing_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_lid ON listing_images(listing_id);

CREATE TABLE IF NOT EXISTS dynamic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_bullet_list TEXT,                -- Current synthesized prompt bullet points
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
the CLS token is still a **768-dimensional float32** vector, so XGBoost feature
concatenation and the SQLite BLOB schema are unchanged.

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
  improves, so no schema or XGBoost-head changes are required.

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
  "predicted_score": float          // 1.0 to 5.0 score based on user context rules
}
"""

def construct_full_prompt(sqlite_conn):
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT prompt_bullet_list FROM dynamic_preferences ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    
    dynamic_rules = row[0] if row else "- Prioritize dry/wet separation in bathroom.\n- Flag shower-sink combo faucet setups."
    
    return f"{BASE_SYSTEM_PROMPT}\n\n### User Context & Evolving Preferences ###\n{dynamic_rules}"

```

---

### Component 4: Scoring Engine (Cold-Start & XGBoost)

* **Cold-Start Phase ($\le 20$ Labeled Samples):** Set `predicted_score = qwen_direct_score`.
* **Supervised Phase ($> 20$ Labeled Samples):** Train an XGBoost classifier/regressor using the 768-dimensional DINOv3 multi-photo feature vector concatenated with binary Qwen metadata flags.
* **Unsupervised Feature Learning:** Store all unrated listing embeddings in SQLite for dimensionality reduction (PCA/UMAP) and cluster feature analysis.

```python
import xgboost as xgb
import numpy as np

def train_or_predict_score(db_cursor, new_item_features):
    # Check total user-rated samples
    db_cursor.execute("SELECT COUNT(*) FROM listings WHERE user_rated = TRUE")
    rated_count = db_cursor.fetchone()[0]
    
    if rated_count <= 20:
        # Cold Start: Return Qwen direct rating
        return new_item_features["qwen_direct_score"]
    
    # Extract training dataset
    db_cursor.execute("""
        SELECT dino_embedding, qwen_vision_flags, user_score 
        FROM listings WHERE user_rated = TRUE
    """)
    rows = db_cursor.fetchall()
    
    X_train, y_train = [], []
    for emb_blob, flags_json, target in rows:
        emb_vec = np.frombuffer(emb_blob, dtype=np.float32)
        flags = json.loads(flags_json)
        feature_vector = np.concatenate([emb_vec, list(flags.values())])
        X_train.append(feature_vector)
        y_train.append(target)
        
    # Fit light XGBoost Regressor
    model = xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
    model.fit(np.array(X_train), np.array(y_train))
    
    # Predict for new item
    new_vec = np.concatenate([new_item_features["dino_vec"], list(new_item_features["flags"].values())])
    return float(model.predict(np.array([new_vec]))[0])

```

---

### Component 5: Push Notification Payload (`ntfy.sh`)

When `predicted_score >= 3.5`, trigger an HTTP `POST` to `ntfy.sh`.

```python
import requests

def send_ntfy_alert(topic_name, listing):
    warnings_str = " | ".join(listing["qwen_warnings"]) if listing["qwen_warnings"] else "No major issues"
    
    message = f"NT${listing['price']} - {listing['title']}\n⚠️ {warnings_str}"
    
    headers = {
        "Title": f"Apartment Match ({listing['predicted_score']:.1f}/5)",
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
│   └── dinov3_cache/            # Local DINOv3 model weights (offline)
│       └── facebook_dinov3-vit-base/  # config.json + model.safetensors (ViT-B/16)
├── src/
│   ├── __init__.py
│   ├── database.py              # SQLite connection, DDL, upserts
│   ├── client591.py            # Vendored mcp-591 Client591 (MIT)
│   ├── constants591.py         # Vendored region/section/kind ID maps
│   ├── ingestion.py             # Ingestion combining mcp-591 & 591scraper
│   ├── deduplication.py         # Multi-image DINO group cosine similarity
│   ├── vision_llm.py            # Qwen 27B (Ollama) wrapper & JSON parser
│   ├── scoring.py               # Cold-start router & XGBoost inference
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
5. **Predict:** Compute score using Qwen direct rating (if $\le 20$ ratings exist) or XGBoost regressor (if $> 20$ ratings exist).
6. **Notify:** If predicted score $\ge 3.5$, trigger `ntfy.sh` push notification with listing URL and warning highlights.
7. **Store:** Persist features, warnings, and scores into SQLite `listings` table.