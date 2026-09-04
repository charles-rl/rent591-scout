"""SQLite schema, connection management, and persistence helpers.

Schema (listings / listing_images / dynamic_preferences) is defined in
docs/data-maxification.md and IMPLEMENTATION_OUTLINE.md §3.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id        TEXT PRIMARY KEY,
    title             TEXT,
    price             INTEGER,
    price_unit        TEXT,
    url               TEXT,
    status            TEXT,
    is_active         BOOLEAN DEFAULT TRUE,
    is_duplicate      BOOLEAN DEFAULT FALSE,
    region            TEXT, section TEXT,
    address           TEXT,
    lat REAL, lng REAL,
    community_name TEXT, community_id INTEGER,
    layout TEXT, area REAL, floor TEXT, shape TEXT, kind_name TEXT,
    deposit TEXT, rent_per REAL, rent_per_unit TEXT,
    browse_count INTEGER, refresh_time TEXT,
    tags JSON, contain_cost JSON,
    raw_search JSON, raw_metadata JSON, scraper_raw JSON,
    social_house BOOLEAN, facilities JSON,
    description TEXT,
    image_urls JSON, image_paths JSON,
    qwen_warnings JSON, qwen_vision_flags JSON, qwen_direct_score REAL,
    qwen_score REAL,
    dino_visual_score REAL,
    dino_embedding BLOB,
    predicted_score REAL, score_source TEXT,
    heuristic_score REAL,
    image_status TEXT DEFAULT 'pending',
    text_only_notified BOOLEAN DEFAULT FALSE,
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
    is_bathroom INTEGER,
    UNIQUE(listing_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_lid ON listing_images(listing_id);

CREATE TABLE IF NOT EXISTS dynamic_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_bullet_list TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS relay_state (
    listing_id     TEXT PRIMARY KEY,
    payload_sha256 TEXT,
    processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path) if db_path else Path(__file__).resolve().parent.parent / "data" / "apartments.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    conn.commit()
    return conn


_EXTRA_COLUMNS = [
    ("social_house", "BOOLEAN"),
    ("facilities", "JSON"),
    ("is_duplicate", "BOOLEAN DEFAULT FALSE"),
    ("image_status", "TEXT DEFAULT 'pending'"),
    ("text_only_notified", "BOOLEAN DEFAULT FALSE"),
    ("heuristic_score", "REAL"),
    ("qwen_score", "REAL"),
    ("dino_visual_score", "REAL"),
    ("bath_model_score", "REAL"),
]

_IMAGE_EXTRA_COLUMNS = [
    ("is_bathroom", "INTEGER"),
]


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add columns introduced after the initial schema (migration)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(listings)").fetchall()}
    for name, decl in _EXTRA_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE listings ADD COLUMN {name} {decl}")
            if name == "image_status":
                _backfill_image_status(conn)
    img_existing = {r[1] for r in conn.execute("PRAGMA table_info(listing_images)").fetchall()}
    for name, decl in _IMAGE_EXTRA_COLUMNS:
        if name not in img_existing:
            conn.execute(f"ALTER TABLE listing_images ADD COLUMN {name} {decl}")


def _backfill_image_status(conn: sqlite3.Connection) -> None:
    """ALTER TABLE applies the DEFAULT to pre-existing rows, which would flood
    the queue with already-processed listings. Rows that made it through the
    pre-hybrid pipeline get their terminal status; only unfinished active
    listings (vision never scored them) stay 'pending' for the proxy to drain.
    """
    conn.execute(
        "UPDATE listings SET image_status = CASE "
        "WHEN image_paths IS NOT NULL THEN 'completed' ELSE 'skipped' END "
        "WHERE predicted_score IS NOT NULL OR IFNULL(is_active, 1) = 0"
    )
    _requeue_missing_image_files(conn)


def _requeue_missing_image_files(conn: sqlite3.Connection) -> None:
    """'completed' is a lie when the photo files never made it to disk (or are
    1KB solid-color placeholders from PLACEHOLDER_IMAGES fixture runs). Requeue
    those listings so the hybrid proxy drain re-downloads real photos.
    """
    root = Path(__file__).resolve().parent.parent
    rows = conn.execute(
        "SELECT listing_id, image_paths FROM listings "
        "WHERE image_status = 'completed' AND image_urls IS NOT NULL AND image_urls NOT IN ('', '[]')"
    ).fetchall()
    for listing_id, paths_json in rows:
        try:
            paths = json.loads(paths_json) if paths_json else []
        except (TypeError, ValueError):
            paths = []
        real = False
        for p in paths:
            if not p:
                continue
            f = Path(p) if Path(p).is_absolute() else root / p
            try:
                if f.is_file() and f.stat().st_size > 4096:
                    real = True
                    break
            except OSError:
                continue
        if not real:
            conn.execute(
                "UPDATE listings SET image_status = 'pending' WHERE listing_id = ?", (listing_id,)
            )


def _json(value):
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def upsert_listing(conn: sqlite3.Connection, listing: dict) -> None:
    cols = [
        "listing_id", "title", "price", "price_unit", "url", "status", "is_active",
        "is_duplicate",
        "region", "section", "address", "lat", "lng",
        "community_name", "community_id", "layout", "area", "floor", "shape", "kind_name",
        "deposit", "rent_per", "rent_per_unit", "browse_count", "refresh_time",
        "tags", "contain_cost", "raw_search", "raw_metadata", "scraper_raw",
        "social_house", "facilities",
        "description", "image_urls", "image_paths",
        "qwen_warnings", "qwen_vision_flags", "qwen_direct_score", "qwen_score",
        "dino_visual_score",
        "dino_embedding", "predicted_score", "score_source", "heuristic_score",
        "image_status", "bath_model_score",
    ]
    placeholders = ", ".join(
        # First insert: never bind a raw NULL — fall back to the stored flag
        # (inactive re-runs) then FALSE, so is_duplicate is never NULL.
        "COALESCE(:is_duplicate, (SELECT is_duplicate FROM listings WHERE listing_id = :listing_id), FALSE)"
        if c == "is_duplicate" else
        "COALESCE(:image_status, (SELECT image_status FROM listings WHERE listing_id = :listing_id), 'pending')"
        if c == "image_status" else f":{c}"
        for c in cols
    )
    # COALESCE keeps previously stored values when this run's payload is degraded
    # (e.g. detail fetch failed), preventing NULL-wipe of lat/tags/scores.
    sets = ", ".join(
        "is_duplicate=COALESCE(excluded.is_duplicate, listings.is_duplicate, FALSE)"
        if c == "is_duplicate" else
        "image_status=COALESCE(excluded.image_status, listings.image_status, 'pending')"
        if c == "image_status" else f"{c}=COALESCE(excluded.{c}, listings.{c})"
        for c in cols
    )
    sql = f"INSERT INTO listings ({', '.join(cols)}) VALUES ({placeholders}) " \
          f"ON CONFLICT(listing_id) DO UPDATE SET updated_at=CURRENT_TIMESTAMP, {sets}"
    row = {c: listing.get(c) for c in cols}
    if row.get("is_duplicate") is not None:
        row["is_duplicate"] = int(bool(row["is_duplicate"]))
    for j in ("tags", "contain_cost", "raw_search", "raw_metadata", "scraper_raw",
              "qwen_warnings", "qwen_vision_flags", "image_urls", "image_paths",
              "facilities"):
        if row.get(j) is not None and not isinstance(row[j], str):
            row[j] = _json(row[j])
    conn.execute(sql, row)
    conn.commit()


def replace_images(conn: sqlite3.Connection, listing_id: str, images: list[dict]) -> None:
    conn.execute("DELETE FROM listing_images WHERE listing_id=?", (listing_id,))
    for img in images:
        conn.execute(
            "INSERT OR REPLACE INTO listing_images (listing_id, ordinal, image_url, image_path, dino_embedding, is_bathroom) "
            "VALUES (?,?,?,?,?,?)",
            (listing_id, img.get("ordinal"), img.get("image_url"), img.get("image_path"),
             img.get("dino_embedding"), img.get("is_bathroom")),
        )
    conn.commit()


def get_all_images(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    rows = conn.execute(
        "SELECT listing_id, ordinal, image_url, image_path, dino_embedding FROM listing_images ORDER BY ordinal"
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["listing_id"], []).append(
            {"ordinal": r["ordinal"], "image_url": r["image_url"],
             "image_path": r["image_path"], "dino_embedding": r["dino_embedding"]}
        )
    return out


def get_rating_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM listings WHERE user_rated = 1").fetchone()[0]


def get_rated_samples(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT listing_id, dino_embedding, qwen_vision_flags, qwen_warnings, user_score, qwen_direct_score, "
        "price, area, floor, shape, tags, facilities, contain_cost, description, bath_model_score "
        "FROM listings WHERE user_rated = 1"
    ).fetchall()


def get_scoring_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT listing_id, dino_embedding, qwen_vision_flags, qwen_warnings, qwen_direct_score, "
        "price, area, floor, shape, tags, facilities, contain_cost, description, bath_model_score "
        "FROM listings WHERE IFNULL(user_rated, 0) != 1"
    ).fetchall()


def get_bath_rated_samples(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rated bathroom embeddings: one row per bathroom photo of bathroom-rated listings."""
    return conn.execute(
        "SELECT l.listing_id, l.bathroom_score, i.dino_embedding FROM listings l "
        "JOIN listing_images i ON i.listing_id = l.listing_id "
        "WHERE l.bathroom_score IS NOT NULL AND IFNULL(i.is_bathroom, 0) = 1 "
        "AND i.dino_embedding IS NOT NULL"
    ).fetchall()


def get_bath_labeled_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All DINO-labelled bathroom photos (any listing) for the selection centroid."""
    return conn.execute(
        "SELECT dino_embedding FROM listing_images WHERE is_bathroom = 1 AND dino_embedding IS NOT NULL"
    ).fetchall()


def set_bath_image_flags(conn: sqlite3.Connection, listing_id: str, flags: dict[int, int]) -> None:
    """Persist per-photo bathroom labels {ordinal: 0|1}."""
    conn.executemany(
        "UPDATE listing_images SET is_bathroom=? WHERE listing_id=? AND ordinal=?",
        [(int(v), listing_id, int(k)) for k, v in flags.items()],
    )
    conn.commit()


def get_liked_embeddings(conn: sqlite3.Connection, min_score: float = 4.0) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT dino_embedding FROM listings WHERE user_rated = 1 AND user_score >= ?", (min_score,)
    ).fetchall()


def set_preference_scores(conn: sqlite3.Connection, updates: list[tuple[str, float, float]]) -> int:
    conn.executemany(
        "UPDATE listings SET dino_visual_score=?, qwen_score=?, updated_at=CURRENT_TIMESTAMP WHERE listing_id=?",
        [(dino, qwen, listing_id) for listing_id, dino, qwen in updates],
    )
    conn.commit()
    return len(updates)


def set_predicted_scores(conn: sqlite3.Connection, updates: list[tuple[str, float, str]]) -> int:
    conn.executemany(
        "UPDATE listings SET predicted_score=?, score_source=?, updated_at=CURRENT_TIMESTAMP WHERE listing_id=?",
        [(score, source, listing_id) for listing_id, score, source in updates],
    )
    conn.commit()
    return len(updates)


def get_listing_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT listing_id FROM listings")]


def get_latest_preferences(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT prompt_bullet_list FROM dynamic_preferences ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def save_preferences(conn: sqlite3.Connection, bullets: str) -> None:
    conn.execute("INSERT INTO dynamic_preferences (prompt_bullet_list) VALUES (?)", (bullets,))
    conn.commit()


def rate_listing(conn: sqlite3.Connection, listing_id: str, score: float,
                 bathroom: float = 3.0, comment: str = "") -> bool:
    cur = conn.execute(
        "UPDATE listings SET user_rated=1, user_score=?, bathroom_score=?, user_comments=? WHERE listing_id=?",
        (score, bathroom, comment, listing_id),
    )
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Pending-image queue (hybrid proxy mode)
# ---------------------------------------------------------------------------
def get_pending_image_listings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Active listings still awaiting image download/vision processing (oldest first)."""
    return conn.execute(
        "SELECT * FROM listings WHERE IFNULL(is_active, 1) = 1 AND image_status = 'pending' "
        "ORDER BY updated_at ASC"
    ).fetchall()


def set_image_status(conn: sqlite3.Connection, listing_id: str, status: str,
                     image_paths: list | None = None) -> bool:
    """Transition listings.image_status; optionally refresh image_paths JSON."""
    cur = conn.execute(
        "UPDATE listings SET image_status=?, image_paths=COALESCE(?, image_paths), "
        "updated_at=CURRENT_TIMESTAMP WHERE listing_id=?",
        (status, _json(image_paths) if image_paths is not None else None, str(listing_id)),
    )
    conn.commit()
    return cur.rowcount > 0


def get_completed_unscored(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Images ready but vision never finished (Ollama was down) — retried each run."""
    return conn.execute(
        "SELECT * FROM listings WHERE IFNULL(is_active, 1) = 1 "
        "AND image_status IN ('completed', 'skipped') AND predicted_score IS NULL "
        "AND IFNULL(is_duplicate, 0) = 0"
    ).fetchall()


def count_pending_unnotified(conn: sqlite3.Connection) -> int:
    """Pending listings that never triggered a proxy-request alert."""
    return conn.execute(
        "SELECT COUNT(*) FROM listings "
        "WHERE IFNULL(is_active, 1) = 1 AND image_status = 'pending' AND IFNULL(text_only_notified, 0) = 0"
    ).fetchone()[0]


def mark_text_only_notified(conn: sqlite3.Connection) -> int:
    """Flag all currently-pending listings as alerted (anti-spam for proxy requests)."""
    cur = conn.execute(
        "UPDATE listings SET text_only_notified = TRUE "
        "WHERE image_status = 'pending' AND IFNULL(text_only_notified, 0) = 0"
    )
    conn.commit()
    return cur.rowcount


def reset_text_only_notified(conn: sqlite3.Connection, listing_id: str) -> bool:
    """Re-arm the proxy-request alert after a payload update (changed photos)."""
    cur = conn.execute(
        "UPDATE listings SET text_only_notified = FALSE WHERE listing_id = ?", (str(listing_id),)
    )
    conn.commit()
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# GitHub relay processing state (avoids re-processing unchanged payloads)
# ---------------------------------------------------------------------------
def relay_is_processed(conn: sqlite3.Connection, listing_id: str, payload_sha256: str | None) -> bool:
    row = conn.execute(
        "SELECT payload_sha256 FROM relay_state WHERE listing_id=?", (listing_id,)
    ).fetchone()
    return bool(row) and row[0] is not None and row[0] == payload_sha256


def mark_relay_processed(conn: sqlite3.Connection, listing_id: str, payload_sha256: str | None) -> None:
    conn.execute(
        "INSERT INTO relay_state (listing_id, payload_sha256) VALUES (?,?) "
        "ON CONFLICT(listing_id) DO UPDATE SET payload_sha256=excluded.payload_sha256, "
        "processed_at=CURRENT_TIMESTAMP",
        (listing_id, payload_sha256),
    )
    conn.commit()
