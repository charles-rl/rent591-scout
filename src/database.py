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
]


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add columns introduced after the initial schema (migration)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(listings)").fetchall()}
    for name, decl in _EXTRA_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE listings ADD COLUMN {name} {decl}")


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
        "qwen_warnings", "qwen_vision_flags", "qwen_direct_score",
        "dino_embedding", "predicted_score", "score_source",
    ]
    placeholders = ", ".join(
        # First insert: never bind a raw NULL — fall back to the stored flag
        # (inactive re-runs) then FALSE, so is_duplicate is never NULL.
        "COALESCE(:is_duplicate, (SELECT is_duplicate FROM listings WHERE listing_id = :listing_id), FALSE)"
        if c == "is_duplicate" else f":{c}"
        for c in cols
    )
    # COALESCE keeps previously stored values when this run's payload is degraded
    # (e.g. detail fetch failed), preventing NULL-wipe of lat/tags/scores.
    sets = ", ".join(
        "is_duplicate=COALESCE(excluded.is_duplicate, listings.is_duplicate, FALSE)"
        if c == "is_duplicate" else f"{c}=COALESCE(excluded.{c}, listings.{c})"
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
            "INSERT OR REPLACE INTO listing_images (listing_id, ordinal, image_url, image_path, dino_embedding) "
            "VALUES (?,?,?,?,?)",
            (listing_id, img.get("ordinal"), img.get("image_url"), img.get("image_path"), img.get("dino_embedding")),
        )
    conn.commit()


def get_all_images(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    rows = conn.execute("SELECT listing_id, ordinal, image_path, dino_embedding FROM listing_images").fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["listing_id"], []).append(
            {"ordinal": r["ordinal"], "image_path": r["image_path"], "dino_embedding": r["dino_embedding"]}
        )
    return out


def get_rating_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM listings WHERE user_rated = 1").fetchone()[0]


def get_rated_samples(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT dino_embedding, qwen_vision_flags, qwen_warnings, user_score FROM listings WHERE user_rated = 1"
    ).fetchall()


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
