"""Old-schema migration path: connect() on a pre-hybrid DB must ADD the extra
columns and backfill image_status so already-processed listings do NOT flood
the pending queue (the 2026-08 bug class), while unfinished ones stay drainable."""

import sqlite3

import pytest

from src import database

# listings table as it existed before the hybrid pipeline: none of the
# database._EXTRA_COLUMNS (is_duplicate, image_status, text_only_notified,
# heuristic_score, qwen_score, dino_visual_score, social_house, facilities).
OLD_SCHEMA = """
CREATE TABLE listings (
    listing_id        TEXT PRIMARY KEY,
    title             TEXT,
    price             INTEGER,
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
"""


@pytest.fixture
def old_db(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO listings (listing_id, predicted_score, image_paths) "
                 "VALUES ('scored-no-images', 4.2, NULL)")
    conn.execute("INSERT INTO listings (listing_id, predicted_score, is_active) "
                 "VALUES ('unscored-active', NULL, 1)")
    conn.execute("INSERT INTO listings (listing_id, predicted_score, is_active) "
                 "VALUES ('dead-no-images', NULL, 0)")
    # 'completed is a lie': scored with photo paths, but the files never land
    # on disk in this fixture -> _requeue_missing_image_files requeues it.
    conn.execute("INSERT INTO listings (listing_id, predicted_score, image_urls, image_paths) "
                 "VALUES ('ghost-photos', 3.9, '[\"https://cdn/x.jpg\"]', '[\"/nonexistent/x.webp\"]')")
    real = tmp_path / "real.webp"
    real.write_bytes(b"\x00" * 5000)
    conn.execute("INSERT INTO listings (listing_id, predicted_score, image_urls, image_paths) "
                 "VALUES ('real-photos', 4.0, '[\"https://cdn/y.jpg\"]', ?)",
                 (f'["{real}"]',))
    # refresh_time sources for the days_on_market backfill snapshot. These rows
    # carry predicted_score + NULL image_paths so the image_status backfill
    # classifies them 'skipped' (already processed), keeping the status
    # assertion in test_migration_backfills_terminal_statuses focused on images.
    for lid, rt in (("fresh-rel", "此房屋在3天前發佈"), ("just-now", "此房屋在剛剛發佈"),
                    ("no-refresh", None), ("garbage-refresh", "not-a-date")):
        conn.execute("INSERT INTO listings (listing_id, predicted_score, image_paths, refresh_time) "
                     "VALUES (?, 3.5, NULL, ?)", (lid, rt))
    conn.commit()
    conn.close()
    return db_path


def _statuses(conn):
    return dict(conn.execute("SELECT listing_id, image_status FROM listings").fetchall())


def test_migration_backfills_terminal_statuses(old_db):
    conn = database.connect(old_db)
    try:
        st = _statuses(conn)
        assert st == {
            "scored-no-images": "skipped",   # through the old pipeline, no photos
            "unscored-active": "pending",    # vision never scored it -> drainable
            "dead-no-images": "skipped",     # inactive, nothing left to fetch
            "ghost-photos": "pending",       # requeued: 'completed' files are missing
            "real-photos": "completed",      # real >4KB photo on disk
            # days_on_market fixtures: already scored, no photos -> terminal
            "fresh-rel": "skipped",
            "just-now": "skipped",
            "no-refresh": "skipped",
            "garbage-refresh": "skipped",
        }
    finally:
        conn.close()


def test_migration_idempotent_on_reconnect(old_db):
    first = _statuses(database.connect(old_db))
    again = _statuses(database.connect(old_db))
    assert again == first


def _doms(conn):
    return dict(conn.execute("SELECT listing_id, days_on_market FROM listings").fetchall())


def test_migration_backfills_days_on_market_snapshot(old_db):
    """Pre-migration rows get their age snapshotted from refresh_time.

    Relative 591 text is only meaningful at parse time, so the migration must
    freeze it into days_on_market rather than leaving the pipeline to re-derive a
    drifting value at every scoring pass.
    """
    conn = database.connect(old_db)
    try:
        doms = _doms(conn)
        assert doms["fresh-rel"] == 3.0            # '3天前' -> 3 days
        assert doms["just-now"] == 0.0             # '剛剛' -> brand new, not NULL
        assert doms["no-refresh"] is None          # unparseable stays NULL
        assert doms["garbage-refresh"] is None     # unparseable stays NULL
    finally:
        conn.close()


def test_days_on_market_backfill_idempotent(old_db):
    conn = database.connect(old_db)
    try:
        first = _doms(conn)
        # A manual edit must survive reconnect: the backfill only runs on the
        # migration itself, not on every connect().
        conn.execute("UPDATE listings SET days_on_market=99.0 WHERE listing_id='fresh-rel'")
        conn.commit()
        conn.close()
        again = _doms(database.connect(old_db))
        assert again["fresh-rel"] == 99.0
        assert {k: v for k, v in again.items() if k != "fresh-rel"} == \
               {k: v for k, v in first.items() if k != "fresh-rel"}
    finally:
        conn.close()
