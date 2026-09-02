"""Dynamic preference prompt manager.

Fetches the latest preference bullets from the `dynamic_preferences` SQLite table,
falls back to DEFAULT_BULLETS (immutable user context rules) when the table is
empty, and appends them to the immutable BASE_SYSTEM_PROMPT. Also consolidates
new user feedback into <=7 bullets via a text-only Qwen call.
"""

from __future__ import annotations

import sqlite3

from . import database
from .vision_llm import DEFAULT_BULLETS, consolidate_preferences, construct_full_prompt


def get_bullets(conn: sqlite3.Connection) -> str:
    """Latest bulleted preferences from dynamic_preferences; default rules when empty."""
    return database.get_latest_preferences(conn) or DEFAULT_BULLETS


def build_system_prompt(conn: sqlite3.Connection) -> str:
    """Immutable base system prompt + latest dynamic preference bullets."""
    return construct_full_prompt(get_bullets(conn))


def update_preferences(conn: sqlite3.Connection, feedback: str) -> str:
    current = get_bullets(conn)
    bullets = consolidate_preferences(current, feedback)
    if bullets and bullets != current:
        database.save_preferences(conn, bullets)
    return bullets
