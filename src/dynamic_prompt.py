"""Dynamic preference prompt manager: consolidate feedback into <=7 bullets via Qwen."""

from __future__ import annotations

import sqlite3

from . import database
from .vision_llm import consolidate_preferences


def update_preferences(conn: sqlite3.Connection, feedback: str) -> str:
    current = database.get_latest_preferences(conn)
    bullets = consolidate_preferences(current, feedback)
    if bullets and bullets != current:
        database.save_preferences(conn, bullets)
    return bullets
