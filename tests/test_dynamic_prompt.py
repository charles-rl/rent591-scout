"""Offline dynamic_prompt tests: temp SQLite only, Qwen consolidation mocked."""

from src import database, dynamic_prompt, vision_llm


def test_empty_table_falls_back_to_default(tmp_path):
    conn = database.connect(tmp_path / "t.db")
    assert dynamic_prompt.get_bullets(conn) == vision_llm.DEFAULT_BULLETS
    conn.close()


def test_latest_row_wins(tmp_path):
    conn = database.connect(tmp_path / "t.db")
    database.save_preferences(conn, "- v1")
    database.save_preferences(conn, "- v2")
    assert dynamic_prompt.get_bullets(conn) == "- v2"
    conn.close()


def test_build_system_prompt_appends_bullets(tmp_path):
    conn = database.connect(tmp_path / "t.db")
    database.save_preferences(conn, "- Avoid top floors without elevator.")
    prompt = dynamic_prompt.build_system_prompt(conn)
    assert prompt.startswith(vision_llm.BASE_SYSTEM_PROMPT.rstrip())
    assert "- Avoid top floors without elevator." in prompt
    conn.close()


def test_update_preferences_saves_only_when_changed(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "t.db")
    monkeypatch.setattr(dynamic_prompt, "consolidate_preferences", lambda cur, fb: "- unchanged")
    dynamic_prompt.update_preferences(conn, "whatever")
    assert database.get_latest_preferences(conn) == "- unchanged"
    monkeypatch.setattr(dynamic_prompt, "consolidate_preferences", lambda cur, fb: cur)
    dynamic_prompt.update_preferences(conn, "again")
    rows = conn.execute("SELECT COUNT(*) FROM dynamic_preferences").fetchone()[0]
    assert rows == 1  # no duplicate write when consolidation returns identical bullets
    conn.close()
