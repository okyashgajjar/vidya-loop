from src import db


def test_db_down_persists_and_restores_sessions_and_messages_in_json(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_get_pool", lambda: None)
    monkeypatch.setattr(db, "_disabled_reason", "database intentionally unavailable")
    monkeypatch.setattr("src.local_store._PATH", tmp_path / "local.json")

    db.ensure_session("session-test", "profile-test")
    db.start_run("followup-run-test", "session-test", "Follow-up: go deeper", update_title=False)
    assert next(item for item in db.list_sessions("profile-test")
                if item["session_id"] == "session-test")["title"] == "New lesson"
    db.save_message("session-test", "user", "Introduction to RAG")
    db.save_message("session-test", "assistant", "A short lesson.")

    sessions = db.list_sessions("profile-test")
    messages = db.load_session_messages("session-test")
    assert any(item["session_id"] == "session-test" for item in sessions)
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "A short lesson."

    shipped = {"run_id": "lesson-run", "status": "passed", "lesson_content": "A shipped lesson.",
               "source_pack": {"sources": [{"title": "Saved source", "snippet": "Evidence."}]}}
    db.save_session_artifact("session-test", shipped)
    assert db.load_session_artifact("session-test") == shipped
    failed = {"run_id": "failed-run", "status": "failed", "lesson_content": "", "source_pack": {}}
    db.save_session_artifact("session-test", failed)
    assert db.load_session_artifact("session-test") == failed

    # Existing sessions from before the artifact feature can still resume chat.
    db.ensure_session("legacy-session", "profile-test")
    db.start_run("legacy-lesson-run", "legacy-session", "Introduction to RAG")
    db.save_message("legacy-session", "user", "Introduction to RAG")
    db.save_message("legacy-session", "assistant", "A previously shipped lesson.")
    db.finish_run("legacy-lesson-run", "passed", 1, {})
    restored = db.load_session_artifact("legacy-session")
    assert restored["status"] == "passed"
    assert restored["lesson_content"] == "A previously shipped lesson."
    assert restored["run_id"] == "legacy-lesson-run"

    db.delete_session("session-test")
    assert not any(item["session_id"] == "session-test" for item in db.list_sessions("profile-test"))
