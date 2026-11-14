"""Tests for topic-scoped vs profile-scoped failure lessons."""

import json
from pathlib import Path
from unittest.mock import patch


def _write_memory(profile_dir: Path, data: dict):
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "memory.json").write_text(json.dumps(data))


def test_style_lesson_applies_across_topics(tmp_path):
    """A readability failure on topic A should produce a lesson available for topic B."""
    with patch("src.memory.PROFILES_DIR", tmp_path):
        from src.memory import record_failures, get_active_lessons

        # Fail readability (a style check) on topic "RAG"
        record_failures(
            ["NATIVE_READABILITY"], profile_id="default", topic="Introduction to RAG",
            lessons_by_check={"NATIVE_READABILITY": ["Keep reading ease above 60."]},
        )

        # The style lesson should be available for a DIFFERENT topic
        lessons = get_active_lessons("default", "Neural Networks", max_lessons=5)
        assert any("reading ease" in l.lower() for l in lessons)


def test_content_lesson_stays_topic_scoped(tmp_path):
    """An accuracy failure on topic A should NOT produce a lesson for topic B."""
    with patch("src.memory.PROFILES_DIR", tmp_path):
        from src.memory import record_failures, get_active_lessons

        record_failures(
            ["JUDGE_ACCURACY_GROUNDING"], profile_id="default", topic="Introduction to RAG",
            lessons_by_check={"JUDGE_ACCURACY_GROUNDING": ["Ground every claim in sources for RAG."]},
        )

        # The content lesson should NOT be available for a different topic
        lessons = get_active_lessons("default", "Neural Networks", max_lessons=5)
        assert not any("RAG" in l for l in lessons)


def test_topic_lesson_available_for_same_topic(tmp_path):
    """A content lesson for topic A should be available when generating topic A again."""
    with patch("src.memory.PROFILES_DIR", tmp_path):
        from src.memory import record_failures, get_active_lessons

        record_failures(
            ["JUDGE_KEY_POINTS_COVERAGE"], profile_id="default", topic="Introduction to RAG",
            lessons_by_check={"JUDGE_KEY_POINTS_COVERAGE": ["Cover chunking step in RAG lesson."]},
        )

        lessons = get_active_lessons("default", "Introduction to RAG", max_lessons=5)
        assert any("chunking" in l.lower() for l in lessons)
