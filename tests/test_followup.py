from unittest.mock import patch

import pytest

from src.followup import answer_followup, NoShippedLessonError


def test_followup_uses_one_direct_call_with_lesson_and_frozen_sources():
    answer = (
        "Chunking breaks a long file into small parts. Each part is easy to search. "
        "A chai stall can keep one card for each snack."
    )
    with patch("src.followup.call_llm", return_value=answer) as call:
        result = answer_followup(
            "Explain chunking like I am in 12th standard, with a chai stall example.",
            "The lesson explains retrieval.",
            {"sources": [{"title": "Source", "snippet": "Chunking splits long documents."}]},
            conversation_history=[
                {"role": "user", "content": "What is a chunk?"},
                {"role": "assistant", "content": "A chunk is one small document part."},
            ],
        )
    assert result["answer"] == answer
    assert result["passed"] is True
    assert call.call_count == 1
    system = call.call_args.args[0][0]["content"]
    user = call.call_args.args[0][1]["content"]
    assert "Extra, not in your lesson:" in system
    assert "Never refuse" in system
    assert "FROZEN SESSION SOURCE SNIPPETS" in user
    assert "RECENT TURNS FROM THIS SESSION ONLY" in user
    assert "A chunk is one small document part." in user
    assert "another session" in system
    assert "lesson does not contain" not in system.casefold()
    assert call.call_args.kwargs["max_retries"] == 1


def test_followup_is_blocked_without_shipped_lesson():
    with patch("src.followup.call_llm") as call:
        with pytest.raises(NoShippedLessonError):
            answer_followup("Explain chunking", "", {})
    call.assert_not_called()
