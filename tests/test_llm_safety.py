"""Tests for safe handling of empty or reasoning-only model responses."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.llm import call_llm


def test_reasoning_is_never_returned_as_lesson_content():
    response = SimpleNamespace(
        model="test-model",
        usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning="private reasoning"), finish_reason="stop")],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
    )
    with patch("src.llm.get_openrouter_client", return_value=client):
        with pytest.raises(RuntimeError, match="empty message") as exc:
            call_llm([{"role": "user", "content": "Generate a lesson"}], max_retries=1)
    assert "private reasoning" not in str(exc.value)


def test_empty_stream_falls_back_to_user_facing_nonstream_response():
    calls = []
    fallback = SimpleNamespace(
        model="test-model", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content="A usable lesson."), finish_reason="stop")],
    )

    def create(**kwargs):
        calls.append(kwargs["stream"])
        return [] if kwargs["stream"] else fallback

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    chunks = []
    with patch("src.llm.get_openrouter_client", return_value=client):
        content = call_llm(
            [{"role": "user", "content": "Generate a lesson"}],
            max_retries=1,
            stream_callback=chunks.append,
        )
    assert content == "A usable lesson."
    assert calls == [True, False]
    assert chunks == ["\x00RESET\x00", "\x00RESET\x00", "A usable lesson."]


def test_truncated_stream_is_discarded_before_complete_fallback():
    requests = []
    streamed = SimpleNamespace(usage=None)
    def create(**kwargs):
        requests.append(kwargs)
        if kwargs["stream"]:
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial draft"), finish_reason=None)]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")]),
            ]
        return SimpleNamespace(model="test-model", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="complete draft"), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    visible = []
    def render(token):
        if token == "\x00RESET\x00":
            visible.clear()
        else:
            visible.append(token)

    with patch("src.llm.get_openrouter_client", return_value=client):
        content = call_llm([{"role": "user", "content": "draft"}], max_tokens=2048,
                           stream_callback=render, max_retries=1)
    assert content == "complete draft"
    assert "".join(visible) == "complete draft"
    assert [request["stream"] for request in requests] == [True, False]
    assert requests[0]["max_tokens"] == requests[1]["max_tokens"] == 2048
    assert requests[0].get("extra_body") == requests[1].get("extra_body")


def test_nonstream_truncation_retries_and_logs_finish_reason():
    response = SimpleNamespace(model="test-model", usage=None,
                               choices=[SimpleNamespace(message=SimpleNamespace(content="partial"), finish_reason="length")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    from src import llm
    llm.reset_usage_stats()
    with patch("src.llm.get_openrouter_client", return_value=client), patch("src.llm.time.sleep"):
        with pytest.raises(RuntimeError, match="truncated"):
            call_llm([{"role": "user", "content": "draft"}], max_retries=2, stage="generate")
    events = llm.USAGE_STATS["call_events"]
    assert len(events) == 2
    assert all(event["finish_reason"] == "length" for event in events)
    assert all(event["error_class"] == "truncated" for event in events)
