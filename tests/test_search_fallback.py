"""Unit tests for the 3-step search fallback pipeline."""

import pytest
import sys
import types
from unittest.mock import patch
from src.nodes.search import fetch_source_pack, search_tavily
from src.schemas import SourceItem, SourcePack


def test_search_tavily_primary_success(tmp_path):
    mock_items = [
        SourceItem(title="Tavily Item", url="https://example.com/tavily", snippet="Tavily snippet")
    ]
    with patch("src.nodes.search.search_tavily", return_value=mock_items):
        pack = fetch_source_pack("RAG Test", cache_dir=str(tmp_path))
        assert pack.source_tool == "tavily"
        assert len(pack.sources) == 1
        assert pack.sources[0].url == "https://example.com/tavily"


def test_search_fallback_to_ddg_when_tavily_fails(tmp_path):
    mock_items = [
        SourceItem(title="DDG Item", url="https://example.com/ddg", snippet="DDG snippet")
    ]
    with patch("src.nodes.search.search_tavily", return_value=None):
        with patch("src.nodes.search.search_duckduckgo", return_value=mock_items):
            pack = fetch_source_pack("RAG Test", cache_dir=str(tmp_path))
            assert pack.source_tool == "ddgs"
            assert len(pack.sources) == 1
            assert pack.sources[0].url == "https://example.com/ddg"


def test_search_fallback_to_cache_when_both_fail(tmp_path):
    with patch("src.nodes.search.search_tavily", return_value=None):
        with patch("src.nodes.search.search_duckduckgo", return_value=None):
            pack = fetch_source_pack("Introduction to RAG", cache_dir=str(tmp_path))
            assert pack.source_tool in ["cache", "offline_stub"]
            assert len(pack.sources) >= 1


def test_tavily_search_requests_detailed_material(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured = {}

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "test-key"

        def search(self, **kwargs):
            captured.update(kwargs)
            return {"results": [{
                "title": "Detailed source", "url": "https://example.com",
                "content": "Relevant chunks", "raw_content": "Full source text", "score": 0.9,
            }]}

    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeClient))
    items = search_tavily("topic", max_results=6, max_retries=1)

    assert items and items[0].raw_content == "Full source text"
    assert captured["search_depth"] == "advanced"
    assert captured["chunks_per_source"] == 3
    assert captured["include_raw_content"] is True
    assert captured["include_answer"] is False
    assert "reddit.com" in captured["exclude_domains"]


def test_legacy_snippet_cache_does_not_block_a_detailed_refresh(tmp_path):
    cache = tmp_path / "source_pack_machine_learning.json"
    cache.write_text('{"sources":[{"title":"Old","url":"https://old.example","snippet":"Old snippets"}],"fetched_at":"old"}')
    detailed = [SourceItem(title="Fresh", url="https://fresh.example", snippet="Detailed chunks", raw_content="Full text")]
    with patch("src.nodes.search.search_tavily", return_value=detailed):
        pack = fetch_source_pack("Machine Learning", cache_dir=str(tmp_path))
    assert pack.source_tool == "tavily"
    assert pack.retrieval_version == 2
    assert pack.sources[0].raw_content == "Full text"

