"""Search node: session cache → topic disk cache → Tavily → DDGS → stub."""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from langsmith import traceable
from src.schemas import SourceItem, SourcePack
from src.db import save_source_pack

# Load environment variables
load_dotenv()


def slugify_topic(topic: str) -> str:
    """Create a safe filename slug from a topic string."""
    clean = re.sub(r"[^\w\s-]", "", topic.lower()).strip()
    return re.sub(r"[-\s]+", "_", clean)


def search_tavily(query: str, max_results: int = 6, max_retries: int = 2,
                  diagnostics: Optional[List[str]] = None) -> Optional[List[SourceItem]]:
    """Retrieve detailed, source-grounded material from Tavily."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        if diagnostics is not None:
            diagnostics.append("Tavily skipped: TAVILY_API_KEY is not configured.")
        return None

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"Tavily client unavailable ({type(e).__name__}).")
        print(f"[Search:Tavily] Init error: {e}")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                chunks_per_source=3,
                include_raw_content=True,
                include_answer=False,
                exclude_domains=["reddit.com", "quora.com"],
            )
            raw_results = response.get("results", [])
            if raw_results:
                items = [
                    SourceItem(
                        title=r.get("title", "Untitled Source"),
                        url=r.get("url", ""),
                        snippet=(r.get("content") or r.get("raw_content") or "")[:5000],
                        score=float(r.get("score", 0.0)) if r.get("score") is not None else None,
                        raw_content=(r.get("raw_content") or "")[:8000] or None,
                    )
                    for r in raw_results
                    if r.get("url") and (r.get("content") or r.get("raw_content"))
                ]
                if items:
                    return items
        except Exception as e:
            if diagnostics is not None:
                diagnostics.append(f"Tavily attempt {attempt}: {type(e).__name__}.")
            print(f"[Search:Tavily] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    if diagnostics is not None and not diagnostics:
        diagnostics.append("Tavily returned no usable results.")
    return None


def search_duckduckgo(query: str, max_results: int = 4, max_retries: int = 2,
                      diagnostics: Optional[List[str]] = None) -> Optional[List[SourceItem]]:
    """Search DuckDuckGo with up to 2 retries."""
    try:
        from duckduckgo_search import DDGS
    except Exception as e:
        if diagnostics is not None:
            diagnostics.append(f"DDGS unavailable ({type(e).__name__}).")
        print(f"[Search:DDGS] Import error: {e}")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                if raw_results:
                    items = [
                        SourceItem(
                            title=r.get("title", "Untitled Source"),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                            score=None,
                        )
                        for r in raw_results
                        if r.get("href") and r.get("body")
                    ]
                    if items:
                        return items
        except Exception as e:
            if diagnostics is not None:
                diagnostics.append(f"DDGS attempt {attempt}: {type(e).__name__}.")
            print(f"[Search:DDGS] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    if diagnostics is not None and not diagnostics:
        diagnostics.append("DDGS returned no usable results.")
    return None


def load_cached_source_pack(topic: str, cache_dir: str = "cache") -> Optional[SourcePack]:
    """Load pre-cached source pack from disk. Topic-specific only — no RAG fallback."""
    slug = slugify_topic(topic)
    cache_path = Path(cache_dir) / f"source_pack_{slug}.json"
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sources = [SourceItem(**s) for s in data.get("sources", [])]
        return SourcePack(
            topic=topic,
            source_tool="cache",
            sources=sources,
            fetched_at=data.get("fetched_at", datetime.now(timezone.utc).isoformat()),
            retrieval_version=data.get("retrieval_version", 1),
        )
    except Exception as e:
        print(f"[Search:Cache] Failed to load cache from {cache_path}: {e}")
        return None


@traceable(name="search_grounding")
def fetch_source_pack(topic: str, cache_dir: str = "cache", session_sources: Optional[Dict[str, Any]] = None,
                      semantic_source_pack: Optional[Dict[str, Any]] = None) -> SourcePack:
    """Smart search: session cache → topic disk cache → Tavily → DDGS → stub."""
    now_str = datetime.now(timezone.utc).isoformat()
    diagnostics: List[str] = []
    clean_topic = re.sub(r"^(what\s+is|explain|introduction\s+to|teach\s+me\s+about)\s+", "", topic.strip(), flags=re.IGNORECASE)
    search_query = (
        f"Detailed beginner-friendly explanation of {clean_topic}: core concepts, "
        "how it works, applications, and practical examples"
    )

    if semantic_source_pack and semantic_source_pack.get("sources"):
        return SourcePack(topic=topic, source_tool="semantic_cache",
                          sources=[SourceItem(**source) for source in semantic_source_pack["sources"]],
                          fetched_at=now_str, retrieval_version=2,
                          retrieval_details={"cache": "semantic topic cache hit"})

    # Step 1: Session cache — reuse sources already fetched this session
    if (session_sources
            and int(session_sources.get("retrieval_version", 1)) >= 2
            and str(session_sources.get("topic", "")).strip().casefold() == topic.strip().casefold()):
        sources = session_sources.get("sources", [])
        if sources:
            return SourcePack(
                topic=topic,
                source_tool=session_sources.get("source_tool", "session_cache"),
                sources=[SourceItem(**s) if isinstance(s, dict) else s for s in sources],
                fetched_at=session_sources.get("fetched_at", now_str),
                retrieval_version=int(session_sources.get("retrieval_version", 1)),
                retrieval_details={"cache": "session cache hit"},
            )

    # Step 2: Topic-specific disk cache (no RAG fallback!)
    cached_pack = load_cached_source_pack(topic, cache_dir=cache_dir)
    if cached_pack and cached_pack.retrieval_version >= 2:
        save_source_pack(topic, [item.model_dump() for item in cached_pack.sources])
        return cached_pack

    # Step 3: Tavily primary
    tavily_results = search_tavily(search_query, max_results=6, max_retries=2, diagnostics=diagnostics)
    if tavily_results:
        pack = SourcePack(
            topic=topic,
            source_tool="tavily",
            sources=tavily_results,
            fetched_at=now_str,
            retrieval_version=2,
            retrieval_details={"served_by": "tavily", "prior_failures": diagnostics},
        )
        _save_to_cache(pack, cache_dir)
        save_source_pack(topic, pack.model_dump().get("sources", []))
        return pack

    # Step 4: DuckDuckGo fallback
    ddg_results = search_duckduckgo(search_query, max_results=4, max_retries=2, diagnostics=diagnostics)
    if ddg_results:
        pack = SourcePack(
            topic=topic,
            source_tool="ddgs",
            sources=ddg_results,
            fetched_at=now_str,
            retrieval_details={"served_by": "ddgs", "fallback_reason": diagnostics},
        )
        _save_to_cache(pack, cache_dir)
        save_source_pack(topic, pack.model_dump().get("sources", []))
        return pack

    # Keep legacy cached sources available offline, after attempting a fresh search.
    if cached_pack:
        cached_pack.retrieval_details = {"served_by": "topic_cache", "fallback_reason": diagnostics}
        save_source_pack(topic, [item.model_dump() for item in cached_pack.sources])
        return cached_pack

    # Step 5: Topic-aware minimal stub (NOT RAG-specific)
    return SourcePack(
        topic=topic,
        source_tool="offline_stub",
        sources=[
            SourceItem(
                title=f"Offline topic stub for {topic}",
                url="",
                snippet=f"No verified live source was available for {topic}. Treat this stub as a search failure notice, not factual grounding.",
            )
        ],
        fetched_at=now_str,
        retrieval_details={"served_by": "offline_stub", "fallback_reason": diagnostics or ["Live search returned no results."]},
    )


def _save_to_cache(pack: SourcePack, cache_dir: str):
    """Save active source pack to cache for future runs."""
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        slug = slugify_topic(pack.topic)
        file_path = Path(cache_dir) / f"source_pack_{slug}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pack.model_dump_json(indent=2))
    except Exception as e:
        print(f"[Search] Cache write failed: {e}")
