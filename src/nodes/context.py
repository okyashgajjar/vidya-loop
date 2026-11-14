"""Semantic context retrieval. This node never participates in quality gates."""

import os
from typing import Any, Dict

from src.db import find_related_context, find_related_knowledge
from src.embed import embed_text, reset_embedding_budget


def retrieve_semantic_context(topic: str, profile_id: str) -> Dict[str, Any]:
    reset_embedding_budget()
    embedding = embed_text(topic)
    if not embedding:
        return {"has_embedding": False, "memory_lessons": [], "prompt_block": "", "source_pack": None,
                "document_chunks": [], "similarity_hits": [], "status": "skipped",
                "skip_reason": "embedding unavailable"}
    threshold = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.60"))
    related = find_related_context(profile_id, embedding, threshold)
    document_chunks = find_related_knowledge(profile_id, embedding, threshold, limit=6)
    glossary = related.get("glossary", [])
    failures = related.get("failure_lessons", [])
    exemplar = related.get("exemplar")
    lines = []
    if glossary:
        lines.append("Related glossary from successful lessons (use only if helpful):")
        lines.extend(f"- {item['term']}: {item['definition']}" for item in glossary)
    if exemplar:
        lines.append("One related successful lesson style exemplar. Copy its clarity, never its facts:")
        lines.append(exemplar["body"][:2500])
    if document_chunks:
        lines.append("Related persistent document/site excerpts (use as grounding sources):")
        lines.extend(f"- {item['source_name']}: {item['content']}" for item in document_chunks)
    return {
        "has_embedding": True,
        "memory_lessons": [item["text"] for item in failures],
        "prompt_block": "\n".join(lines),
        "source_pack": related.get("source_pack"),
        "document_chunks": document_chunks,
        "similarity_hits": [
            {"kind": "glossary", "topic": item["topic"], "similarity": item["similarity"]}
            for item in glossary
        ] + [
            {"kind": "failure_lesson", "topic": item["topic"], "similarity": item["similarity"]}
            for item in failures
        ] + ([{"kind": "lesson_exemplar", "topic": exemplar["topic"], "similarity": exemplar["similarity"]}]
             if exemplar else []) + [
                 {"kind": item.get("source_type", "document"), "topic": item["source_name"],
                  "similarity": item["similarity"]} for item in document_chunks
             ] + ([{"kind": "source_pack", "topic": related["source_pack"]["topic"],
                                      "similarity": related["source_pack"]["similarity"]}]
                                     if related.get("source_pack") else []),
        "status": "hit" if (glossary or failures or exemplar or related.get("source_pack") or document_chunks) else "no_related_memory",
        "threshold": threshold,
    }
