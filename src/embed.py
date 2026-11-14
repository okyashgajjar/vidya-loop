"""Gemini embedding client with a Postgres-backed content-addressed cache."""

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from dotenv import load_dotenv
from src.db import get_cached_embeddings, save_embeddings, text_hash

load_dotenv()

EMBEDDING_DIMENSIONS = 768
_calls_this_run = 0


def reset_embedding_budget() -> None:
    global _calls_this_run
    _calls_this_run = 0


def embedding_calls_used() -> int:
    return _calls_this_run


def embedding_call_limit() -> int:
    return int(os.getenv("EMBEDDING_CALLS_PER_RUN", "3"))


def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed several texts in one request, using the persistent cache where possible."""
    global _calls_this_run
    cleaned = [text.strip() for text in texts]
    results: List[Optional[List[float]]] = [None] * len(cleaned)
    cached = get_cached_embeddings(cleaned)
    missing = []
    for index, text in enumerate(cleaned):
        if not text:
            continue
        vector = cached.get(text_hash(text))
        if vector:
            results[index] = vector
        else:
            missing.append((index, text))
    if not missing:
        return results
    if _calls_this_run >= int(os.getenv("EMBEDDING_CALLS_PER_RUN", "3")):
        print("[Embed] Per-run budget reached; context retrieval skipped.")
        return results
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not key or not model:
        if not model:
            print("[Embed] Context retrieval skipped: EMBEDDING_MODEL is not configured.")
        return results
    payload = json.dumps({"requests": [
        {"model": model, "content": {"parts": [{"text": text}]},
         "outputDimensionality": EMBEDDING_DIMENSIONS}
        for _, text in missing
    ]}).encode("utf-8")
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/" + model + ":batchEmbedContents?key=" + key,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        _calls_this_run += 1
        with urllib.request.urlopen(request, timeout=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "8"))) as response:
            data = json.loads(response.read().decode("utf-8"))
        vectors = data.get("embeddings", [])
        if len(vectors) != len(missing):
            print(f"[Embed] Unexpected batch size: {len(vectors)}")
            return results
        to_save = []
        for (index, text), item in zip(missing, vectors):
            values = item.get("values", [])
            if len(values) != EMBEDDING_DIMENSIONS:
                print(f"[Embed] Unexpected dimensions: {len(values)}")
                continue
            vector = [float(value) for value in values]
            to_save.append((text, vector))
            results[index] = vector
        save_embeddings(to_save)
        return results
    except Exception as exc:
        print(f"[Embed] Context retrieval skipped: {type(exc).__name__}")
        return results


def embed_text(text: str) -> Optional[List[float]]:
    """Return a cached/generated 768-dimensional embedding, or None on outage."""
    return embed_texts([text])[0]
