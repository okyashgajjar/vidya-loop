"""Persistent, profile-scoped text knowledge ingestion and chunking."""

import hashlib
from io import BytesIO
from html.parser import HTMLParser
from typing import Any, Dict, List
from langsmith import traceable

from src.db import save_knowledge_chunks
from src.embed import embed_texts


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str):
        value = data.strip()
        if value:
            self.parts.append(value)


def clean_document_text(filename: str, content: bytes) -> str:
    """Extract text from plain text, Markdown, HTML, or PDF files."""
    if filename.casefold().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    text = content.decode("utf-8-sig", errors="replace")
    if filename.casefold().endswith((".html", ".htm")):
        parser = _TextExtractor()
        parser.feed(text)
        return "\n".join(parser.parts)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 160,
               max_chunks: int = 32) -> List[str]:
    """Split by paragraph first, then use bounded overlapping windows."""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(normalized) and len(chunks) < max_chunks:
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        value = normalized[start:end].strip()
        if value:
            chunks.append(value)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


@traceable(
    name="ingest_learning_document",
    process_inputs=lambda inputs: {"profile_id": inputs.get("profile_id"),
                                   "filename": inputs.get("filename"),
                                   "text_chars": len(inputs.get("text", ""))},
)
def ingest_document(profile_id: str, filename: str, text: str) -> Dict[str, Any]:
    parts = chunk_text(text)
    vectors = embed_texts(parts)
    entries = []
    for index, (content, vector) in enumerate(zip(parts, vectors), start=1):
        chunk_id = hashlib.sha256(f"{profile_id}\n{filename}\n{index}\n{content}".encode()).hexdigest()
        entries.append({
            "id": chunk_id, "profile_id": profile_id, "source_name": filename,
            "source_type": "uploaded_document", "source_url": "", "content": content,
            "embedding": vector,
        })
    return {"chunks_saved": save_knowledge_chunks(entries),
            "chunks_embedded": sum(vector is not None for vector in vectors),
            "chunks_total": len(parts)}
