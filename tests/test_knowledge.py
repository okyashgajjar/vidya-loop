from src import db
from src.knowledge import chunk_text, clean_document_text, ingest_document


def test_document_ingestion_persists_chunk_embeddings_across_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.local_store._PATH", tmp_path / "store.json")
    monkeypatch.setattr(db, "_get_pool", lambda: None)
    monkeypatch.setattr(db, "_execute", lambda *args, **kwargs: False)
    monkeypatch.setattr("src.knowledge.embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])

    result = ingest_document("learner-1", "notes.md", "RAG retrieves useful notes.\n\nEmbeddings represent meaning.")
    query = db.find_related_knowledge("learner-1", [1.0, 0.0], 0.9)
    other_profile = db.find_related_knowledge("learner-2", [1.0, 0.0], 0.9)

    assert result == {"chunks_saved": 1, "chunks_embedded": 1, "chunks_total": 1}
    assert query[0]["source_name"] == "notes.md"
    assert query[0]["similarity"] == 1.0
    assert other_profile == []


def test_document_chunking_and_html_extraction():
    chunks = chunk_text("A" * 900 + "\n\n" + "B" * 900, max_chars=1000, overlap=100)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert "Hello world" in clean_document_text("page.html", b"<p>Hello world</p>")
