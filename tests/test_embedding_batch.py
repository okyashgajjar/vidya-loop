import json

from src import db
from src import embed


def test_embeddings_fetch_cache_and_persist_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr("src.local_store._PATH", tmp_path / "store.json")
    monkeypatch.setattr(db, "_get_pool", lambda: None)
    monkeypatch.setattr(db, "_execute_many", lambda query, rows: False)
    db.save_embeddings([("cached", [1.0, 0.0])])
    assert db.get_cached_embeddings(["cached", "missing"]) == {db.text_hash("cached"): [1.0, 0.0]}

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/test-embedding")
    monkeypatch.setenv("EMBEDDING_CALLS_PER_RUN", "3")
    monkeypatch.setattr(embed, "get_cached_embeddings", lambda texts: {})
    persisted = []
    monkeypatch.setattr(embed, "save_embeddings", lambda items: persisted.extend(items) or True)
    request_data = {}

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({"embeddings": [{"values": [0.25] * 768}, {"values": [0.5] * 768}]}).encode()

    def fake_urlopen(request, timeout):
        request_data["url"] = request.full_url
        request_data["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(embed.urllib.request, "urlopen", fake_urlopen)
    embed.reset_embedding_budget()
    result = embed.embed_texts(["first document chunk", "second document chunk"])

    assert "batchEmbedContents" in request_data["url"]
    assert len(request_data["body"]["requests"]) == 2
    assert len(result) == len(persisted) == 2
    assert embed.embedding_calls_used() == 1
