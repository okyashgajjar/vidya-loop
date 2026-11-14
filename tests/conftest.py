import pytest


@pytest.fixture(autouse=True)
def disable_external_langsmith_tracing(monkeypatch):
    """Unit tests should not post traces to a live workspace."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
