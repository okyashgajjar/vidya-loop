"""Small JSON persistence fallback used when optional Postgres is unavailable."""

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict

_PATH = Path(os.getenv("LOCAL_STORE_PATH", "cache/local_store.json"))
_LOCK = threading.RLock()
_EMPTY: Dict[str, Any] = {
    "profiles": {}, "sessions": {}, "messages": {}, "runs": {}, "events": [],
    "embeddings": {}, "source_packs": {}, "lessons": {}, "glossary": [],
    "failure_lessons": [], "knowledge_chunks": [],
}


def read_store() -> Dict[str, Any]:
    with _LOCK:
        try:
            data = json.loads(_PATH.read_text(encoding="utf-8"))
            base = {key: ({} if isinstance(value, dict) else []) for key, value in _EMPTY.items()}
            base.update(data)
            return base
        except (OSError, json.JSONDecodeError, TypeError):
            return {key: ({} if isinstance(value, dict) else []) for key, value in _EMPTY.items()}


def update_store(update: Callable[[Dict[str, Any]], Any]) -> Any:
    with _LOCK:
        try:
            data = read_store()
            result = update(data)
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _PATH.with_suffix(_PATH.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(_PATH)
            return result
        except Exception as exc:
            print(f"[LocalStore] JSON fallback write failed: {type(exc).__name__}")
            return None
