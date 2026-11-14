"""Check configured model, search, embedding, and optional database connectivity."""

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src import llm
from src.db import database_status, delete_session, ensure_session, list_sessions
from src.embed import embed_text
from src.nodes.search import search_tavily


def report(name, passed, detail=""):
    print(f"{'PASS' if passed else 'FAIL'} | {name}" + (f" | {detail}" if detail else ""))
    return bool(passed)


def main() -> int:
    ok = True
    required = {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
        "DRAFT_MODEL": os.getenv("DRAFT_MODEL") or os.getenv("GEN_MODEL") or os.getenv("NVIDIA_MODEL"),
        "JUDGE_MODEL": os.getenv("JUDGE_MODEL") or os.getenv("NVIDIA_MODEL"),
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY") if os.getenv("LLM_PROVIDER", "nvidia") == "nvidia" else os.getenv("OPENROUTER_API_KEY"),
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
    }
    missing = [key for key, value in required.items() if not value]
    ok &= report("environment", not missing, "missing: " + ", ".join(missing) if missing else "required settings present")

    try:
        model = os.getenv("DRAFT_MODEL") or os.getenv("GEN_MODEL") or os.getenv("NVIDIA_MODEL") or None
        answer = llm.call_llm([{"role": "user", "content": "Reply with exactly: healthcheck ok"}],
                              model=model, max_tokens=40, max_retries=1, stage="healthcheck")
        ok &= report("model call", bool(answer.strip()), f"response received from {model or 'configured default'}")
    except Exception as exc:
        category, status = llm.classify_provider_error(exc)
        ok &= report("model call", False, f"{category} ({type(exc).__name__}, HTTP {status})")

    diagnostics = []
    try:
        results = search_tavily("Retrieval augmented generation overview", max_results=1,
                                max_retries=1, diagnostics=diagnostics)
        ok &= report("Tavily search", bool(results), "one source returned" if results else "; ".join(diagnostics[-2:]) or "no results")
    except Exception as exc:
        ok &= report("Tavily search", False, type(exc).__name__)

    try:
        vector = embed_text(f"healthcheck embedding sample {uuid.uuid4().hex}")
        ok &= report("embedding call", bool(vector), f"{len(vector)} dimensions" if vector else "embedding unavailable")
    except Exception as exc:
        ok &= report("embedding call", False, type(exc).__name__)

    status = database_status()
    if status.get("available"):
        sid = f"health-{uuid.uuid4().hex[:10]}"
        try:
            written = ensure_session(sid, "default")
            read_back = any(item.get("session_id") == sid for item in list_sessions("default"))
            ok &= report("Postgres write/read", written and read_back,
                         "session row round-tripped" if written and read_back else "write/read failed")
            delete_session(sid)
        except Exception as exc:
            ok &= report("Postgres write/read", False, type(exc).__name__)
    else:
        ok &= report("Postgres write/read", False, status.get("reason") or "database unavailable")
        sid = f"health-json-{uuid.uuid4().hex[:10]}"
        ensure_session(sid, "default")
        fallback_read = any(item.get("session_id") == sid for item in list_sessions("default"))
        report("JSON session fallback", fallback_read, "local session round-tripped" if fallback_read else "local write/read failed")
        delete_session(sid)

    print("OVERALL: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
