"""Optional Postgres persistence for semantic retrieval and run audits.

Every public helper fails closed to ``None`` or ``False``. JSON memory and JSONL
logging remain the application's fallback when Postgres is unavailable.
"""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from src.local_store import read_store, update_store

load_dotenv()

_pool = None
_disabled_reason: Optional[str] = None
_retry_after = 0.0
_knowledge_table_ready = False
_session_artifacts_table_ready = False


def _vector(values: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def database_status() -> Dict[str, Any]:
    pool = _get_pool()
    if pool is None:
        return {"available": False, "reason": _disabled_reason}
    try:
        with pool.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.profiles')")
                ready = cursor.fetchone()[0] is not None
        return {"available": ready, "reason": None if ready else "Database schema is not installed"}
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def _get_pool():
    global _pool, _disabled_reason, _retry_after
    if _pool is not None:
        return _pool
    if time.monotonic() < _retry_after:
        return None
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        _disabled_reason = "DATABASE_URL is not configured"
        _retry_after = time.monotonic() + 30
        return None
    try:
        from psycopg_pool import ConnectionPool
        pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "3")),
            timeout=float(os.getenv("DB_POOL_TIMEOUT_SECONDS", "2")),
            kwargs={"connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "2"))},
            open=True,
        )
        pool.wait(timeout=float(os.getenv("DB_POOL_TIMEOUT_SECONDS", "2")))
        _pool = pool
        _disabled_reason = None
        return _pool
    except Exception as exc:
        try:
            pool.close()
        except Exception:
            pass
        missing_package = getattr(exc, "name", None)
        _disabled_reason = (f"Install Python dependency '{missing_package}' from requirements.txt"
                            if missing_package else type(exc).__name__)
        retry_seconds = 60 if missing_package else float(os.getenv("DB_RETRY_SECONDS", "30"))
        _retry_after = time.monotonic() + retry_seconds
        print(f"[DB] Unavailable; retrying in {retry_seconds:.0f}s: {_disabled_reason}")
        return None


@contextmanager
def _connection() -> Iterator[Any]:
    pool = _get_pool()
    if pool is None:
        yield None
        return
    try:
        conn = pool.getconn(timeout=float(os.getenv("DB_POOL_TIMEOUT_SECONDS", "2")))
    except Exception as exc:
        print(f"[DB] Operation skipped: {type(exc).__name__}")
        yield None
        return
    try:
        yield conn
    finally:
        # Read helpers can leave an implicit transaction open. Return pooled
        # connections in a clean state instead of letting psycopg roll them
        # back noisily on checkout by the next request.
        try:
            if not conn.closed and conn.info.transaction_status != 0:  # 0 == IDLE
                conn.rollback()
        except Exception as exc:
            print(f"[DB] Connection cleanup skipped: {type(exc).__name__}")
        pool.putconn(conn)


def _execute(query: str, params: tuple = ()) -> bool:
    with _connection() as conn:
        if conn is None:
            return False
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            print(f"[DB] Write skipped: {type(exc).__name__}")
            return False


def _execute_many(query: str, rows: List[tuple]) -> bool:
    if not rows:
        return True
    with _connection() as conn:
        if conn is None:
            return False
        try:
            with conn.cursor() as cursor:
                cursor.executemany(query, rows)
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            print(f"[DB] Batch write skipped: {type(exc).__name__}")
            return False


def sync_profile(profile: Dict[str, Any]) -> bool:
    db_ok = _execute(
        """INSERT INTO profiles (id, name, level, prefs) VALUES (%s, %s, %s, %s::jsonb)
           ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, level = EXCLUDED.level,
           prefs = EXCLUDED.prefs, updated_at = now()""",
        (profile.get("profile_id", "default"), profile.get("name", "Learner"),
         profile.get("expertise_level", "beginner"), json.dumps(profile)),
    )
    update_store(lambda data: data["profiles"].update({profile.get("profile_id", "default"): profile}))
    return db_ok


def ensure_session(session_id: str, profile_id: str) -> bool:
    db_ok = _execute(
        "INSERT INTO sessions (id, profile_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
        (session_id, profile_id),
    )
    update_store(lambda data: data["sessions"].setdefault(session_id, {
        "session_id": session_id, "profile_id": profile_id, "title": "New lesson", "created_at": time.time()
    }))
    return db_ok


def list_sessions(profile_id: str) -> List[Dict[str, str]]:
    with _connection() as conn:
        if conn is None:
            sessions = [s for s in read_store()["sessions"].values() if s.get("profile_id") == profile_id]
            return sorted(sessions, key=lambda s: s.get("created_at", 0), reverse=True)
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT s.id, COALESCE((SELECT r.topic FROM runs r WHERE r.session_id=s.id
                               ORDER BY r.started_at DESC LIMIT 1), 'New lesson')
                       FROM sessions s WHERE s.profile_id=%s ORDER BY s.created_at DESC""",
                    (profile_id,),
                )
                result = [{"session_id": row[0], "title": row[1]} for row in cursor.fetchall()]
                return result or [s for s in read_store()["sessions"].values() if s.get("profile_id") == profile_id]
        except Exception as exc:
            print(f"[DB] Session restore skipped: {type(exc).__name__}")
            return [s for s in read_store()["sessions"].values() if s.get("profile_id") == profile_id]


def load_session_messages(session_id: str) -> List[Dict[str, str]]:
    with _connection() as conn:
        if conn is None:
            return read_store()["messages"].get(session_id, [])
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT role, content FROM messages WHERE session_id=%s ORDER BY created_at, id",
                    (session_id,),
                )
                rows = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
                return rows or read_store()["messages"].get(session_id, [])
        except Exception as exc:
            print(f"[DB] Message restore skipped: {type(exc).__name__}")
            return read_store()["messages"].get(session_id, [])


def save_session_artifact(session_id: str, artifact: Dict[str, Any]) -> bool:
    """Persist the latest lesson-run status and only the latest shipped lesson."""
    global _session_artifacts_table_ready
    if not _session_artifacts_table_ready:
        _session_artifacts_table_ready = _execute("""CREATE TABLE IF NOT EXISTS session_artifacts (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            artifact JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    db_ok = _execute(
        """INSERT INTO session_artifacts (session_id, artifact) VALUES (%s, %s::jsonb)
           ON CONFLICT (session_id) DO UPDATE SET artifact=EXCLUDED.artifact, updated_at=now()""",
        (session_id, json.dumps(artifact)),
    )
    update_store(lambda data: data["sessions"].setdefault(session_id, {}).update({"latest_artifact": artifact}))
    return db_ok


def load_session_artifact(session_id: str) -> Dict[str, Any]:
    with _connection() as conn:
        if conn is not None:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT artifact FROM session_artifacts WHERE session_id=%s", (session_id,))
                    row = cursor.fetchone()
                if row:
                    value = row[0]
                    return value if isinstance(value, dict) else json.loads(value)
            except Exception as exc:
                print(f"[DB] Session artifact restore skipped: {type(exc).__name__}")
    data = read_store()
    stored = data["sessions"].get(session_id, {}).get("latest_artifact")
    if stored:
        return stored

    # One-time compatibility path for lessons saved before session artifacts existed.
    runs = [{**run, "run_id": run_id} for run_id, run in data["runs"].items()
            if run.get("session_id") == session_id
            and not str(run.get("topic", "")).startswith(("Follow-up:", "Document ingestion:"))]
    if not runs:
        return {}
    latest = max(runs, key=lambda run: run.get("started_at", 0))
    topic = latest.get("topic", "")
    messages = data["messages"].get(session_id, [])
    lesson = ""
    for index in range(len(messages) - 1):
        if messages[index].get("role") == "user" and messages[index].get("content", "").strip() == topic:
            if messages[index + 1].get("role") == "assistant":
                lesson = messages[index + 1].get("content", "")
    passed = latest.get("status") == "passed"
    source_pack = data["source_packs"].get(topic, {}) if passed else {}
    return {"run_id": latest.get("run_id", ""), "status": latest.get("status", "failed"),
            "topic": topic, "lesson_content": lesson if passed else "",
            "source_pack": source_pack if passed else {}}


def delete_session(session_id: str) -> bool:
    db_ok = _execute("DELETE FROM sessions WHERE id=%s", (session_id,))
    def remove(data):
        data["sessions"].pop(session_id, None)
        data["messages"].pop(session_id, None)
        for rid in [rid for rid, run in data["runs"].items() if run.get("session_id") == session_id]:
            data["runs"].pop(rid, None)
    update_store(remove)
    return db_ok


def save_message(session_id: str, role: str, content: str) -> bool:
    db_ok = _execute("INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                     (session_id, role, content))
    update_store(lambda data: data["messages"].setdefault(session_id, []).append({"role": role, "content": content}))
    return db_ok


def start_run(run_id: str, session_id: str, topic: str, update_title: bool = True) -> bool:
    db_ok = _execute(
        """INSERT INTO runs (id, session_id, topic, status) VALUES (%s, %s, %s, 'in_progress')
           ON CONFLICT (id) DO NOTHING""", (run_id, session_id, topic),
    )
    def start_local(data):
        data["runs"].setdefault(run_id, {
            "session_id": session_id, "topic": topic, "status": "in_progress", "attempts": 1,
            "tokens": {}, "started_at": time.time(),
        })
        if update_title and session_id in data["sessions"]:
            data["sessions"][session_id]["title"] = topic[:36]
    update_store(start_local)
    return db_ok


def finish_run(run_id: str, status: str, attempts: int, tokens: Dict[str, Any]) -> bool:
    db_ok = _execute(
        "UPDATE runs SET status=%s, attempts=%s, tokens=%s::jsonb, ended_at=now() WHERE id=%s",
        (status, attempts, json.dumps(tokens), run_id),
    )
    def finish(data):
        if run_id in data["runs"]:
            data["runs"][run_id].update(status=status, attempts=attempts, tokens=tokens, ended_at=time.time())
    update_store(finish)
    return db_ok


def log_run_event(run_id: str, step: str, status: str, details: Dict[str, Any]) -> bool:
    db_ok = _execute(
        "INSERT INTO run_events (run_id, step, status, detail) VALUES (%s, %s, %s, %s::jsonb)",
        (run_id, step, status, json.dumps(details)),
    )
    update_store(lambda data: data["events"].append({"run_id": run_id, "step": step,
                                                       "status": status, "details": details}))
    return db_ok


def set_lesson_feedback(run_id: str, thumbs: bool) -> bool:
    db_ok = _execute("UPDATE lessons SET thumbs=%s WHERE run_id=%s", (1 if thumbs else -1, run_id))
    update_store(lambda data: data["lessons"].get(run_id, {}).update(thumbs=1 if thumbs else -1))
    return db_ok


def get_cached_embedding(text: str) -> Optional[List[float]]:
    return get_cached_embeddings([text]).get(text_hash(text))


def get_cached_embeddings(texts: List[str]) -> Dict[str, List[float]]:
    keys = list(dict.fromkeys(text_hash(text) for text in texts if text.strip()))
    if not keys:
        return {}
    with _connection() as conn:
        if conn is None:
            local = read_store()["embeddings"]
            return {key: local[key] for key in keys if key in local}
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT text_hash, embedding::text FROM embedding_cache WHERE text_hash = ANY(%s)",
                               (keys,))
                rows = cursor.fetchall()
            found = {key: [float(value) for value in value_text.strip("[]").split(",") if value]
                     for key, value_text in rows}
            local = read_store()["embeddings"]
            found.update({key: local[key] for key in keys if key not in found and key in local})
            return found
        except Exception as exc:
            print(f"[DB] Cache read skipped: {type(exc).__name__}")
            local = read_store()["embeddings"]
            return {key: local[key] for key in keys if key in local}


def save_embedding(text: str, embedding: List[float]) -> bool:
    return save_embeddings([(text, embedding)])


def save_embeddings(items: List[tuple]) -> bool:
    db_ok = _execute_many(
        """INSERT INTO embedding_cache (text_hash, embedding) VALUES (%s, %s::vector)
           ON CONFLICT (text_hash) DO NOTHING""",
        [(text_hash(text), _vector(vector)) for text, vector in items],
    )
    update_store(lambda data: [data["embeddings"].setdefault(text_hash(text), vector)
                               for text, vector in items])
    return db_ok


def find_related_context(profile_id: str, embedding: List[float], threshold: float, limit: int = 2) -> Dict[str, Any]:
    """Return only records above a caller-supplied similarity threshold."""
    result: Dict[str, Any] = {"glossary": [], "failure_lessons": [], "exemplar": None, "source_pack": None}
    vector = _vector(embedding)

    def local_context():
        import math
        data = read_store()
        def similarity(other):
            if not other or len(other) != len(embedding):
                return 0.0
            dot = sum(a * b for a, b in zip(embedding, other))
            norm_a = math.sqrt(sum(a * a for a in embedding))
            norm_b = math.sqrt(sum(b * b for b in other))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        gloss = []
        for item in data["glossary"]:
            score = similarity(item.get("embedding"))
            if item.get("profile_id") == profile_id and score >= threshold:
                gloss.append({**item, "similarity": score})
        result["glossary"] = sorted(gloss, key=lambda item: item["similarity"], reverse=True)[:10]
        failures = []
        for item in data["failure_lessons"]:
            score = similarity(item.get("embedding"))
            if score >= threshold:
                failures.append({**item, "similarity": score})
        result["failure_lessons"] = sorted(failures, key=lambda item: item["similarity"], reverse=True)[:3]
        lessons = [item for item in data["lessons"].values() if item.get("thumbs") == 1]
        lessons = sorted(lessons, key=lambda item: similarity(item.get("embedding")), reverse=True)
        if lessons and similarity(lessons[0].get("embedding")) >= threshold:
            result["exemplar"] = {**lessons[0], "similarity": similarity(lessons[0].get("embedding"))}
        packs = sorted(data["source_packs"].values(), key=lambda item: similarity(item.get("embedding")), reverse=True)
        if packs and similarity(packs[0].get("embedding")) >= threshold:
            result["source_pack"] = {**packs[0], "similarity": similarity(packs[0].get("embedding"))}
        return result

    with _connection() as conn:
        if conn is None:
            return local_context()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT term, definition, topic, 1 - (embedding <=> %s::vector) AS similarity
                       FROM glossary WHERE profile_id=%s AND embedding IS NOT NULL
                       AND 1 - (embedding <=> %s::vector) >= %s
                       ORDER BY embedding <=> %s::vector LIMIT %s""",
                    (vector, profile_id, vector, threshold, vector, 10),
                )
                result["glossary"] = [
                    {"term": row[0], "definition": row[1], "topic": row[2], "similarity": float(row[3])}
                    for row in cursor.fetchall()
                ]
                cursor.execute(
                    """SELECT text, check_id, topic, 1 - (embedding <=> %s::vector) AS similarity FROM failure_lessons
                       WHERE embedding IS NOT NULL AND 1 - (embedding <=> %s::vector) >= %s
                       ORDER BY embedding <=> %s::vector LIMIT %s""", (vector, vector, threshold, vector, 3),
                )
                result["failure_lessons"] = [
                    {"text": row[0], "check_id": row[1], "topic": row[2], "similarity": float(row[3])}
                    for row in cursor.fetchall()
                ]
                cursor.execute(
                    """SELECT body, topic, 1 - (embedding <=> %s::vector) AS similarity
                       FROM lessons WHERE thumbs=1 AND embedding IS NOT NULL
                       AND 1 - (embedding <=> %s::vector) >= %s
                       ORDER BY embedding <=> %s::vector LIMIT 1""", (vector, vector, threshold, vector),
                )
                row = cursor.fetchone()
                if row:
                    result["exemplar"] = {"body": row[0], "topic": row[1], "similarity": float(row[2])}
                cursor.execute(
                    """SELECT sources, topic, 1 - (embedding <=> %s::vector) AS similarity
                       FROM source_packs WHERE embedding IS NOT NULL
                       AND 1 - (embedding <=> %s::vector) >= %s
                       ORDER BY embedding <=> %s::vector LIMIT 1""", (vector, vector, threshold, vector),
                )
                row = cursor.fetchone()
                if row:
                    result["source_pack"] = {"sources": row[0], "topic": row[1], "similarity": float(row[2])}
            return result
        except Exception as exc:
            print(f"[DB] Semantic context skipped: {type(exc).__name__}")
            return local_context()


def save_source_pack(topic: str, sources: List[Dict[str, Any]], embedding: Optional[List[float]] = None) -> bool:
    embedding = embedding or get_cached_embedding(topic)
    db_ok = False
    if embedding:
        db_ok = _execute(
            """INSERT INTO source_packs (topic, embedding, sources) VALUES (%s, %s::vector, %s::jsonb)
               ON CONFLICT (topic) DO UPDATE SET embedding=EXCLUDED.embedding, sources=EXCLUDED.sources,
               fetched_at=now()""", (topic, _vector(embedding), json.dumps(sources)),
        )
    update_store(lambda data: data["source_packs"].update({topic: {
        "topic": topic, "sources": sources, "embedding": embedding,
    }}))
    return db_ok


def save_knowledge_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Persist profile-scoped document/site chunks in Postgres and JSON fallback."""
    global _knowledge_table_ready
    if not chunks:
        return 0
    if not _knowledge_table_ready:
        _knowledge_table_ready = _execute("""CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, source_name TEXT NOT NULL,
            source_type TEXT NOT NULL, source_url TEXT NOT NULL DEFAULT '', content TEXT NOT NULL,
            embedding vector(768), created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
    saved, rows = [], []
    for chunk in chunks:
        item = dict(chunk)
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        item.setdefault("id", text_hash("\n".join((item.get("profile_id", "default"),
                      item.get("source_name", ""), content))))
        item["content"] = content
        rows.append(
            (item["id"], item.get("profile_id", "default"), item.get("source_name", "Source"),
             item.get("source_type", "document"), item.get("source_url", ""), content,
             _vector(item["embedding"]) if item.get("embedding") else None)
        )
        saved.append(item)
    if saved:
        _execute_many(
            """INSERT INTO knowledge_chunks (id, profile_id, source_name, source_type, source_url, content, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
               ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content, embedding=EXCLUDED.embedding""",
            rows,
        )
        def upsert(data):
            existing = {item.get("id"): index for index, item in enumerate(data["knowledge_chunks"])}
            for item in saved:
                if item["id"] in existing:
                    data["knowledge_chunks"][existing[item["id"]]] = item
                else:
                    data["knowledge_chunks"].append(item)
        update_store(upsert)
    return len(saved)


def find_related_knowledge(profile_id: str, embedding: List[float], threshold: float,
                           limit: int = 6) -> List[Dict[str, Any]]:
    """Find semantically similar persistent document/site chunks for this learner."""
    vector = _vector(embedding)
    with _connection() as conn:
        if conn is not None:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """SELECT id, source_name, source_type, source_url, content,
                                  1 - (embedding <=> %s::vector) AS similarity
                           FROM knowledge_chunks WHERE profile_id=%s AND embedding IS NOT NULL
                             AND 1 - (embedding <=> %s::vector) >= %s
                           ORDER BY embedding <=> %s::vector LIMIT %s""",
                        (vector, profile_id, vector, threshold, vector, limit),
                    )
                    return [{"id": row[0], "source_name": row[1], "source_type": row[2],
                             "source_url": row[3], "content": row[4], "similarity": float(row[5])}
                            for row in cursor.fetchall()]
            except Exception as exc:
                print(f"[DB] Knowledge retrieval skipped: {type(exc).__name__}")
    import math
    def similarity(other):
        if not other or len(other) != len(embedding):
            return 0.0
        dot = sum(a * b for a, b in zip(embedding, other))
        norm_a = math.sqrt(sum(a * a for a in embedding))
        norm_b = math.sqrt(sum(b * b for b in other))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    matches = []
    for item in read_store()["knowledge_chunks"]:
        if item.get("profile_id", "default") != profile_id:
            continue
        score = similarity(item.get("embedding"))
        if score >= threshold:
            matches.append({**item, "similarity": score})
    return sorted(matches, key=lambda item: item["similarity"], reverse=True)[:limit]


def save_failure_lessons(topic: str, entries: List[Dict[str, Any]], embedding: Optional[List[float]] = None) -> bool:
    embedding = embedding or get_cached_embedding(topic)
    wrote = False
    for entry in entries:
        failed_ids = entry.get("failed_check_ids") or ["UNKNOWN"]
        reasons = entry.get("reasons", [])[:2]
        for index, reason in enumerate(reasons):
            text = str(reason).replace("\n", " ")[:280]
            if not text:
                continue
            check_id = failed_ids[min(index, len(failed_ids) - 1)]
            wrote = _execute(
                """INSERT INTO failure_lessons (check_id, topic, text, embedding) VALUES (%s, %s, %s, %s::vector)
                   ON CONFLICT (check_id, topic, text) DO UPDATE
                   SET count=failure_lessons.count + 1, last_seen=now()""",
                (check_id, topic, text, _vector(embedding) if embedding else None),
            ) or wrote
            update_store(lambda data, cid=check_id, txt=text: data["failure_lessons"].append({
                "check_id": cid, "topic": topic, "text": txt, "embedding": embedding,
            }))
    return wrote


def save_approved_lesson(run_id: str, topic: str, embedding: Optional[List[float]], body: str,
                         scorecard: Dict[str, Any]) -> bool:
    wrote = _execute(
        """INSERT INTO lessons (run_id, topic, embedding, body, scorecard, thumbs) VALUES
           (%s, %s, %s::vector, %s, %s::jsonb, 1)
           ON CONFLICT (run_id) DO UPDATE SET thumbs=1, body=EXCLUDED.body, scorecard=EXCLUDED.scorecard""",
        (run_id, topic, _vector(embedding) if embedding else None, body, json.dumps(scorecard)),
    )
    update_store(lambda data: data["lessons"].update({run_id: {
        "run_id": run_id, "topic": topic, "embedding": embedding, "body": body,
        "scorecard": scorecard, "thumbs": 1,
    }}))
    return wrote


def save_glossary_entries(profile_id: str, topic: str, glossary: List[Dict[str, Any]]) -> bool:
    wrote = False
    for item in glossary:
        term, definition = item.get("term", "").strip(), item.get("definition", "").strip()
        embedding = item.get("embedding")
        if not term or not definition:
            continue
        wrote = _execute(
            """INSERT INTO glossary (profile_id, term, definition, topic, embedding, thumbs_up)
               VALUES (%s, %s, %s, %s, %s::vector, 1)
               ON CONFLICT (profile_id, term, topic) DO UPDATE SET definition=EXCLUDED.definition,
               thumbs_up=GREATEST(glossary.thumbs_up, 1), last_used=now()""",
                (profile_id, term, definition, topic, _vector(embedding) if embedding else None),
            ) or wrote
        def upsert_local_glossary(data):
            key = (profile_id, term.casefold(), topic)
            for existing in data["glossary"]:
                if (existing.get("profile_id"), existing.get("term", "").casefold(), existing.get("topic")) == key:
                    existing.update(definition=definition, embedding=embedding)
                    return
            data["glossary"].append({"profile_id": profile_id, "term": term,
                                     "definition": definition, "topic": topic,
                                     "embedding": embedding})
        update_store(upsert_local_glossary)
    return wrote
