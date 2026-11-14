"""Streamlit Web UI for the Self-Evaluating Lesson Content Generator."""

import uuid
import hashlib
import time
import streamlit as st
from langsmith.run_helpers import tracing_context

from src.schemas import AgentState, UserProfile
from src.graph import MAX_ATTEMPTS, build_lesson_graph
from src.logger import RunLogger
from src.db import (database_status, ensure_session, finish_run, save_approved_lesson,
                    delete_session, list_sessions, load_session_messages, load_session_artifact,
                    save_session_artifact, save_glossary_entries, save_message,
                    set_lesson_feedback, start_run, sync_profile)
from src.embed import embed_text, embedding_call_limit, embedding_calls_used, reset_embedding_budget
from src.nodes.generate import extract_glossary
from src.followup import answer_followup, NoShippedLessonError
from src.topic_validation import topic_error
from src.knowledge import clean_document_text, ingest_document
from src.memory import (
    load_memory,
    record_human_feedback,
    get_active_lessons,
    record_run_completion,
    list_profiles,
    load_profile,
    create_profile,
    record_feedback_reason,
)
from src import llm

st.set_page_config(
    page_title="Self-Evaluating Lesson Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
    .badge-pass { background-color: #DCFCE7; color: #15803D; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    .badge-fail { background-color: #FEE2E2; color: #B91C1C; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("System Controls")
profiles = list_profiles()
profile_labels = {p.name: p.profile_id for p in profiles}
selected_name = st.sidebar.selectbox("Learner profile", list(profile_labels), key="profile_name")
profile_id = profile_labels[selected_name]
learner_profile = load_profile(profile_id)
profile_signature = learner_profile.model_dump_json()
if st.session_state.get("synced_profile_signature") != profile_signature:
    sync_profile(learner_profile.model_dump())
    st.session_state.synced_profile_signature = profile_signature
mem = load_memory(profile_id)
with st.sidebar.expander("Create learner profile"):
    with st.form("new_profile_form"):
        profile_name = st.text_input("Name", placeholder="e.g. Beginner learner")
        education = st.text_input("Education background", placeholder="e.g. First-year college student")
        expertise = st.selectbox("Expertise in learning needs", ["beginner", "intermediate", "expert"])
        target = st.text_input("Learning needs", value="What do you want to learn?")
        if st.form_submit_button("Create profile") and profile_name.strip():
            new_id = uuid.uuid4().hex[:12]
            create_profile(UserProfile(profile_id=new_id, name=profile_name.strip(),
                                       education_background=education.strip(), expertise_level=expertise,
                                       target_audience=target.strip()))
            st.rerun()

with st.sidebar.expander("Edit selected learner profile"):
    with st.form(f"edit_profile_{profile_id}"):
        edited_name = st.text_input("Profile name", value=learner_profile.name)
        edited_education = st.text_input("Education background", value=learner_profile.education_background)
        expertise_options = ["beginner", "intermediate", "expert"]
        edited_expertise = st.selectbox("Expertise in learning needs", expertise_options,
                                        index=expertise_options.index(learner_profile.expertise_level))
        edited_needs = st.text_area("Learning needs", value=learner_profile.target_audience,
                                    help="Describe what the learner wants to learn and any support they need.")
        if st.form_submit_button("Save profile"):
            create_profile(learner_profile.model_copy(update={
                "name": edited_name.strip() or learner_profile.name,
                "education_background": edited_education.strip(),
                "expertise_level": edited_expertise,
                "target_audience": edited_needs.strip(),
            }))
            st.rerun()

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
    for saved in list_sessions(profile_id):
        sid = saved["session_id"]
        messages = load_session_messages(sid)
        st.session_state.sessions[sid] = {
            "title": saved["title"][:36], "profile_id": profile_id, "messages": messages,
            "latest_run": {}, "source_pack": {},
        }
    if not st.session_state.sessions:
        sid = uuid.uuid4().hex[:12]
        st.session_state.sessions[sid] = {"title": "New lesson", "profile_id": profile_id, "messages": [], "latest_run": {}, "source_pack": {}}
        ensure_session(sid, profile_id)
    st.session_state.active_session = next(iter(st.session_state.sessions))
if st.sidebar.button("＋ New session", width="stretch"):
    sid = uuid.uuid4().hex[:12]
    st.session_state.sessions[sid] = {"title": "New lesson", "profile_id": profile_id, "messages": [], "latest_run": {}}
    ensure_session(sid, profile_id)
    st.session_state.active_session = sid
    st.session_state.active_session_picker = sid
session_ids = [sid for sid, session in st.session_state.sessions.items() if session.get("profile_id") == profile_id]
if not session_ids:
    for saved in list_sessions(profile_id):
        sid = saved["session_id"]
        st.session_state.sessions[sid] = {
            "title": saved["title"][:36], "profile_id": profile_id,
            "messages": load_session_messages(sid), "latest_run": {}, "source_pack": {},
        }
    session_ids = [sid for sid, session in st.session_state.sessions.items() if session.get("profile_id") == profile_id]
if not session_ids:
    sid = uuid.uuid4().hex[:12]
    st.session_state.sessions[sid] = {"title": "New lesson", "profile_id": profile_id, "messages": [], "latest_run": {}, "source_pack": {}}
    ensure_session(sid, profile_id)
    st.session_state.active_session = sid
    st.session_state.active_session_picker = sid
    session_ids = [sid]
if st.session_state.get("active_session") not in session_ids:
    st.session_state.active_session = session_ids[0]
    st.session_state.active_session_picker = session_ids[0]
session_labels = {sid: st.session_state.sessions[sid]["title"] for sid in session_ids}
# Keep the widget's selected value in one place. Supplying both a Session State
# value and `index` makes Streamlit warn about conflicting defaults.
if st.session_state.get("active_session_picker") not in session_ids:
    st.session_state["active_session_picker"] = st.session_state.active_session
selected_session = st.sidebar.selectbox("Session", session_ids,
    format_func=lambda sid: session_labels[sid],
    key="active_session_picker")
st.session_state.active_session = selected_session
if st.sidebar.button("Delete selected session", type="secondary", key=f"delete_session_{selected_session}"):
    del st.session_state.sessions[selected_session]
    delete_session(selected_session)
    remaining = [sid for sid, session in st.session_state.sessions.items() if session.get("profile_id") == profile_id]
    if not remaining:
        sid = uuid.uuid4().hex[:12]
        st.session_state.sessions[sid] = {"title": "New lesson", "profile_id": profile_id, "messages": [], "latest_run": {}}
        ensure_session(sid, profile_id)
        remaining = [sid]
    st.session_state.active_session = remaining[0]
    st.session_state.active_session_picker = remaining[0]
    st.rerun()

st.sidebar.markdown("### Target Learner")
st.sidebar.info(
    f"**Education**: {learner_profile.education_background}\n"
    f"**Experience level**: {learner_profile.expertise_level.title()}\n"
    f"**Learning needs**: {learner_profile.target_audience}\n"
    f"**Sentence limit**: {learner_profile.sentence_len_max} words"
)

st.sidebar.markdown("### System Memory")
col_m1, col_m2 = st.sidebar.columns(2)
col_m1.metric("Total Runs", mem.total_runs)
col_m2.metric("Success Rate", f"{(mem.successful_runs / max(mem.total_runs, 1) * 100):.0f}%")

col_t1, col_t2 = st.sidebar.columns(2)
col_t1.metric("👍 Thumbs Up", mem.thumbs_up)
col_t2.metric("👎 Thumbs Down", mem.thumbs_down)

db_health_cache = st.session_state.get("database_status_cache")
if not db_health_cache or time.monotonic() - db_health_cache["checked_at"] >= 15:
    db_health_cache = {"checked_at": time.monotonic(), "status": database_status()}
    st.session_state.database_status_cache = db_health_cache
db_health = db_health_cache["status"]
if not db_health["available"]:
    detail = f" ({db_health['reason']})" if db_health.get("reason") else ""
    st.sidebar.warning(f"Semantic database unavailable{detail}. Sessions and audit data use local JSON/JSONL fallback.")

if llm.DEFAULT_JUDGE_MODEL and llm.DEFAULT_JUDGE_MODEL == llm.DEFAULT_GEN_MODEL:
    st.sidebar.warning("Judge and draft currently use the same model. Set a distinct JUDGE_MODEL for independent review.")

st.sidebar.markdown("### Active Learned Lessons")
active_lessons = get_active_lessons(profile_id=profile_id, topic=st.session_state.sessions[selected_session].get("topic", ""), max_lessons=3)
for lesson in active_lessons:
    st.sidebar.caption(f"• {lesson}")

with st.sidebar.expander("Persistent Knowledge Library"):
    st.caption("Documents are chunked and saved to this learner profile across sessions.")
    uploaded_files = st.file_uploader(
        "Add source documents", type=["txt", "md", "html", "pdf"], accept_multiple_files=True,
        help="Text, Markdown, HTML, and PDF files. Maximum 5 MB per file.",
    )
    if uploaded_files:
        seen_uploads = st.session_state.setdefault("ingested_upload_hashes", set())
        for uploaded in uploaded_files:
            content_bytes = uploaded.getvalue()
            upload_hash = hashlib.sha256(profile_id.encode() + uploaded.name.encode() + content_bytes).hexdigest()
            if upload_hash in seen_uploads:
                continue
            if len(content_bytes) > 5 * 1024 * 1024:
                st.error(f"{uploaded.name}: file exceeds the 5 MB limit.")
                seen_uploads.add(upload_hash)
                continue
            ingest_logger = RunLogger()
            ensure_session(selected_session, profile_id)
            start_run(ingest_logger.run_id, selected_session,
                      f"Document ingestion: {uploaded.name}", update_title=False)
            ingest_logger.start_step("knowledge_ingest")
            try:
                text = clean_document_text(uploaded.name, content_bytes)
                reset_embedding_budget()
                result = ingest_document(profile_id, uploaded.name, text)
                status = "success" if result["chunks_saved"] else "empty"
                ingest_logger.log_event(
                    step="knowledge_ingest", attempt=1, status=status,
                    details={"profile_id": profile_id, "filename": uploaded.name,
                             "chunks_saved": result["chunks_saved"],
                             "chunks_embedded": result["chunks_embedded"],
                             "chunks_total": result["chunks_total"]},
                )
                finish_run(ingest_logger.run_id, status, 1,
                           {"embedding_requests": embedding_calls_used()})
                if result["chunks_embedded"]:
                    st.success(f"Saved {uploaded.name}: {result['chunks_saved']} chunks, semantically searchable across sessions.")
                elif result["chunks_saved"]:
                    st.warning(f"Saved {uploaded.name} as text, but embedding was unavailable. It will not enter similarity retrieval until embeddings are available.")
                else:
                    st.warning(f"No readable text found in {uploaded.name}.")
                seen_uploads.add(upload_hash)
            except Exception as exc:
                ingest_logger.log_event(step="knowledge_ingest", attempt=1, status="error",
                                        details={"filename": uploaded.name,
                                                 "error_type": type(exc).__name__,
                                                 "error_message": str(exc)[:240]})
                finish_run(ingest_logger.run_id, "failed", 1, {})
                seen_uploads.add(upload_hash)
                st.error(f"Could not ingest {uploaded.name} ({type(exc).__name__}): {str(exc)[:180]}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Self-Healing Test")
inject_mode = st.sidebar.selectbox(
    "Inject Synthetic Error (Test Retries)",
    options=["None (Normal)", "Long Sentence (> 25 words)", "Missing Required Section", "Factual error + unexplained jargon"],
    help="Inject one controlled failure into the first draft to demonstrate evaluation and revision.",
)

inject_error_key = None
if inject_mode == "Long Sentence (> 25 words)":
    inject_error_key = "long_sentence"
elif inject_mode == "Missing Required Section":
    inject_error_key = "missing_section"
elif inject_mode == "Factual error + unexplained jargon":
    inject_error_key = "factual_jargon"

# Main Layout
st.markdown('<div class="main-header">🎓 Self-Evaluating Lesson Generator</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Adaptive lessons for {learner_profile.expertise_level} learners, grounded in retrieved sources.</div>',
    unsafe_allow_html=True,
)

topic_input = st.text_input(
    "Enter Lesson Topic",
    value="",
    placeholder="e.g. How does machine learning work?",
    help="Enter a question or topic to start a new lesson.",
)
topic_error_message = topic_error(topic_input)
if topic_input.strip() and topic_error_message:
    st.error(topic_error_message)

active_session = st.session_state.sessions[selected_session]
if "latest_artifact" not in active_session:
    active_session["latest_artifact"] = load_session_artifact(selected_session)
latest_for_session = active_session.get("latest_run") or {}
latest_state = latest_for_session.get("final_state") or {}
restored_artifact = active_session.get("latest_artifact") or {}
has_current_run = bool(latest_for_session)
followup_lesson = ((latest_state.get("draft") or {}).get("content", "") if has_current_run
                   else restored_artifact.get("lesson_content", ""))
followup_status = (latest_state.get("final_status") if has_current_run
                   else restored_artifact.get("status"))
followup_enabled = bool(
    followup_status == "passed" and followup_lesson.strip()
    and (latest_for_session.get("is_success", True) if has_current_run else True)
)
if not active_session.get("source_pack") and restored_artifact.get("status") == "passed":
    active_session["source_pack"] = restored_artifact.get("source_pack", {})

for message in active_session["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.markdown(message["content"])

if not followup_enabled:
    st.info("Follow-ups unlock after this session has a lesson that passes every quality check. Generate or retry a lesson first.")
followup = st.chat_input("Ask a follow-up about this session's lesson", disabled=not followup_enabled)
if followup:
    session = st.session_state.sessions[selected_session]
    history_start = int(restored_artifact.get(
        "conversation_start_index", max(0, len(session["messages"]) - 8)
    ))
    conversation_history = session["messages"][history_start:][-8:]
    session["messages"].append({"role": "user", "content": followup})
    save_message(selected_session, "user", followup)
    followup_logger = RunLogger()
    parent_lesson_run = latest_for_session.get("run_id") or restored_artifact.get("run_id", "")
    start_run(followup_logger.run_id, selected_session, f"Follow-up: {followup[:80]}", update_title=False)

    def followup_audit(event):
        followup_logger.log_event(step="llm_call", attempt=1, status=event.get("status", "unknown"),
                                  model=event.get("model"), details={
                                      **event, "session_id": selected_session,
                                      "parent_lesson_run_id": parent_lesson_run,
                                  })

    llm.reset_usage_stats()
    followup_status = "passed"
    followup_checks = []
    try:
        with tracing_context(
            tags=["lesson-followup", "self-evaluating-agent"],
            metadata={"session_id": selected_session, "parent_lesson_run_id": parent_lesson_run,
                      "followup_run_id": followup_logger.run_id},
        ):
            with st.spinner("Answering from the shipped lesson and its saved sources…"):
                result = answer_followup(followup, followup_lesson, session.get("source_pack", {}),
                                         learner_profile.model_dump(), audit_callback=followup_audit,
                                         conversation_history=conversation_history)
        answer = result["answer"]
        followup_checks = [check.model_dump() if hasattr(check, "model_dump") else check.__dict__
                           for check in result.get("checks", [])]
        if not result["passed"]:
            followup_status = "readability_warning"
            st.warning("The answer still misses a readability check after one simplification. Review it before relying on it.")
    except NoShippedLessonError as exc:
        answer = str(exc)
        followup_status = "blocked"
    except Exception as exc:
        answer = f"Follow-up could not be completed ({type(exc).__name__}): {str(exc)[:200]} Check provider connectivity and the audit log."
        followup_status = "error"
    session["messages"].append({"role": "assistant", "content": answer})
    save_message(selected_session, "assistant", answer)
    followup_logger.log_event(
        step="followup", attempt=1, status=followup_status,
        details={"session_id": selected_session, "parent_lesson_run_id": parent_lesson_run,
                 "check_results": followup_checks, "model_calls": llm.USAGE_STATS.get("total_calls", 0)},
    )
    finish_run(followup_logger.run_id, followup_status, 1, dict(llm.USAGE_STATS))
    session.setdefault("followup_runs", []).append({
        "run_id": followup_logger.run_id, "parent_lesson_run_id": parent_lesson_run,
        "status": followup_status, "events": followup_logger.read_all_events(),
    })
    st.rerun()

run_button = st.button("🚀 Generate & Self-Evaluate Lesson", type="primary", width="stretch",
                       disabled=bool(topic_error_message))

if run_button:
    llm.reset_usage_stats()
    logger = RunLogger()
    ensure_session(selected_session, profile_id)
    start_run(logger.run_id, selected_session, topic_input)
    save_session_artifact(selected_session, {"run_id": logger.run_id, "status": "in_progress"})

    initial_state: AgentState = {
        "topic": topic_input,
        "learner_profile": learner_profile.model_dump(),
        "profile_id": profile_id,
        "session_id": selected_session,
        "session_source_pack": st.session_state.sessions[selected_session].get("source_pack", {}),
        "attempt": 1,
        "rejection_log": [],
        "memory_lessons": active_lessons,
        "inject_error": inject_error_key,
        "final_status": "in_progress",
        "semantic_context": {},
    }

    streamed_draft = st.empty()
    token_buffer = []
    active_stream_attempt = [None]
    def render_token(attempt, token):
        if token == "\x00RESET\x00":
            active_stream_attempt[0] = attempt
            token_buffer.clear()
            streamed_draft.empty()
            return
        if active_stream_attempt[0] != attempt:
            active_stream_attempt[0] = attempt
            token_buffer.clear()
            st.write(f"✍️ Writing lesson draft · attempt {attempt}")
        token_buffer.append(token)
        streamed_draft.markdown(f"**Live draft · attempt {attempt}**\n\n" + "".join(token_buffer))

    graph = build_lesson_graph(logger, stream_callback=render_token)

    # LangSmith tracing configuration metadata & tags
    langsmith_config = {
        "run_name": f"lesson_{topic_input.replace(' ', '_').lower()}",
        "metadata": {
            "agent_id": "your-comp",
            "environment": "local",
            "topic": topic_input,
            "run_id": logger.run_id,
            "profile_id": profile_id,
            "session_id": selected_session,
        },
        "tags": ["your-comp", "local", "self-evaluating-agent"],
    }

    status_box = st.status("Executing Quality-Control State Machine...", expanded=True)

    if llm.DEFAULT_JUDGE_MODEL and llm.DEFAULT_JUDGE_MODEL == llm.DEFAULT_GEN_MODEL:
        logger.log_event(step="configuration", attempt=1, status="warning",
                         model=llm.DEFAULT_GEN_MODEL,
                         details={"failure_reason": "Judge and draft share the same model; review is not independent."})

    with status_box:
        st.write("🔍 Gathering sources from the session cache, topic cache, Tavily, or fallback search...")

        # Stream node completions so users see progress while the pipeline runs.
        final_state = dict(initial_state)
        pipeline_error = None
        step_names = {
            "context": "🧠 Checking related learning context",
            "search": "🔍 Source retrieval complete",
            "plan": "📝 Lesson plan ready",
            "generate": "✍️ Draft complete",
            "eval_native": "🛡️ Native quality gate complete",
            "eval_judge": "⚖️ Content review complete",
            "record_retry": "🔄 Retrying generation with the same plan",
        }
        try:
            for update in graph.stream(initial_state, config=langsmith_config, stream_mode="updates"):
                for node, values in update.items():
                    if isinstance(values, dict):
                        final_state.update(values)
                    if node in step_names:
                        if node == "record_retry":
                            streamed_draft.empty()
                        attempt_label = final_state.get("attempt", 1)
                        label = step_names[node]
                        if node in {"eval_native", "eval_judge"}:
                            passed_key = "native_eval_passed" if node == "eval_native" else "judge_eval_passed"
                            label = ("✅ " if final_state.get(passed_key) else "⚠️ ") + label
                        st.write(f"{label} · attempt {attempt_label}" if node in {"plan", "generate", "eval_native", "eval_judge"} else label)
        except Exception as exc:
            error_text = str(exc).casefold()
            if "rate_limit" in error_text:
                pipeline_error = "The model provider rate limit was reached. Wait, then retry."
            elif "timeout" in error_text:
                pipeline_error = "The model provider timed out before completing the response."
            elif "connectivity" in error_text or "connection" in error_text or "dns" in error_text:
                pipeline_error = "The model provider could not be reached. Check network connectivity."
            elif "truncated" in error_text or "finish_reason=length" in error_text:
                pipeline_error = "The model response was truncated at the token limit. Increase DRAFT_MAX_TOKENS or reduce prompt size."
            elif "auth" in error_text or "401" in error_text or "403" in error_text:
                pipeline_error = "The model provider rejected authentication. Check the configured API key."
            elif "zero completion choices" in error_text or "empty message" in error_text:
                pipeline_error = "The model provider returned an empty or malformed response. See the per-call audit details."
            elif "llm provider call failed" in error_text or "openrouter llm call failed" in error_text:
                pipeline_error = f"The model provider request failed: {str(exc)[:240]}"
            elif "api_key is not set" in error_text:
                pipeline_error = "The selected model provider API key is missing from .env."
            elif isinstance(exc, ValueError):
                pipeline_error = f"The pipeline received invalid model or lesson data: {str(exc)[:240]}"
            else:
                pipeline_error = f"Pipeline stopped ({type(exc).__name__}): {str(exc)[:240]}"
            logger.log_event(step="pipeline", attempt=final_state.get("attempt", 1),
                             status="error", details={"error_type": type(exc).__name__,
                                                      "error_message": str(exc)[:500],
                                                      "failure_reason": pipeline_error})
            final_state["final_status"] = "failed"
            final_state.setdefault("draft", {"title": "", "content": "", "word_count": 0})

        # Check final status
        is_success = (not pipeline_error and final_state.get("native_eval_passed", False)
                      and final_state.get("judge_eval_passed", False))
        final_state["final_status"] = "passed" if is_success else "failed"
        record_run_completion(is_success, profile_id=profile_id)
        usage = dict(llm.USAGE_STATS)
        usage["embedding_calls"] = embedding_calls_used()
        usage["embedding_call_limit"] = embedding_call_limit()
        finish_run(logger.run_id, final_state["final_status"], final_state.get("attempt", 1), usage)
        shipped_content = (final_state.get("draft") or {}).get("content", "") if is_success else ""
        session_artifact = {
            "run_id": logger.run_id, "status": final_state["final_status"],
            "topic": topic_input, "lesson_content": shipped_content,
            "source_pack": final_state.get("source_pack", {}) if is_success else {},
            "conversation_start_index": len(st.session_state.sessions[selected_session]["messages"]) + 2,
        }
        save_session_artifact(selected_session, session_artifact)

        attempts_taken = final_state.get("attempt", 1)
        revisions = max(0, attempts_taken - 1)
        st.write(f"🔄 **Drafts**: {attempts_taken} of {MAX_ATTEMPTS} (1 initial + {revisions} revision{'s' if revisions != 1 else ''})")

        if pipeline_error:
            status_box.update(label="⚠️ Generation stopped before completion.", state="error", expanded=True)
            st.error(f"{pipeline_error} Check the audit log for the failed step, then retry.")
        elif final_state.get("judge_unavailable"):
            status_box.update(label="⚠️ The judge could not evaluate this lesson.", state="error", expanded=True)
            st.warning("The judge did not return every required check. The run stopped as an evaluator error; no missing check was assumed to pass.")
        elif is_success:
            status_box.update(label="✅ Quality Verification Passed! Lesson Ready to Ship.", state="complete", expanded=False)
        else:
            status_box.update(label="⚠️ Terminal Rejection: Maximum retries reached.", state="error", expanded=True)

    # Store results in session state for UI persistence
    st.session_state["latest_run"] = {
        "final_state": final_state,
        "run_id": logger.run_id,
        "is_success": is_success,
        "error": pipeline_error,
        "events": logger.read_all_events(),
        "usage": usage,
    }
    session = st.session_state.sessions[selected_session]
    session["topic"] = topic_input
    session["title"] = topic_input[:36]
    session["source_pack"] = final_state.get("source_pack", {})
    session["latest_artifact"] = session_artifact
    session["messages"].append({"role": "user", "content": topic_input})
    session["messages"].append({"role": "assistant", "content": (final_state.get("draft") or {}).get("content", "")})
    save_message(selected_session, "user", topic_input)
    save_message(selected_session, "assistant", (final_state.get("draft") or {}).get("content", ""))
    session["latest_run"] = st.session_state["latest_run"]

if st.session_state.sessions[selected_session].get("latest_run"):
    run_data = st.session_state.sessions[selected_session]["latest_run"]
    state = run_data["final_state"]
    draft = state.get("draft") or {}
    content = draft.get("content", "")
    rejection_log = state.get("rejection_log", [])
    native_results = state.get("native_results", [])
    judge_results = state.get("judge_results", [])
    source_pack = state.get("source_pack", {})

    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Quality Status", "PASSED ✅" if run_data["is_success"] else "FAILED ❌")
    col2.metric("Drafts (incl. revisions)", f"{state.get('attempt', 1)} / {MAX_ATTEMPTS}")
    col3.metric("Search Source", source_pack.get("source_tool", "tavily").upper())
    col4.metric("LLM Calls Used", run_data["usage"].get("total_calls", 0))
    col5.metric("Embedding Calls", f"{run_data['usage'].get('embedding_calls', 0)}/{embedding_call_limit()}")

    tab_lesson, tab_rejection, tab_rubric, tab_audit = st.tabs([
        "📖 Final Lesson Output",
        f"📋 Rejection Log ({len(rejection_log)})",
        "🎯 Rubric Evaluation Scorecard",
        "🔍 Audit Events & Token Stats",
    ])

    with tab_lesson:
        st.markdown(content if content else "*No lesson text available.*")

        sources = source_pack.get("sources", [])
        retrieval_details = source_pack.get("retrieval_details", {})
        st.caption(f"Search served by: {source_pack.get('source_tool', 'unknown')}. {retrieval_details}")
        with st.expander(f"Grounding sources ({len(sources)})"):
            for source in sources:
                if source.get("url"):
                    st.markdown(f"- [{source.get('title', 'Source')}]({source['url']})")
                else:
                    st.markdown(f"- {source.get('title', 'Source')} *(uploaded learning material)*")
                excerpt = source.get("raw_content") or source.get("snippet", "")
                if excerpt:
                    st.caption(excerpt[:700])

        st.markdown("---")
        st.markdown("#### Learner Feedback")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 6])
        if col_fb1.button("👍 Good Lesson", key="thumb_up_btn", disabled=not run_data["is_success"]):
            record_human_feedback(True, profile_id=profile_id)
            topic = state.get("topic", "")
            embedding = embed_text(topic)
            save_approved_lesson(run_data["run_id"], topic, embedding, content,
                                 {"native": native_results, "judge": judge_results})
            glossary = extract_glossary(content)
            for item in glossary[:2]:
                item["embedding"] = embed_text(f"{item['term']}: {item['definition']}")
            save_glossary_entries(profile_id, topic, glossary)
            set_lesson_feedback(run_data["run_id"], True)
            st.success("Feedback recorded. Approved lesson and glossary were saved to the available persistence store.")
            st.rerun()
        down_reason = col_fb3.text_input("One-line reason (optional)", key=f"down_reason_{run_data['run_id']}")
        if col_fb2.button("👎 Needs Work", key="thumb_down_btn"):
            record_human_feedback(False, profile_id=profile_id)
            record_feedback_reason(down_reason or "Learner marked this lesson as needing work.", profile_id=profile_id)
            set_lesson_feedback(run_data["run_id"], False)
            st.warning("Feedback recorded. No glossary was learned from this lesson.")
            st.rerun()

    with tab_rejection:
        if not rejection_log:
            st.success("🎉 Passed on the first attempt! No rejections occurred.")
        else:
            st.warning(f"The autonomous self-evaluator caught and corrected {len(rejection_log)} rejection(s):")
            for idx, entry in enumerate(rejection_log, start=1):
                with st.expander(f"Attempt {entry.get('attempt')} — Failed in {entry.get('stage_failed', '').upper()} Gate", expanded=True):
                    st.markdown(f"**Failed Checks**: `{', '.join(entry.get('failed_check_ids', []))}`")
                    st.markdown("**Reasons:**")
                    for r in entry.get("reasons", []):
                        st.markdown(f"- {r}")
                    st.info(f"**Action Taken on Retry**: {entry.get('suggested_fix', '')}")
                    revised = entry.get("revised_sections", [])
                    if revised:
                        st.caption("Sections targeted for revision: " + ", ".join(revised))
                    for evidence in entry.get("source_evidence", []):
                        st.markdown(f"**Source evidence — [{evidence.get('title', 'Source')}]({evidence.get('url', '')}):** {evidence.get('excerpt', '')}")
                    if entry.get("revision_diff"):
                        with st.expander("Revision diff"):
                            st.code(entry["revision_diff"], language="diff")

    with tab_rubric:
        st.markdown("### Tier 1: Deterministic Native Checks (Zero-LLM)")
        if not native_results:
            st.info("Native checks not available.")
        else:
            for r in native_results:
                col_icon, col_desc = st.columns([1, 8])
                if r.get("passed"):
                    col_icon.markdown('<span class="badge-pass">PASS</span>', unsafe_allow_html=True)
                else:
                    col_icon.markdown('<span class="badge-fail">FAIL</span>', unsafe_allow_html=True)
                col_desc.markdown(f"**{r.get('name')}** (`{r.get('check_id')}`): {r.get('message')}")

        st.markdown("### Tier 2: LLM-as-a-Judge Binary Checkpoints")
        if not judge_results:
            st.info("Judge checks were skipped (either Native checks failed or first attempt rejected).")
        else:
            for r in judge_results:
                col_icon, col_desc = st.columns([1, 8])
                if r.get("passed"):
                    col_icon.markdown('<span class="badge-pass">PASS</span>', unsafe_allow_html=True)
                else:
                    col_icon.markdown('<span class="badge-fail">FAIL</span>', unsafe_allow_html=True)
                evidence = f" — *Evidence: \"{r.get('evidence_quote')}\"*" if r.get("evidence_quote") else ""
                col_desc.markdown(f"**{r.get('name')}** (`{r.get('check_id')}`): {r.get('reason')}{evidence}")

    with tab_audit:
        st.markdown(f"**Session Run ID**: `{run_data['run_id']}`")
        st.markdown(f"**Model Used**: `{run_data['usage'].get('last_model_used', 'openrouter/free')}`")
        col_tok1, col_tok2 = st.columns(2)
        col_tok1.metric("Prompt Tokens", run_data["usage"].get("total_prompt_tokens", 0))
        col_tok2.metric("Completion Tokens", run_data["usage"].get("total_completion_tokens", 0))
        stage_stats = run_data["usage"].get("calls_by_stage", {})
        if stage_stats:
            st.markdown("#### Model Call Cost by Stage")
            st.dataframe([
                {"Stage": stage, "Calls": values.get("calls", 0),
                 "Latency (ms)": values.get("latency_ms", 0),
                 "Prompt tokens": values.get("prompt_tokens", 0),
                 "Completion tokens": values.get("completion_tokens", 0)}
                for stage, values in stage_stats.items()
            ], width="stretch", hide_index=True)
        if run_data["usage"].get("usage_incomplete"):
            st.caption("Token counts are partial because the provider did not return usage for streamed generation.")

        st.markdown("#### Chronological Step Events")
        events = run_data.get("events", [])
        if events:
            st.dataframe(events, width="stretch")
        else:
            st.caption("No events recorded.")
        followup_runs = st.session_state.sessions[selected_session].get("followup_runs", [])
        if followup_runs:
            st.markdown("#### Follow-up traces")
            for followup_run in followup_runs:
                st.markdown(
                    f"Follow-up run `{followup_run['run_id']}` · parent lesson run "
                    f"`{followup_run.get('parent_lesson_run_id') or 'unknown'}` · "
                    f"status: **{followup_run.get('status')}**"
                )
                if followup_run.get("events"):
                    st.dataframe(followup_run["events"], width="stretch", hide_index=True)
