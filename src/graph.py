"""LangGraph state machine wiring for the Self-Evaluating Lesson Content Generator."""

import os
import difflib
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from src.schemas import (
    AgentState,
    SourcePack,
    LessonPlan,
    DraftLesson,
    RejectionLogEntry,
)
from src.nodes.search import fetch_source_pack
from src.nodes.context import retrieve_semantic_context
from src.embed import embed_text, embed_texts
from src.knowledge import chunk_text
from src.db import find_related_knowledge, save_knowledge_chunks
from src.nodes.plan import plan_lesson
from src.nodes.generate import generate_lesson_draft
from src.nodes.eval_native import evaluate_native_all
from src.nodes.eval_judge import evaluate_with_judge
from src.memory import record_failures, get_active_lessons
from src.db import save_failure_lessons
from src.logger import RunLogger
from src import llm

MAX_ATTEMPTS = max(1, int(os.getenv("MAX_ATTEMPTS", "3")))  # Total attempts, including the first.


def _model_audit(logger: RunLogger):
    def record(event: Dict[str, Any]) -> None:
        logger.log_event(
            step="llm_call", attempt=int(event.get("attempt", 1)),
            status=event.get("status", "unknown"), model=event.get("model"),
            details=event,
        )
    return record


def context_node(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Retrieve optional related material; it never influences evaluation routing."""
    attempt = state.get("attempt", 1)
    logger.start_step("context")
    context = retrieve_semantic_context(state.get("topic", ""), state.get("profile_id", "default"))
    logger.log_event(
        step="context", attempt=attempt, status=context["status"],
        details={"threshold": context.get("threshold"), "has_embedding": context.get("has_embedding", False),
                 "memory_lessons": len(context.get("memory_lessons", [])),
                 "has_source_pack": bool(context.get("source_pack")),
                 "similarity_hits": context.get("similarity_hits", []),
                 "skip_reason": context.get("skip_reason")},
    )
    memory_lessons = list(state.get("memory_lessons", []))
    for lesson in context.get("memory_lessons", []):
        if lesson not in memory_lessons:
            memory_lessons.append(lesson)
    return {"semantic_context": context, "memory_lessons": memory_lessons[:3]}


def search_node(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Node 1: Grounding search fallback pipeline (Tavily -> DDGS -> Cache)."""
    topic = state.get("topic", "")
    attempt = state.get("attempt", 1)
    logger.start_step("search")

    semantic_context = state.get("semantic_context", {})
    source_pack: SourcePack = fetch_source_pack(
        topic, session_sources=state.get("session_source_pack"),
        semantic_source_pack=semantic_context.get("source_pack"),
    )

    # Persist site content as overlapping chunks, then retrieve only chunks
    # semantically related to this topic. Uploaded documents use the same store.
    profile_id = state.get("profile_id", "default")
    query_embedding = embed_text(topic)  # Normally served by the context-stage cache.
    site_entries = []
    site_texts = []
    for source in source_pack.sources:
        source_text = source.raw_content or source.snippet
        for index, chunk in enumerate(chunk_text(source_text, max_chunks=4), start=1):
            site_texts.append(chunk)
            site_entries.append({
                "profile_id": profile_id, "source_name": source.title,
                "source_type": "website", "source_url": source.url,
                "content": chunk,
            })
    if site_entries and query_embedding:
        for entry, vector in zip(site_entries, embed_texts(site_texts)):
            entry["embedding"] = vector
        save_knowledge_chunks(site_entries)

    related_chunks = (find_related_knowledge(
        profile_id, query_embedding, float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.60")), limit=6
    ) if query_embedding else [])
    if related_chunks:
        chunk_sources = [SourceItem(
            title=f"{item['source_name']} (related excerpt)", url=item.get("source_url", ""),
            snippet=item["content"], raw_content=item["content"], score=item["similarity"],
        ) for item in related_chunks]
        seen = {(item.title, item.url, item.snippet[:120]) for item in chunk_sources}
        originals = [item for item in source_pack.sources
                     if (item.title, item.url, item.snippet[:120]) not in seen]
        source_pack.sources = chunk_sources + originals
    source_pack.retrieval_details["semantic_chunks"] = [
        {"source": item["source_name"], "type": item["source_type"],
         "similarity": round(item["similarity"], 4)} for item in related_chunks
    ]
    pack_dict = source_pack.model_dump()

    logger.log_event(
        step="search",
        attempt=attempt,
        status="success",
        tool=source_pack.source_tool,
        details={
            "sources_count": len(source_pack.sources),
            "semantic_chunks": source_pack.retrieval_details.get("semantic_chunks", []),
            "source_tool": source_pack.source_tool,
            "search_details": source_pack.model_dump().get("retrieval_details", {}),
            "sources": [
                {"title": item.title, "url": item.url, "score": item.score,
                 "snippet_chars": len(item.raw_content or item.snippet)}
                for item in source_pack.sources
            ],
        },
    )

    return {"source_pack": pack_dict}


def plan_node(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Node 2: Lesson planning with pedagogical outline and memory lessons."""
    topic = state.get("topic", "")
    attempt = state.get("attempt", 1)
    source_pack = state.get("source_pack", {})
    profile_id = state.get("profile_id", "default")
    memory_lessons = state.get("memory_lessons", []) or get_active_lessons(profile_id, topic, max_lessons=3)

    logger.start_step("plan")
    plan: LessonPlan = plan_lesson(topic=topic, source_pack=source_pack, memory_lessons=memory_lessons,
                                   learner_profile=state.get("learner_profile"),
                                   semantic_context=(state.get("semantic_context") or {}).get("prompt_block", ""),
                                   run_attempt=attempt, audit_callback=_model_audit(logger))
    plan_dict = plan.model_dump()

    logger.log_event(
        step="plan",
        attempt=attempt,
        status="success",
        details={"core_sections": plan.core_sections, "must_cover_points": plan.must_cover_points,
                 "analogies": plan.key_analogies},
    )

    return {"plan": plan_dict, "memory_lessons": memory_lessons}


def generate_node(state: AgentState, logger: RunLogger, stream_callback=None) -> Dict[str, Any]:
    """Node 3: Markdown draft synthesis from plan and grounded sources."""
    topic = (state.get("plan") or {}).get("topic") or state.get("topic", "")
    attempt = state.get("attempt", 1)
    plan = state.get("plan", {})
    source_pack = state.get("source_pack", {})
    memory_lessons = state.get("memory_lessons", [])
    rejection_log = state.get("rejection_log", [])
    inject_error = state.get("inject_error") if attempt == 1 else None

    logger.start_step("generate")
    draft: DraftLesson = generate_lesson_draft(
        topic=topic,
        plan=plan,
        source_pack=source_pack,
        memory_lessons=memory_lessons,
        rejection_feedback=rejection_log,
        inject_error=inject_error,
        stream_callback=(lambda token: stream_callback(attempt, token)) if stream_callback else None,
        learner_profile=state.get("learner_profile"),
        semantic_context=(state.get("semantic_context") or {}).get("prompt_block", ""),
        revision_sections=state.get("revision_sections", []) if attempt > 1 else [],
        revision_source_content=state.get("revision_source_content", "") if attempt > 1 else "",
        run_attempt=attempt, audit_callback=_model_audit(logger),
    )
    draft_dict = draft.model_dump()

    logger.log_event(
        step="generate",
        attempt=attempt,
        status="success",
        details={"word_count": draft.word_count, "content_chars": len(draft.content), "title": draft.title},
    )

    updated_rejections = list(rejection_log)
    if attempt > 1 and updated_rejections:
        previous = state.get("revision_source_content", "").splitlines(keepends=True)
        current = draft.content.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(previous, current,
                                             fromfile=f"attempt-{attempt - 1}",
                                             tofile=f"attempt-{attempt}", n=2))[:6000]
        updated_rejections[-1]["revision_diff"] = diff or "No textual change detected."
        logger.log_event(step="revision", attempt=attempt, status="success",
                         details={"sections": state.get("revision_sections", []),
                                  "diff": updated_rejections[-1]["revision_diff"]})
    return {"draft": draft_dict, "rejection_log": updated_rejections}


def eval_native_node(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Node 4: Tier-1 deterministic evaluation gate (regex, textstat, jargon)."""
    attempt = state.get("attempt", 1)
    draft = state.get("draft", {})
    content = draft.get("content", "")
    source_pack = state.get("source_pack", {})

    logger.start_step("eval_native")
    plan = state.get("plan") or {}
    passed, results = evaluate_native_all(content, source_pack=source_pack,
                                          required_headers=plan.get("core_sections", []),
                                          profile=state.get("learner_profile"))

    serialized_results = [r.model_dump() for r in results]
    failed_names = [r.check_id for r in results if not r.passed]

    logger.log_event(
        step="eval_native",
        attempt=attempt,
        status="passed" if passed else "failed",
        details={
            "failed_checks": failed_names,
            "total_checks": len(results),
            "results": [
                {"check_id": r.check_id, "passed": r.passed, "message": r.message,
                 "violations": r.violations[:3]}
                for r in results
            ],
        },
    )

    return {
        "native_eval_passed": passed,
        "native_results": serialized_results,
    }


def eval_judge_node(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Node 5: Tier-2 LLM-as-a-judge binary checkpoints."""
    topic = (state.get("plan") or {}).get("topic") or state.get("topic", "")
    attempt = state.get("attempt", 1)
    draft = state.get("draft", {})
    content = draft.get("content", "")
    source_pack = state.get("source_pack", {})

    logger.start_step("eval_judge")
    plan = state.get("plan") or {}
    passed, results = evaluate_with_judge(topic=topic, draft_text=content, source_pack=source_pack,
                                          must_cover_points=plan.get("must_cover_points", []),
                                          learner_profile=state.get("learner_profile"),
                                          run_attempt=attempt, audit_callback=_model_audit(logger))

    serialized_results = [r.model_dump() for r in results]
    failed_names = [r.check_id for r in results if not r.passed]
    judge_unavailable = any(r.judge_error for r in results)

    logger.log_event(
        step="eval_judge",
        attempt=attempt,
        status="passed" if passed else "failed",
        details={
            "failed_checks": failed_names,
            "judge_unavailable": judge_unavailable,
            "total_checks": len(results),
            "results": [
                {"check_id": r.check_id, "passed": r.passed, "reason": r.reason,
                 "evidence_quote": r.evidence_quote, "source_quote": r.source_quote}
                for r in results
            ],
        },
    )

    return {
        "judge_eval_passed": passed,
        "judge_results": serialized_results,
        "judge_unavailable": judge_unavailable,
    }


def routing_decision(state: AgentState) -> Literal["eval_judge", "retry_generate", "ship_success", "ship_failure"]:
    """Conditional router deciding next step in the quality-control loop."""
    native_passed = state.get("native_eval_passed", False)
    attempt = state.get("attempt", 1)

    # If native failed, skip judge immediately to save LLM tokens/calls
    if not native_passed:
        source_tool = (state.get("source_pack") or {}).get("source_tool", "")
        if source_tool == "offline_stub" and attempt >= MAX_ATTEMPTS:
            return "ship_failure"
        if attempt < MAX_ATTEMPTS:
            return "retry_generate"
        else:
            return "ship_failure"

    # If native passed, check judge
    judge_passed = state.get("judge_eval_passed")
    if judge_passed is None:
        # Judge hasn't run yet
        return "eval_judge"

    if judge_passed:
        return "ship_success"
    else:
        if state.get("judge_unavailable", False):
            return "ship_failure"
        if attempt < MAX_ATTEMPTS:
            return "retry_generate"
        else:
            return "ship_failure"


def record_retry_and_rejection(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Record failed checks and prepare a regenerated draft using the same plan."""
    attempt = state.get("attempt", 1)
    native_passed = state.get("native_eval_passed", False)
    rejection_log = list(state.get("rejection_log", []))

    if not native_passed:
        stage = "native"
        failed_checks = [r["check_id"] for r in state.get("native_results", []) if not r["passed"]]
        reasons = []
        for result in state.get("native_results", []):
            if not result["passed"]:
                reasons.append(result.get("message", "Native check failed."))
                reasons.extend(result.get("violations", [])[:3])
        suggested_fix = "Shorten sentences, include every planned section, and define technical terms on first use."
        required = (state.get("plan") or {}).get("core_sections", [])
        content = (state.get("draft") or {}).get("content", "")
        revision_sections = [section for section in required if f"## {section}" not in content]
        if not revision_sections:
            if set(failed_checks) & {"NATIVE_KNOWN_FALSE_CLAIMS", "NATIVE_JARGON_DEFINED", "NATIVE_ORDERED_STEPS"}:
                revision_sections = required[2:3] or required[-1:]
            else:
                revision_sections = required
    else:
        stage = "judge"
        failed_checks = [r["check_id"] for r in state.get("judge_results", []) if not r["passed"]]
        reasons = [f"{r['name']}: {r['reason']}" for r in state.get("judge_results", []) if not r["passed"]]
        suggested_fix = "Ground facts in sources, cover every planned key point, and use a clear practical example."
        failed_set = set(failed_checks)
        sections = (state.get("plan") or {}).get("core_sections", [])
        # JUDGE_COHERENT_FLOW and coverage failures require a full rewrite;
        # patching only some sections rarely fixes them.
        if "JUDGE_COHERENT_FLOW" in failed_set or "JUDGE_KEY_POINTS_COVERAGE" in failed_set:
            revision_sections = sections
        elif "JUDGE_TEACH_BY_EXAMPLE" in failed_set:
            revision_sections = sections[-1:]
        else:
            revision_sections = sections

    lessons_by_check = {}
    if not native_passed:
        lessons_by_check = {
            result.get("check_id", "UNKNOWN"): (result.get("violations", []) or [result.get("message", "")])[:2]
            for result in state.get("native_results", []) if not result.get("passed")
        }
    else:
        lessons_by_check = {
            result.get("check_id", "UNKNOWN"): [result.get("reason", "")]
            for result in state.get("judge_results", []) if not result.get("passed")
        }
    source_evidence = []
    for source in (state.get("source_pack") or {}).get("sources", [])[:3]:
        snippet = source.get("raw_content") or source.get("snippet", "")
        if snippet:
            source_evidence.append({"title": source.get("title", "Source"),
                                    "url": source.get("url", ""),
                                    "excerpt": str(snippet).strip()[:500]})

    # Record failure counters in memory.json
    topic = state.get("topic", "")
    profile_id = state.get("profile_id", "default")
    record_failures(failed_checks, profile_id=profile_id, topic=topic,
                    lessons_by_check=lessons_by_check)

    rejection_entry = RejectionLogEntry(
        attempt=attempt,
        stage_failed=stage,
        failed_check_ids=failed_checks,
        reasons=reasons,
        suggested_fix=suggested_fix,
        revised_sections=revision_sections,
        source_evidence=source_evidence,
    ).model_dump()

    rejection_log.append(rejection_entry)
    save_failure_lessons(topic, [rejection_entry])

    logger.log_event(
        step="router",
        attempt=attempt,
        status="retry",
        details={"stage_failed": stage, "failed_checks": failed_checks, "next_attempt": attempt + 1},
    )

    # Refresh active lessons from memory
    updated_lessons = get_active_lessons(profile_id=profile_id, topic=topic, max_lessons=3)

    return {
        "attempt": attempt + 1,
        "rejection_log": rejection_log,
        "memory_lessons": updated_lessons,
        "revision_sections": revision_sections,
        "revision_source_content": (state.get("draft") or {}).get("content", ""),
        # Clear judge state for next attempt
        "judge_eval_passed": None,
        "judge_results": [],
        "judge_unavailable": False,
    }


def record_terminal_failure(state: AgentState, logger: RunLogger) -> Dict[str, Any]:
    """Persist the final failed checks or make judge unavailability explicit without retrying."""
    attempt = state.get("attempt", 1)
    source_tool = (state.get("source_pack") or {}).get("source_tool", "")
    if source_tool == "offline_stub":
        logger.log_event(step="router", attempt=attempt, status="search_error",
                         details={"failure_reason": "All search backends failed; run drafted against offline stub and cannot pass."})
        return {"final_status": "search_error", "terminal_error": "search_error"}
        logger.log_event(step="router", attempt=attempt, status="evaluator_error",
                         details={"failure_reason": "Judge did not evaluate every required check; run stopped."})
        return {"final_status": "failed", "terminal_error": "judge_unavailable"}

    native_passed = state.get("native_eval_passed", False)
    stage = "judge" if native_passed else "native"
    results = state.get("judge_results", []) if native_passed else state.get("native_results", [])
    failed = [item for item in results if not item.get("passed")]
    check_ids = [item.get("check_id", "UNKNOWN") for item in failed]
    reasons = ([f"{item.get('name', item.get('check_id'))}: {item.get('reason', '')}" for item in failed]
               if native_passed else [line for item in failed for line in
                                      ([item.get("message", "Native check failed.")] + item.get("violations", [])[:2])])
    sections = (state.get("plan") or {}).get("core_sections", [])
    evidence = []
    for source in (state.get("source_pack") or {}).get("sources", [])[:3]:
        excerpt = source.get("raw_content") or source.get("snippet", "")
        if excerpt:
            evidence.append({"title": source.get("title", "Source"), "url": source.get("url", ""),
                             "excerpt": str(excerpt).strip()[:500]})
    entry = RejectionLogEntry(
        attempt=attempt, stage_failed=stage, failed_check_ids=check_ids, reasons=reasons,
        suggested_fix="The final attempt did not clear every hard pass/fail checkpoint.",
        revised_sections=[], source_evidence=evidence,
    ).model_dump()
    rejection_log = list(state.get("rejection_log", []))
    rejection_log.append(entry)
    profile_id = state.get("profile_id", "default")
    topic = state.get("topic", "")
    record_failures(check_ids, profile_id=profile_id, topic=topic,
                    lessons_by_check={item.get("check_id", "UNKNOWN"): [item.get("reason", item.get("message", ""))]
                                      for item in failed})
    save_failure_lessons(topic, [entry])
    logger.log_event(step="router", attempt=attempt, status="stopped",
                     details={"stage_failed": stage, "failed_checks": check_ids,
                              "reason": "maximum attempts exhausted"})
    return {"rejection_log": rejection_log, "final_status": "failed"}


def build_lesson_graph(logger: RunLogger, stream_callback=None):
    """Construct the executable LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Define nodes with logger injected
    workflow.add_node("context", lambda s: context_node(s, logger))
    workflow.add_node("search", lambda s: search_node(s, logger))
    workflow.add_node("plan", lambda s: plan_node(s, logger))
    workflow.add_node("generate", lambda s: generate_node(s, logger, stream_callback))
    workflow.add_node("eval_native", lambda s: eval_native_node(s, logger))
    workflow.add_node("eval_judge", lambda s: eval_judge_node(s, logger))
    workflow.add_node("record_retry", lambda s: record_retry_and_rejection(s, logger))
    workflow.add_node("record_terminal_failure", lambda s: record_terminal_failure(s, logger))

    # Static edges
    workflow.set_entry_point("context")
    workflow.add_edge("context", "search")
    workflow.add_edge("search", "plan")
    workflow.add_edge("plan", "generate")
    workflow.add_edge("generate", "eval_native")

    # Native failures retry generation with the original approved plan.
    workflow.add_conditional_edges(
        "eval_native",
        routing_decision,
        {
            "eval_judge": "eval_judge",
            "retry_generate": "record_retry",
            "ship_failure": "record_terminal_failure",
        },
    )

    # Conditional edge after eval_judge:
    # If judge passed -> END (success)
    # If judge failed -> record_retry (if retries left) or END (failure)
    workflow.add_conditional_edges(
        "eval_judge",
        routing_decision,
        {
            "ship_success": END,
            "retry_generate": "record_retry",
            "ship_failure": "record_terminal_failure",
        },
    )

    # Keep the approved plan fixed and regenerate against its feedback.
    workflow.add_edge("record_retry", "generate")
    workflow.add_edge("record_terminal_failure", END)

    return workflow.compile()

