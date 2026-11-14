"""Run one lesson through the same LangGraph pipeline used by Streamlit."""

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph import MAX_ATTEMPTS, build_lesson_graph
from src.logger import RunLogger
from src.memory import get_active_lessons, load_profile
from src.db import ensure_session, finish_run, start_run, sync_profile
from src.topic_validation import topic_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate, evaluate, and optionally revise a beginner lesson.")
    parser.add_argument("topic", help="Lesson topic, for example: Introduction to RAG")
    parser.add_argument("--profile", default="default", help="Learner profile ID")
    parser.add_argument("--inject-error", nargs="?", const="factual_jargon", default=None,
                        help="Inject an error for testing: 'factual_jargon' (caught by native gate) or 'paraphrased_falsehood' (caught by judge only)")
    args = parser.parse_args()
    problem = topic_error(args.topic)
    if problem:
        parser.error(problem)

    logger = RunLogger()
    profile = load_profile(args.profile)
    sync_profile(profile.model_dump())
    session_id = f"cli-{uuid.uuid4().hex[:12]}"
    ensure_session(session_id, args.profile)
    start_run(logger.run_id, session_id, args.topic.strip())
    state = {
        "topic": args.topic.strip(), "profile_id": args.profile, "learner_profile": profile.model_dump(),
        "attempt": 1, "rejection_log": [], "memory_lessons": get_active_lessons(args.profile, args.topic, 3),
        "inject_error": args.inject_error,
        "final_status": "in_progress", "semantic_context": {},
        "session_id": session_id, "session_source_pack": {},
    }
    try:
        final = build_lesson_graph(logger).invoke(state, config={
            "run_name": "cli_lesson",
            "metadata": {"run_id": logger.run_id, "profile_id": args.profile,
                         "session_id": session_id, "topic": args.topic.strip()},
            "tags": ["self-evaluating-agent", "cli"],
        })
    except Exception as exc:
        finish_run(logger.run_id, "failed", 1, {})
        print(f"FAILED: {type(exc).__name__}: {exc}")
        for event in logger.read_all_events():
            print(event)
        return 1

    passed = final.get("native_eval_passed", False) and final.get("judge_eval_passed", False)
    finish_run(logger.run_id, "passed" if passed else "failed", final.get("attempt", 1), {})
    print(f"Status: {'PASSED' if passed else 'FAILED'}; attempts: {final.get('attempt', 1)}/{MAX_ATTEMPTS}")
    print("Rejection log:")
    for entry in final.get("rejection_log", []):
        print(entry)
    print("Lesson:")
    print((final.get("draft") or {}).get("content", "No lesson generated."))
    print(f"Audit: logs/{logger.run_id}.jsonl")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
