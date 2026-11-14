"""Single-turn follow-up answers grounded in the shipped lesson and its frozen sources."""

from typing import Any, Dict, Optional
from langsmith import traceable

from src.llm import call_llm, DEFAULT_GEN_MODEL
from src.nodes.eval_native import check_readability, check_sentence_length


class NoShippedLessonError(ValueError):
    pass


def _native_results(answer: str, profile: Optional[Dict[str, Any]] = None):
    profile = profile or {}
    return [
        check_readability(answer,
                          min_reading_ease=float(profile.get("min_reading_ease", 60)),
                          max_grade_level=float(profile.get("max_grade_level", 8.5))),
        check_sentence_length(answer,
                              max_avg_words=int(profile.get("sentence_len_avg_max", 16)),
                              hard_max_words=int(profile.get("sentence_len_max", 25))),
    ]


@traceable(name="lesson_followup")
def answer_followup(question: str, lesson: str, source_pack: Dict[str, Any],
                    learner_profile: Optional[Dict[str, Any]] = None,
                    model: str = DEFAULT_GEN_MODEL, audit_callback=None,
                    conversation_history: Optional[list] = None) -> Dict[str, Any]:
    """Answer directly with one call; one optional native-triggered simplification is allowed."""
    if not lesson.strip():
        raise NoShippedLessonError("Follow-ups require a passing lesson in this session.")
    sources = (source_pack or {}).get("sources", [])[:3]
    source_text = "\n".join(
        f"- {item.get('title', 'Source')}: {(item.get('raw_content') or item.get('snippet', ''))[:900]}"
        for item in sources
    ) or "No saved source snippets are available."
    history = [item for item in (conversation_history or [])
               if item.get("role") in {"user", "assistant"}][-8:]
    history_text = "\n".join(
        f"{item['role'].upper()}: {str(item.get('content', ''))[-1200:]}" for item in history
    ) or "No earlier follow-up turns in this session."
    system = (
        "Answer the learner directly in fewer than 150 words. Use short, simple sentences. "
        "Define any new technical term in one line. Use the example they request; otherwise use one simple Indian everyday example. "
        "Never refuse because the lesson or snippets lack a detail. Answer using layer 3 when needed. "
        "Use these three layers in order: (1) the shipped lesson, (2) the saved source snippets, "
        "(3) careful general knowledge only if needed. Prefix every claim from layer 3 with exactly "
        "'Extra, not in your lesson:'. Use recent session turns only to resolve references. "
        "Never use another session's conversation. If unsure, say so plainly. Do not search, re-plan, or evaluate the lesson."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"SHIPPED LESSON:\n{lesson[:12000]}\n\nFROZEN SESSION SOURCE SNIPPETS:\n{source_text}\n\nRECENT TURNS FROM THIS SESSION ONLY:\n{history_text}\n\nLEARNER QUESTION:\n{question}"},
    ]
    answer = call_llm(messages, model=model, max_tokens=600, max_retries=1,
                      stage="followup", audit_callback=audit_callback)
    checks = _native_results(answer, learner_profile)
    refusal = "lesson does not contain" in answer.casefold() or "lesson doesn't contain" in answer.casefold()
    if not all(check.passed for check in checks) or refusal or len(answer.split()) > 150:
        feedback = "; ".join([check.message for check in checks if not check.passed])
        if refusal:
            feedback += "; answer directly using the source snippets or clearly labeled careful general knowledge"
        answer = call_llm([
            {"role": "system", "content": system + " Simplify the supplied answer while preserving its facts and directness."},
            {"role": "user", "content": f"QUESTION: {question}\nNATIVE CHECK FEEDBACK: {feedback}\nANSWER TO SIMPLIFY:\n{answer}"},
        ], model=model, max_tokens=600, max_retries=1,
           stage="followup_simplify", audit_callback=audit_callback)
        checks = _native_results(answer, learner_profile)
    if len(answer.split()) > 150:
        words = answer.split()
        clipped = " ".join(words[:150])
        last_stop = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
        answer = clipped[:last_stop + 1].strip() if last_stop > 0 else clipped
    return {"answer": answer, "checks": checks,
            "passed": all(check.passed for check in checks) and len(answer.split()) <= 150}
