"""Tier-2 LLM-as-a-judge evaluation gate."""

import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from langsmith import traceable
from src.schemas import JudgeCheckResult
from src.llm import call_llm, extract_json_from_text, DEFAULT_JUDGE_MODEL, DEFAULT_MAX_TOKENS
from src.nodes.plan import load_prompt_template, build_source_context


class JudgeOutputList(BaseModel):
    """Schema wrapper for parsing list of check results."""
    checks: List[JudgeCheckResult]


def expected_judge_checks(topic: str) -> List[Tuple[str, str]]:
    return [
        ("JUDGE_ACCURACY_GROUNDING", "Factual Accuracy & Grounding"),
        ("JUDGE_AUDIENCE_FIT", "Audience Fit"),
        ("JUDGE_TEACH_BY_EXAMPLE", "Concrete Step-by-Step Example"),
        ("JUDGE_KEY_POINTS_COVERAGE", f"Core {topic} Concepts Coverage"),
        ("JUDGE_COHERENT_FLOW", "Pedagogical Coherence & Flow"),
    ]


def parse_judge_checks(response: str, topic: str) -> List[JudgeCheckResult]:
    """Accept the common JSON array/object shapes emitted by chat models."""
    parsed = extract_json_from_text(response)
    expected_pairs = expected_judge_checks(topic)
    expected = dict(expected_pairs)
    id_by_name = {name.casefold(): check_id for check_id, name in expected_pairs}
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = next((parsed[key] for key in ("checks", "results", "evaluations", "check_results")
                      if isinstance(parsed.get(key), list)), None)
        if items is None:
            items = [dict(value, check_id=key) for key, value in parsed.items()
                     if key in expected and isinstance(value, dict)]
    else:
        return []

    results: Dict[str, JudgeCheckResult] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("check_id") or item.get("id") or item.get("check") or ""
        check_id = str(raw_id).strip().upper()
        if check_id not in expected:
            name_id = item.get("name") or item.get("check_name")
            if name_id:
                check_id = id_by_name.get(str(name_id).strip().casefold(), check_id)
        if check_id not in expected:
            continue
        raw_passed = item.get("passed", item.get("pass", item.get("result", item.get("verdict"))))
        if isinstance(raw_passed, str):
            normalized = raw_passed.strip().casefold()
            if normalized in {"pass", "passed", "true", "yes"}:
                raw_passed = True
            elif normalized in {"fail", "failed", "false", "no"}:
                raw_passed = False
        if not isinstance(raw_passed, bool):
            continue
        reason = (item.get("reason") or item.get("rationale") or item.get("feedback")
                  or item.get("explanation") or "No explanation returned.")
        try:
            results[check_id] = JudgeCheckResult(
                check_id=check_id,
                name=expected[check_id],
                passed=raw_passed,
                reason=str(reason),
                evidence_quote=item.get("evidence_quote") or item.get("evidence"),
                source_quote=next((item[key] for key in ("source_quote", "source_evidence")
                                   if isinstance(item.get(key), str) and item[key].strip()), None),
                missing_points=[str(point) for point in item.get("missing_points", [])]
                if isinstance(item.get("missing_points", []), list) else [],
            )
        except Exception as exc:
            print(f"[Judge] Failed to validate {check_id}: {type(exc).__name__}")
    return list(results.values())


def _normalized(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _invalid_evidence(results: List[JudgeCheckResult], draft_text: str,
                      source_context: str) -> List[str]:
    """Return check IDs whose quotes are not verbatim from the lesson or sources."""
    lesson = _normalized(draft_text)
    sources = _normalized(source_context)
    no_sources = "no external sources available" in sources
    invalid: List[str] = []
    for result in results:
        if result.judge_error:
            continue
        quote = _normalized(result.evidence_quote)
        if quote and quote not in lesson:
            invalid.append(result.check_id)
            continue
        source_quote = _normalized(result.source_quote)
        if source_quote and source_quote not in sources:
            invalid.append(result.check_id)
            continue
        if (result.check_id == "JUDGE_ACCURACY_GROUNDING" and not result.passed
                and not no_sources and not source_quote):
            invalid.append(result.check_id)
    return invalid


def _complete_checks(results: List[JudgeCheckResult], topic: str,
                     must_cover_points: Optional[List[str]]) -> List[JudgeCheckResult]:
    """Fill missing judge checks conservatively and validate coverage details."""
    existing_ids = {result.check_id for result in results}
    for check_id, name in expected_judge_checks(topic):
        if check_id not in existing_ids:
            results.append(JudgeCheckResult(
                check_id=check_id,
                name=name,
                passed=False,
                reason="Judge did not evaluate this check; conservatively failed. This is an evaluation error, not a lesson finding.",
                judge_error=True,
            ))

    coverage = next((item for item in results if item.check_id == "JUDGE_KEY_POINTS_COVERAGE"), None)
    if coverage and must_cover_points:
        allowed = set(must_cover_points)
        # Some models name missing points in prose but omit the structured
        # field. Recover exact plan-point mentions before treating this as a
        # judge-format failure.
        if not coverage.missing_points:
            reason_folded = coverage.reason.casefold()
            coverage.missing_points = [point for point in must_cover_points
                                       if point.casefold() in reason_folded]
        coverage.missing_points = [point for point in coverage.missing_points if point in allowed]
        if coverage.missing_points:
            coverage.passed = False
            missing_text = "Missing required points: " + "; ".join(coverage.missing_points)
            coverage.reason = f"{coverage.reason} {missing_text}"
        elif not coverage.passed:
            coverage.reason += " Judge did not identify specific missing points. This is an evaluation error, not a lesson finding."
            coverage.judge_error = True
    return results


def _mark_judge_errors(results: List[JudgeCheckResult], invalid: List[str]) -> None:
    for result in results:
        if result.check_id in invalid and not result.judge_error:
            result.passed = False
            result.judge_error = True
            result.reason = (
                f"{result.reason} Evidence was not copied verbatim from the lesson or sources; "
                "this is an evaluation error, not a lesson finding."
            ).strip()


@traceable(name="eval_tier2_judge")
def evaluate_with_judge(
    topic: str,
    draft_text: str,
    source_pack: Dict[str, Any],
    must_cover_points: Optional[List[str]] = None,  # NEW: from plan
    learner_profile: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_JUDGE_MODEL,
    run_attempt: int = 1,
    audit_callback=None,
) -> Tuple[bool, List[JudgeCheckResult]]:
    """Execute Tier-2 LLM-as-a-judge binary checkpoints."""
    template = load_prompt_template("judge.md")
    # Ensure the judge sees snippets for every URL the draft cites, even if
    # those sources fall beyond the default PROMPT_SOURCE_COUNT window.
    cited_urls = list(set(re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", draft_text)))
    source_context = build_source_context(source_pack, cited_urls=cited_urls)

    points_text = "\n".join([f"  - {p}" for p in must_cover_points]) if must_cover_points else "Evaluate based on the topic's core concepts."

    prompt_content = (
        template.replace("{topic}", topic)
        .replace("{source_context}", source_context)
        .replace("{draft_text}", draft_text)
        .replace("{must_cover_points}", points_text)
        .replace("{learner_profile}", (
            f"Education: {(learner_profile or {}).get('education_background', 'not specified')}; "
            f"expertise: {(learner_profile or {}).get('expertise_level', 'beginner')}; "
            f"learning needs: {(learner_profile or {}).get('target_audience', 'beginner learner')}"
        ))
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful pedagogical auditor. Evaluate only the five named checks. "
                "Output one JSON object with a 'checks' array. Each item must include check_id, passed (boolean), reason, and evidence_quote. "
                "Evidence quotes must be copied verbatim from the lesson draft. "
                "Do not add prose, Markdown fences, or extra check IDs."
            ),
        },
        {"role": "user", "content": prompt_content},
    ]

    raw_response = call_llm(messages=messages, model=model, temperature=0.1,
                            max_tokens=DEFAULT_MAX_TOKENS["judge"], stage="judge",
                            response_format={"type": "json_object"}, run_attempt=run_attempt,
                            audit_callback=audit_callback)
    results = parse_judge_checks(raw_response, topic)

    # Retry the judge once if it returned no parseable checks at all, before
    # declaring judge_unavailable.  This is one cheap call, not a full loop.
    if not results:
        raw_response = call_llm(messages=messages, model=model, temperature=0.1,
                                max_tokens=DEFAULT_MAX_TOKENS["judge"], stage="judge_retry",
                                response_format={"type": "json_object"}, run_attempt=run_attempt,
                                audit_callback=audit_callback)
        results = parse_judge_checks(raw_response, topic)

    results = _complete_checks(results, topic, must_cover_points)
    invalid = _invalid_evidence(results, draft_text, source_context)

    # A quote that is not verbatim is a judge formatting error. Give the judge
    # one chance to copy the evidence exactly; if it still fails, fail closed.
    if invalid and results:
        repair_messages = messages + [{
            "role": "user",
            "content": (
                "Your previous JSON had non-verbatim evidence for these checks: "
                + ", ".join(sorted(set(invalid)))
                + ". Return the same five checks again. Copy each evidence_quote exactly "
                "from the lesson draft. For accuracy failures, copy source_quote exactly "
                "from the verified sources. For a missing coverage point, leave "
                "evidence_quote empty and use missing_points."
            ),
        }]
        raw_response = call_llm(messages=repair_messages, model=model, temperature=0.1,
                                max_tokens=DEFAULT_MAX_TOKENS["judge"], stage="judge_repair",
                                response_format={"type": "json_object"}, run_attempt=run_attempt,
                                audit_callback=audit_callback)
        repaired = parse_judge_checks(raw_response, topic)
        if repaired:
            results = _complete_checks(repaired, topic, must_cover_points)
            invalid = _invalid_evidence(results, draft_text, source_context)

    _mark_judge_errors(results, invalid)
    all_passed = all(r.passed for r in results)
    return all_passed, results
