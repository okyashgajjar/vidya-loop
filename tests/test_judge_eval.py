"""Judge output parsing and one-pass JSON repair tests."""

import json
from unittest.mock import patch

from src.nodes.eval_judge import evaluate_with_judge, parse_judge_checks, expected_judge_checks


def _valid_checks(topic="Machine Learning"):
    return {
        "checks": [
            {"check_id": check_id, "passed": True, "reason": "Evidence supports this check.",
             "evidence_quote": "A lesson."}
            for check_id, _ in expected_judge_checks(topic)
        ]
    }


def test_parser_accepts_wrapped_checks_and_string_boolean():
    payload = _valid_checks()
    payload["checks"][0]["passed"] = "pass"
    results = parse_judge_checks(json.dumps(payload), "Machine Learning")
    assert len(results) == 5
    assert all(result.passed for result in results)


def test_judge_missing_checks_fail_closed_without_an_extra_call():
    with patch("src.nodes.eval_judge.call_llm", return_value="not json") as mock_call:
        passed, results = evaluate_with_judge(
            topic="Machine Learning", draft_text="A lesson.", source_pack={},
            must_cover_points=["Core definition"],
        )
    assert passed is False
    assert len(results) == 5
    assert mock_call.call_count == 2  # initial call + one retry on empty parse


def test_missing_judge_checks_report_format_failure_clearly():
    with patch("src.nodes.eval_judge.call_llm", return_value="bad output") as mock_call:
        passed, results = evaluate_with_judge(
            topic="Machine Learning", draft_text="A lesson.", source_pack={},
        )
    assert passed is False
    assert len(results) == 5
    assert all("evaluation error, not a lesson finding" in result.reason for result in results)
    assert mock_call.call_count == 2  # initial call + one retry on empty parse


def test_non_verbatim_evidence_fails_closed_after_one_repair_call():
    payload = _valid_checks()
    payload["checks"][0]["evidence_quote"] = "Not copied from the lesson."
    with patch("src.nodes.eval_judge.call_llm", return_value=json.dumps(payload)) as mock_call:
        passed, results = evaluate_with_judge(
            topic="Machine Learning", draft_text="A lesson.", source_pack={},
        )
    assert passed is False
    first = next(item for item in results if item.check_id == "JUDGE_ACCURACY_GROUNDING")
    assert first.judge_error is True
    assert "evaluation error" in first.reason
    assert mock_call.call_count == 2  # initial call + one evidence repair call


def test_coverage_point_named_in_reason_is_a_lesson_failure_not_judge_error():
    point = "AI is broader than machine learning."
    payload = _valid_checks()
    coverage = next(item for item in payload["checks"]
                    if item["check_id"] == "JUDGE_KEY_POINTS_COVERAGE")
    coverage.update(passed=False, reason=f"Missing this plan point: '{point}'")
    with patch("src.nodes.eval_judge.call_llm", return_value=json.dumps(payload)):
        passed, results = evaluate_with_judge(
            topic="AI and machine learning", draft_text="A lesson.", source_pack={},
            must_cover_points=[point],
        )
    check = next(item for item in results if item.check_id == "JUDGE_KEY_POINTS_COVERAGE")
    assert passed is False
    assert check.missing_points == [point]
    assert "evaluation error" not in check.reason
