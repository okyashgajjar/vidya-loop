"""Graph integration checks for stable, linear retry execution."""

from unittest.mock import patch

from src.graph import build_lesson_graph
from src.schemas import DraftLesson, LessonPlan, SourcePack, SourceItem, NativeCheckResult, JudgeCheckResult


class FakeLogger:
    run_id = "test-run"

    def start_step(self, step):
        pass

    def log_event(self, **kwargs):
        pass


def test_retries_reuse_plan_and_regenerate_in_sequence():
    calls = {"plan": 0, "generate": 0, "native": 0, "judge": 0}
    injected = []
    plan = LessonPlan(
        topic="AI", core_sections=["What is AI?", "Why does AI matter?"],
        must_cover_points=["Definition", "Example"],
    )

    def fake_plan(**kwargs):
        calls["plan"] += 1
        return plan

    def fake_generate(**kwargs):
        calls["generate"] += 1
        injected.append(kwargs["inject_error"])
        return DraftLesson(title="AI", content="# AI\n\n## What is AI?\nAI is a field.\n## Why does AI matter?\nAI can help.")

    def fake_native(*args, **kwargs):
        calls["native"] += 1
        passed = calls["native"] > 1
        return passed, [NativeCheckResult(
            check_id="NATIVE_TEST", name="test", passed=passed,
            message="pass" if passed else "retry once", violations=[] if passed else ["repair this"],
        )]

    source_pack = SourcePack(topic="AI", source_tool="test", sources=[], fetched_at="now")
    with (
        patch("src.graph.fetch_source_pack", return_value=source_pack),
        patch("src.graph.plan_lesson", side_effect=fake_plan),
        patch("src.graph.generate_lesson_draft", side_effect=fake_generate),
        patch("src.graph.evaluate_native_all", side_effect=fake_native),
        patch("src.graph.evaluate_with_judge", side_effect=lambda **kwargs: (calls.__setitem__("judge", calls["judge"] + 1) or (True, [JudgeCheckResult(
            check_id="JUDGE_TEST", name="test", passed=True, reason="pass",
        )]))),
        patch("src.graph.record_failures"),
        patch("src.graph.get_active_lessons", return_value=[]),
        patch("src.graph.save_failure_lessons"),
        patch("src.graph.retrieve_semantic_context", return_value={"status": "skipped", "memory_lessons": [], "prompt_block": "", "source_pack": None, "similarity_hits": []}),
    ):
        final = build_lesson_graph(FakeLogger()).invoke({
            "topic": "What is AI?", "attempt": 1, "rejection_log": [],
            "memory_lessons": [], "profile_id": "test-profile", "inject_error": "long_sentence",
        })

    assert calls == {"plan": 1, "generate": 2, "native": 2, "judge": 1}
    assert injected == ["long_sentence", None]
    assert final["attempt"] == 2
    assert final["plan"]["core_sections"] == plan.core_sections


def test_judge_unavailable_stops_without_spending_revision_attempt():
    counts = {"generate": 0, "judge": 0}
    plan = LessonPlan(topic="AI", core_sections=["What is AI?", "Why does AI matter?", "How does AI work?", "Practical example: AI"])
    source_pack = SourcePack(topic="AI", source_tool="test", sources=[], fetched_at="now")

    def fake_generate(**kwargs):
        counts["generate"] += 1
        return DraftLesson(title="AI", content="# AI\n\n## What is AI?\nAI helps people.\n\n## Why does AI matter?\nIt can help.\n\n## How does AI work?\nStep 1. It starts.\nStep 2. It acts.\nStep 3. It ends.\n\n## Practical example: AI\nA simple example.")

    unavailable = [JudgeCheckResult(check_id=check_id, name=name, passed=False,
                                    reason="Judge did not evaluate this check; conservatively failed. This is an evaluation error, not a lesson finding.",
                                    judge_error=True)
                   for check_id, name in [("JUDGE_ACCURACY_GROUNDING", "Accuracy"),
                                          ("JUDGE_AUDIENCE_FIT", "Audience"),
                                          ("JUDGE_TEACH_BY_EXAMPLE", "Example"),
                                          ("JUDGE_KEY_POINTS_COVERAGE", "Coverage"),
                                          ("JUDGE_COHERENT_FLOW", "Flow")]]
    def fake_judge(**kwargs):
        counts["judge"] += 1
        return False, unavailable

    with (
        patch("src.graph.retrieve_semantic_context", return_value={"status": "skipped", "memory_lessons": [], "prompt_block": "", "source_pack": None, "similarity_hits": []}),
        patch("src.graph.fetch_source_pack", return_value=source_pack),
        patch("src.graph.plan_lesson", return_value=plan),
        patch("src.graph.generate_lesson_draft", side_effect=fake_generate),
        patch("src.graph.evaluate_native_all", return_value=(True, [NativeCheckResult(check_id="NATIVE_TEST", name="Native", passed=True, message="ok")])),
        patch("src.graph.evaluate_with_judge", side_effect=fake_judge),
    ):
        final = build_lesson_graph(FakeLogger()).invoke({"topic": "AI", "attempt": 1, "rejection_log": [], "profile_id": "test"})

    assert counts == {"generate": 1, "judge": 1}
    assert final["attempt"] == 1
    assert final["judge_unavailable"] is True
