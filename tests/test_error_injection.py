from src.nodes.eval_native import check_jargon_definitions, check_known_false_claims
from src.nodes.generate import generate_lesson_draft
from unittest.mock import patch


def test_injected_false_claim_and_unexplained_jargon_are_caught():
    topic = "RAG"
    plan = {"topic": topic, "core_sections": [
        "What is RAG?", "Why does RAG matter?", "How does RAG work?", "Practical example: RAG",
    ]}
    draft_text = (
        "# RAG\n\n## What is RAG?\nRAG finds facts.\n\n## Why does RAG matter?\nIt can help.\n\n"
        "## How does RAG work?\nStep 1. It searches.\nStep 2. It finds notes.\nStep 3. It answers.\n\n"
        "## Practical example: RAG\nA student checks a book."
    )
    with patch("src.nodes.generate.call_llm", return_value=draft_text):
        draft = generate_lesson_draft(topic, plan, {"sources": []}, [], inject_error="factual_jargon")

    false_claim = check_known_false_claims(draft.content, [
        "RAG retrains the model's weights every time a question is asked",
    ])
    jargon = check_jargon_definitions(draft.content)
    assert false_claim.passed is False
    assert jargon.passed is False
    assert "model weights" in " ".join(jargon.violations)


def test_paraphrased_falsehood_not_caught_by_native_but_injected():
    """The paraphrased falsehood is not in the forbidden_claims list,
    so the native NATIVE_KNOWN_FALSE_CLAIMS check should NOT catch it.
    The LLM judge is expected to catch it via JUDGE_ACCURACY_GROUNDING."""
    topic = "RAG"
    plan = {"topic": topic, "core_sections": [
        "What is RAG?", "Why does RAG matter?", "How does RAG work?", "Practical example: RAG",
    ]}
    draft_text = (
        "# RAG\n\n## What is RAG?\nRAG finds facts.\n\n## Why does RAG matter?\nIt can help.\n\n"
        "## How does RAG work?\nStep 1. It searches.\nStep 2. It finds notes.\nStep 3. It answers.\n\n"
        "## Practical example: RAG\nA student checks a book."
    )
    with patch("src.nodes.generate.call_llm", return_value=draft_text):
        draft = generate_lesson_draft(topic, plan, {"sources": []}, [], inject_error="paraphrased_falsehood")

    # The paraphrased claim should be present in the draft.
    assert "RAG permanently teaches the model every new document you add." in draft.content

    # The native known-false-claims check should NOT catch it (it's not in the forbidden list).
    false_claim = check_known_false_claims(draft.content, [
        "RAG retrains the model's weights every time a question is asked",
    ])
    assert false_claim.passed is True  # Native gate does not catch the paraphrased version.


def test_incomplete_section_revision_falls_back_to_full_draft():
    plan = {"topic": "RAG", "core_sections": [
        "What is RAG?", "Why does RAG matter?", "How does RAG work?", "Practical example: RAG",
    ]}
    full_draft = (
        "# RAG\n\n## What is RAG?\nRAG finds facts.\n\n"
        "## Why does RAG matter?\nIt helps.\n\n"
        "## How does RAG work?\nStep 1. Search.\nStep 2. Find.\nStep 3. Answer.\n\n"
        "## Practical example: RAG\nA student looks up a fact."
    )
    with (
        patch("src.nodes.generate.revise_lesson_sections", side_effect=ValueError(
            "Section revision omitted required section: How does RAG work?")),
        patch("src.nodes.generate.call_llm", return_value=full_draft) as call,
    ):
        draft = generate_lesson_draft(
            "RAG", plan, {"sources": []}, [], rejection_feedback=[{"attempt": 1, "reasons": ["Fix it"]}],
            revision_sections=["How does RAG work?"], revision_source_content=full_draft,
            run_attempt=2,
        )

    assert all(f"## {section}" in draft.content for section in plan["core_sections"])
    assert call.call_args.kwargs["stage"] == "revision_fallback"
