"""Unit tests for Pydantic schema validation and tolerant coercion."""

import pytest
from src.schemas import (
    DEFAULT_CULTURAL_CONTEXT,
    OLD_DEFAULT_CULTURAL_CONTEXT,
    LearnerProfile,
    LessonPlan,
    SourcePack,
    SourceItem,
    UserProfile,
)
from src.llm import coerce_to_schema
from src.nodes.plan import normalize_topic, canonical_sections
from src.nodes.generate import (
    expand_inline_numbered_lists,
    normalize_lesson_format,
    strip_preamble_to_first_heading,
)
from src.nodes.eval_native import check_ordered_process_steps


def test_coerce_valid_dict_to_lesson_plan():
    data = {
        "topic": "Introduction to RAG",
        "target_audience": "12th grade",
        "core_sections": ["Section 1", "Section 2"],
    }
    plan = coerce_to_schema(data, LessonPlan)
    assert plan is not None
    assert plan.topic == "Introduction to RAG"
    assert len(plan.core_sections) == 2


def test_coerce_raw_list_of_strings_to_lesson_plan():
    # This reproduces the exact error reported by the user
    data = [
        "What is RAG?",
        "Why Does RAG Matter?",
        "How RAG Works (Step-by-Step)",
        "Real-World Example & Analogy",
    ]
    plan = coerce_to_schema(data, LessonPlan)
    assert plan is not None
    assert plan.core_sections == data
    assert plan.topic == ""  # The coercer must not invent a topic absent from provider output.


def test_coerce_single_item_dict_list_to_lesson_plan():
    data = [{"topic": "RAG Advanced", "core_sections": ["Part A"]}]
    plan = coerce_to_schema(data, LessonPlan)
    assert plan is not None
    assert plan.topic == "RAG Advanced"
    assert plan.core_sections == ["Part A"]


def test_question_topic_creates_stable_nonduplicated_sections():
    topic = normalize_topic("what is AI/ML?")
    assert topic == "AI/ML"
    assert canonical_sections(topic) == [
        "What is AI/ML?", "Why does AI/ML matter?",
        "How does AI/ML work?", "Practical example: AI/ML",
    ]


def test_comparison_question_normalizes_and_gets_comparison_sections():
    topic = normalize_topic("What is AI & What is ML, how they different, give me a difference here?")
    assert topic == "AI and ML"
    assert canonical_sections(topic) == [
        "What are AI and ML?",
        "Why do AI and ML matter?",
        "How do AI and ML fit together?",
        "Practical example: AI and ML",
    ]


def test_contraction_and_comma_list_normalizes_to_joined_subjects():
    assert normalize_topic("what's a vector, matrix, tensor?") == "vector, matrix and tensor"
    assert canonical_sections("what's a vector, matrix, tensor?") == [
        "What are vector, matrix and tensor?",
        "Why do vector, matrix and tensor matter?",
        "How do vector, matrix and tensor fit together?",
        "Practical example: vector, matrix and tensor",
    ]


def test_difference_between_question_strips_lead_in():
    assert normalize_topic("What is the difference between a vector and a matrix?") == "vector and matrix"


def test_work_question_removes_action_words_from_topic_heading():
    assert normalize_topic("How does RAG work?") == "RAG"


def test_old_cultural_context_default_is_migrated_but_custom_values_are_kept():
    assert UserProfile(cultural_context=OLD_DEFAULT_CULTURAL_CONTEXT).cultural_context == DEFAULT_CULTURAL_CONTEXT
    assert LearnerProfile(cultural_context=OLD_DEFAULT_CULTURAL_CONTEXT).cultural_context == DEFAULT_CULTURAL_CONTEXT
    assert UserProfile(cultural_context="My own examples").cultural_context == "My own examples"


def test_generated_headings_are_aligned_to_the_plan():
    lesson = "## First title\nFirst body.\n### Second title\nSecond body."
    result = normalize_lesson_format(lesson, ["What is AI?", "Why does AI matter?"], "AI")
    assert result.startswith("# AI")
    assert "## What is AI?" in result
    assert "## Why does AI matter?" in result


def test_flattened_numbered_list_is_repaired_before_evaluation():
    flattened = "1. A scalar is one value. 2. A vector is a list. 3. A matrix is a grid."
    repaired = expand_inline_numbered_lists(flattened)
    assert repaired.splitlines() == [
        "1. A scalar is one value.",
        "2. A vector is a list.",
        "3. A matrix is a grid.",
    ]
    lesson = f"## How does X work?\n{flattened}\n\n## Practical example: X\nOne 2 by 2 grid."
    normalized = normalize_lesson_format(lesson, ["How does X work?", "Practical example: X"], "X")
    assert check_ordered_process_steps(normalized, "How does X work?", 3).passed


def test_preamble_stripping_preserves_level_two_heading():
    result = strip_preamble_to_first_heading("Thinking...\n\n## What is AI?\nAI is a field.")
    assert result.startswith("## What is AI?")

