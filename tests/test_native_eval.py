"""Unit tests for Tier-1 deterministic native evaluation gate."""

import pytest
from src.nodes.eval_native import (
    check_readability,
    check_sentence_length,
    check_required_sections,
    check_jargon_definitions,
    check_citation_grounding,
    check_ordered_process_steps,
    check_example_has_numbers,
    evaluate_native_all,
    extract_sentences,
)

SAMPLE_PASSING_LESSON = """# Introduction to RAG for Beginners

## What is RAG?
RAG stands for Retrieval-Augmented Generation, meaning an AI searches notes before answering. Retrieval means finding useful notes. Think of it as an open-book exam for AI. A regular AI answers questions from memory alone. RAG lets the AI look up facts in a textbook first.

## Why Does RAG Matter?
AI models sometimes make mistakes. When an AI invents facts, we call it a hallucination, meaning a confident but false answer. RAG stops hallucinations by providing fresh facts from verified notes.

## How RAG Works (Step-by-Step)
First, we store our documents in a vector database, which is like a digital library index.
Second, the system performs retrieval, meaning it finds the best matching paragraph.
Third, the prompt augmentation step (meaning adding retrieved facts to your question) prepares the complete prompt.
Finally, the AI reads both the question and the facts to give an accurate answer.

## Real-World Example & Analogy
Imagine a student preparing for board exams. An open-book test is always safer than guessing from memory. That is exactly what RAG does for AI assistants.
"""

SAMPLE_FAILING_LONG_SENTENCE = """# Introduction to RAG

## What is RAG?
Retrieval-Augmented Generation is a sophisticated mechanism that bridges external non-parametric retrieval engines with parametric neural language models by continuously querying high-dimensional embedding spaces across multiple dense retrieval partitions simultaneously.

## Why Does RAG Matter?
It prevents errors.

## How RAG Works (Step-by-Step)
Step 1. Step 2.

## Real-World Example & Analogy
Example here.
"""

SAMPLE_FAILING_MISSING_HEADERS = """# Introduction to RAG
This is a short post about RAG.
It does not have the mandatory sections required by the rubric.
"""


def test_passing_lesson_passes_all_checks():
    passed, results = evaluate_native_all(SAMPLE_PASSING_LESSON)
    for r in results:
        assert r.passed, f"Check {r.check_id} failed: {r.message}"
    assert passed is True


def test_sentence_length_hard_failure():
    result = check_sentence_length(SAMPLE_FAILING_LONG_SENTENCE, hard_max_words=25)
    assert result.passed is False
    assert any("exceed hard limit" in v for v in result.violations)


def test_missing_required_headers():
    required = [
        "What is RAG?",
        "Why Does RAG Matter?",
        "How RAG Works (Step-by-Step)",
        "Real-World Example & Analogy",
    ]
    result = check_required_sections(SAMPLE_FAILING_MISSING_HEADERS, required)
    assert result.passed is False
    assert len(result.violations) > 0


def test_ordered_process_check_uses_configured_section_and_minimum():
    good = "## How it works\nStep 1. Find files.\nStep 2. Rank files.\nStep 3. Answer."
    bad = "## How it works\nThe system searches then answers."
    assert check_ordered_process_steps(good, "How it works", 3).passed
    assert not check_ordered_process_steps(bad, "How it works", 3).passed


def test_sentence_splitter_handles_closing_quotes():
    text = (
        'Imagine a manual says: "Vacation days are 20 days yearly." '
        'If you ask, "How many days do I get?", RAG finds this text.'
    )
    sentences = extract_sentences(text)
    assert len(sentences) == 3
    assert all(len(sentence.split()) <= 25 for sentence in sentences)


def test_example_section_requires_a_concrete_number():
    good = "## Practical example: vectors\nA student's marks are [72, 85, 90] across three subjects."
    bad = "## Practical example: vectors\nA vector is like a student's marks in three subjects."
    assert check_example_has_numbers(good, "Practical example: vectors").passed
    assert not check_example_has_numbers(bad, "Practical example: vectors").passed


def test_jargon_glossary_and_appositive_count_as_definitions():
    glossary = "Vector store: A database that groups text by meaning."
    assert check_jargon_definitions(glossary).passed is True

    quoted = "This untrue making up is called 'hallucinations' (meaning false answers)."
    assert check_jargon_definitions(quoted).passed is True

    appositive = "The model stores a number list (embedding) that shows its meaning."
    result = check_jargon_definitions(appositive)
    assert all("embedding" not in violation for violation in result.violations)

    reverse = "RAG helps reduce wrong answers called hallucinations from AI."
    result = check_jargon_definitions(reverse)
    assert all("hallucination" not in violation for violation in result.violations)


def test_jargon_without_definition():
    bad_text = """
    We put all our text into a vector database today.
    The embeddings are stored on disk.
    """
    result = check_jargon_definitions(bad_text)
    assert result.passed is False
    assert any("vector database" in v for v in result.violations)


def test_jargon_definition_must_follow_the_first_use():
    text = "The vector database stores notes. Later, the section heading uses a colon: explanation follows."
    result = check_jargon_definitions(text)
    assert result.passed is False
    assert any("vector database" in v for v in result.violations)


def test_jargon_definition_immediately_after_term_passes():
    result = check_jargon_definitions("A vector database is a digital library index.")
    assert result.passed is True


def test_citation_grounding_valid_and_invalid():
    source_pack = {
        "sources": [
            {"url": "https://geeksforgeeks.org/rag-basics", "title": "RAG Basics"}
        ]
    }
    valid_text = "Learn more at [GFG](https://geeksforgeeks.org/rag-basics)."
    invalid_text = "Learn more at [Untrusted](https://untrusted-fake-site.com/rag)."

    assert check_citation_grounding(valid_text, source_pack).passed is True
    assert check_citation_grounding(invalid_text, source_pack).passed is False


def test_source_grounding_fails_on_offline_stub():
    from src.nodes.eval_native import check_source_grounding
    offline_pack = {"source_tool": "offline_stub", "retrieval_details": {"fallback_reason": ["All backends down."]}}
    result = check_source_grounding(offline_pack)
    assert result.passed is False
    assert "offline stub" in result.message


def test_source_grounding_passes_on_grounded_source():
    from src.nodes.eval_native import check_source_grounding
    grounded_pack = {"source_tool": "tavily", "retrieval_details": {}}
    result = check_source_grounding(grounded_pack)
    assert result.passed is True
