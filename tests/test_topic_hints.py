"""Topic hints must stay scoped to matching topics."""

from src.topic_hints import load_topic_hints


def test_rag_topic_receives_pipeline_hints():
    hints = load_topic_hints("Introduction to RAG")
    assert "Retrieval-Augmented Generation" in hints
    assert "chunking" in hints


def test_unrelated_topic_receives_no_hints():
    assert load_topic_hints("How does UPI work?") == ""
    assert load_topic_hints("Explain photosynthesis") == ""


def test_tensor_topic_gets_a_precise_definition_hint():
    hints = load_topic_hints("what's a vector, matrix, tensor?")
    assert "array of numbers with any number of dimensions" in hints
    assert "ordered process section" in hints


def test_rag_hint_defines_vector_store_before_use():
    hints = load_topic_hints("Introduction to RAG")
    assert "vector store" in hints


def test_hint_keyword_does_not_match_inside_another_word():
    # "storage" contains the letters r-a-g but is not the RAG topic.
    assert load_topic_hints("data storage") == ""
