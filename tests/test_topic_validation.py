from src.topic_validation import topic_error


def test_empty_and_keyboard_noise_topics_return_clear_errors():
    assert topic_error(" ")
    assert topic_error("asdkj qwe")
    assert topic_error("Introduction to RAG") is None
    assert topic_error("AI and ML") is None
