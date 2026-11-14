"""Small deterministic input guard for empty and obvious keyboard-noise topics."""

import re


def topic_error(topic: str) -> str | None:
    text = " ".join(topic.split())
    if not text:
        return "Enter a topic to start a lesson."
    if len(text) < 3:
        return "Topic is too short. Add a few words or a recognized acronym."
    if len(text) > 180:
        return "Topic is too long. Keep it under 180 characters."
    tokens = re.findall(r"[A-Za-z]+", text)
    if len(tokens) >= 2:
        def keyboard_noise(token: str) -> bool:
            if token.isupper() or len(token) <= 2:
                return False
            lower = token.casefold()
            consonant_run = re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", lower)
            odd_q = lower.startswith("q") and len(lower) > 1 and lower[1] != "u" and lower not in {"qwerty", "qwert", "qatar"}
            return bool(consonant_run or odd_q)
        if all(keyboard_noise(token) for token in tokens):
            return "That topic looks like keyboard noise. Enter a recognizable subject or question."
    return None
