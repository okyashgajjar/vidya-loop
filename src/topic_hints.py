"""Topic-specific prompt hints loaded only when the request matches.

Domain facts (RAG pipeline terms, AI/ML distinctions) stay next to the topics
that need them instead of leaking into every lesson.
"""

import re
from pathlib import Path
from typing import List

import yaml

HINTS_DIR = Path("data/topic_hints")


def load_topic_hints(topic: str) -> str:
    """Return matching rule lines, or an empty string when no hint applies."""
    if not topic or not HINTS_DIR.exists():
        return ""
    rules: List[str] = []
    for path in sorted(HINTS_DIR.glob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        pattern = str(config.get("pattern", "")).strip()
        if not pattern:
            continue
        try:
            if not re.search(pattern, topic, re.IGNORECASE):
                continue
        except re.error:
            continue
        for rule in config.get("rules", []):
            text = str(rule).strip()
            if text and text not in rules:
                rules.append(text)
    return "\n".join(f"- {rule}" for rule in rules)
