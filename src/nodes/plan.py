"""Lesson planning node: creates pedagogical outline and must-cover points."""

from pathlib import Path
import re
import json
import hashlib
import os
from typing import Dict, Any, List, Optional
from langsmith import traceable
from src.schemas import LessonPlan
from src.llm import call_llm_json, DEFAULT_PLAN_MODEL, DEFAULT_MAX_TOKENS
from src.topic_hints import load_topic_hints


_LEAD = re.compile(
    r"^\s*(?:"
    r"what(?:'s|’s|s| is| are)\s+the\s+differences?\s+between|"
    r"difference\s+between|"
    r"what(?:'s|’s|s| is| are)|"
    r"explain|tell\s+me\s+about|introduction\s+to|intro\s+to"
    r")\s+(?:an?\s+|the\s+)?",
    re.IGNORECASE,
)
_SUBJECT_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bvs\.?\b|\bversus\b)\s*", re.IGNORECASE)
_SUBJECT_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def split_subjects(topic: str) -> List[str]:
    """Split a comma/and/vs list into individual subjects."""
    parts = [_SUBJECT_ARTICLE.sub("", part.strip()).strip()
             for part in _SUBJECT_SPLIT.split(topic) if part.strip()]
    return [part for part in parts if part]


def join_subjects(parts: List[str]) -> str:
    """Render a subject list as 'vector, matrix and tensor'."""
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def normalize_topic(topic: str) -> str:
    """Reduce common questions, contractions, and comparison lists to a clean subject."""
    cleaned = topic.strip().rstrip(" ?.!:")
    # Remove an appended comparison clause, for example ", how they different".
    cleaned = re.sub(
        r"\s*[,;]?\s*(?:how\s+(?:are\s+)?they\s+different.*|give\s+me\s+(?:a\s+)?difference.*|"
        r"compare(?:\s+them)?(?:\s+.*)?)$",
        "", cleaned, flags=re.IGNORECASE,
    ).strip(" ,;:")

    pair = re.match(r"^what\s+is\s+(.+?)\s*(?:&|and)\s*what\s+is\s+(.+)$", cleaned, re.IGNORECASE)
    if pair:
        return join_subjects([pair.group(1).strip(), pair.group(2).strip()]).rstrip(" ?.!:")

    cleaned = _LEAD.sub("", cleaned, count=1).strip()
    patterns = (
        r"^how\s+does\s+(.+?)\s+work$",
        r"^how\s+do\s+(.+?)\s+work$",
        r"^why\s+does\s+(.+?)\s+matter$",
        r"^how\s+do\s+(.+?)\s+fit\s+together$",
    )
    for pattern in patterns:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
            break
    cleaned = cleaned.replace("&", "and").strip(" ?.!:")
    subjects = split_subjects(cleaned)
    return join_subjects(subjects) if len(subjects) > 1 else (cleaned or topic.strip())


def canonical_sections(topic: str) -> List[str]:
    """Use a stable pedagogical outline across every generation retry."""
    subject = normalize_topic(topic)
    subjects = split_subjects(subject)
    if len(subjects) >= 2:
        joined = join_subjects(subjects)
        return [
            f"What are {joined}?",
            f"Why do {joined} matter?",
            f"How do {joined} fit together?",
            f"Practical example: {joined}",
        ]
    return [
        f"What is {subject}?",
        f"Why does {subject} matter?",
        f"How does {subject} work?",
        f"Practical example: {subject}",
    ]


def load_prompt_template(template_name: str) -> str:
    """Load prompt markdown file."""
    path = Path("prompts") / template_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template {template_name} not found in prompts/")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_source_context(source_pack: Dict[str, Any], cited_urls: Optional[List[str]] = None) -> str:
    """Format source pack into clean text for prompt context.

    When *cited_urls* is provided, sources whose URL appears in the list are
    guaranteed to be included even if they fall beyond ``PROMPT_SOURCE_COUNT``,
    so the judge can verify every citation the draft makes.
    """
    sources = source_pack.get("sources", [])
    if not sources:
        return "No external sources available."

    max_sources = max(1, int(os.getenv("PROMPT_SOURCE_COUNT", "3")))
    excerpt_chars = max(300, int(os.getenv("PROMPT_SOURCE_CHARS", "1200")))

    # Start with the first N sources, then append any cited sources that were
    # beyond the window so the judge can verify them.
    selected = list(sources[:max_sources])
    if cited_urls:
        cited_set = {u.strip().rstrip("/") for u in cited_urls if u}
        already = {s.get("url", "").strip().rstrip("/") for s in selected}
        for s in sources[max_sources:]:
            url = s.get("url", "").strip().rstrip("/")
            if url and url in cited_set and url not in already:
                selected.append(s)
                already.add(url)

    lines = []
    for idx, s in enumerate(selected, start=1):
        title = s.get("title", "Source")
        url = s.get("url", "")
        snippet = (s.get("raw_content") or s.get("snippet", "")).strip()[:excerpt_chars]
        source_label = f"[{title}]({url})" if url else f"[{title}; uploaded learning material]"
        lines.append(f"{idx}. {source_label}: {snippet}")
    return "\n".join(lines)


@traceable(name="plan_lesson")
def plan_lesson(
    topic: str,
    source_pack: Dict[str, Any],
    memory_lessons: List[str],
    learner_profile: Optional[Dict[str, Any]] = None,
    semantic_context: str = "",
    model: str = DEFAULT_PLAN_MODEL,
    run_attempt: int = 1,
    audit_callback=None,
) -> LessonPlan:
    """Generate a structured lesson plan."""
    original_request = topic.strip()
    topic = normalize_topic(original_request)
    sections = canonical_sections(topic)
    template = load_prompt_template("plan.md")
    source_context = build_source_context(source_pack)
    topic_hints = load_topic_hints(original_request) or "None."

    # Deterministic local cache avoids repeat planning calls for identical inputs.
    cache_payload = json.dumps({
        "version": 4, "topic": topic, "request": original_request, "sources": source_context,
        "memory": memory_lessons, "profile": learner_profile or {},
        "semantic": semantic_context, "prompt": template, "hints": topic_hints,
        "model": model,
    }, sort_keys=True, ensure_ascii=False)
    cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
    cache_path = Path(os.getenv("PLAN_CACHE_DIR", "cache/plans")) / f"{cache_key}.json"
    if cache_path.exists():
        try:
            cached = LessonPlan.model_validate_json(cache_path.read_text(encoding="utf-8"))
            cached.core_sections = sections
            return cached
        except Exception:
            cache_path.unlink(missing_ok=True)

    lessons_text = (
        "\n".join([f"- {l}" for l in memory_lessons])
        if memory_lessons
        else "No prior failure lessons recorded yet."
    )

    # Build learner profile text
    if learner_profile:
        profile_text = (
            f"- Target: {learner_profile.get('target_audience', '12th-grade graduate from India')}\n"
            f"- Education background: {learner_profile.get('education_background', 'not specified')}\n"
            f"- Current expertise: {learner_profile.get('expertise_level', 'beginner')}\n"
            f"- Cultural context: {learner_profile.get('cultural_context', 'Indian analogies')}\n"
            f"- Max sentence length: {learner_profile.get('sentence_len_max', 25)} words\n"
            f"- Min reading ease: {learner_profile.get('min_reading_ease', 60.0)}"
        )
    else:
        profile_text = "12th-grade graduate from India with limited English vocabulary. Use Indian analogies."

    prompt_content = (
        template.replace("{topic}", original_request)
        .replace("{source_context}", source_context)
        .replace("{memory_lessons}", lessons_text)
        .replace("{learner_profile}", profile_text)
        .replace("{canonical_sections}", "\n".join(f"- {section}" for section in sections))
        .replace("{canonical_sections_json}", json.dumps(sections, ensure_ascii=False))
        .replace("{topic_hints}", topic_hints)
    )
    if semantic_context:
        prompt_content += "\n\nSEMANTIC CONTEXT (optional; do not treat it as a factual source):\n" + semantic_context

    messages = [
        {"role": "system", "content": "You are an expert AI curriculum designer. Return only a valid JSON object starting with { and ending with }."},
        {"role": "user", "content": prompt_content},
    ]

    try:
        plan: LessonPlan = call_llm_json(messages=messages, schema=LessonPlan, model=model,
                                         temperature=0.3, max_tokens=DEFAULT_MAX_TOKENS["plan"], stage="plan",
                                         run_attempt=run_attempt, audit_callback=audit_callback)
        plan.topic = topic
        # The model supplies teaching points and examples. These exact headings
        # stay fixed so evaluation and retries share one outline.
        plan.core_sections = sections
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return plan
    except Exception as e:
        print(f"[Plan] LLM plan generation error: {e}. Using minimal default plan.")
        return LessonPlan(
            topic=topic,
            target_audience=learner_profile.get('target_audience', '') if learner_profile else '',
            core_sections=sections,
            key_analogies=[],
            must_cover_points=[f"Core concepts of {topic}"],
            avoid_pitfalls=[f"Avoid unexplained jargon when teaching {topic}"],
        )
