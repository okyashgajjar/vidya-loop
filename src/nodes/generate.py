"""Lesson generation node: synthesizes draft lesson from plan and grounded sources."""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langsmith import traceable
from pydantic import BaseModel, Field
from src.schemas import DraftLesson, LessonPlan
from src.llm import call_llm, call_llm_json, DEFAULT_GEN_MODEL, DEFAULT_MAX_TOKENS
from src.nodes.plan import load_prompt_template, build_source_context
from src.topic_hints import load_topic_hints


class SectionRevisionOutput(BaseModel):
    sections: Dict[str, str] = Field(default_factory=dict)


def revise_lesson_sections(topic: str, plan: Dict[str, Any], source_pack: Dict[str, Any],
                           previous_content: str, target_sections: List[str], reasons: List[str],
                           learner_profile: Optional[Dict[str, Any]] = None,
                           model: str = DEFAULT_GEN_MODEL, run_attempt: int = 1,
                           audit_callback=None) -> str:
    """Rewrite only failed sections, then merge them into the previously reviewed draft."""
    all_sections = plan.get("core_sections", [])
    targets = [name for name in all_sections if name in set(target_sections)]
    if not targets:
        targets = all_sections
    existing = {}
    for section in targets:
        match = re.search(rf"(?m)^##\s+{re.escape(section)}\s*$", previous_content)
        if match:
            end_match = re.search(r"(?m)^(?:##\s+|\*\*Glossary:\*\*)", previous_content[match.end():])
            end = match.end() + end_match.start() if end_match else len(previous_content)
            existing[section] = previous_content[match.end():end].strip()
        else:
            existing[section] = "(Section is missing.)"
    prompt = {
        "topic": topic,
        "target_learner": learner_profile or {"expertise_level": "beginner"},
        "verified_sources": build_source_context(source_pack),
        "approved_plan_points": plan.get("must_cover_points", []),
        "sections_to_revise": targets,
        "current_section_text": existing,
        "failure_reasons": reasons,
        "rules": [
            "Return complete replacement prose for every requested section.",
            "Teach from zero. Use simple language and define technical terms on first use.",
            "Ground factual claims in provided sources. Do not invent URLs.",
            "Use an example and keep sentences short. Preserve the plan heading names exactly.",
            "Return section body text only, without Markdown headings. Do not revise other sections.",
            "Return each section body as Markdown with real line breaks (\\n in the JSON string). "
            "Every numbered or bulleted item goes on its own line. Do not change any section not listed.",
            "If a revised section explains a process, a build-up, or how parts fit together, "
            "keep at least three numbered steps on separate lines starting with \"1.\", \"2.\", \"3.\". "
            "Never replace numbered steps with plain prose.",
        ],
    }
    response = call_llm_json(
        messages=[
            {"role": "system", "content": "Revise only the requested lesson sections. Return JSON with a sections object mapping exact headings to replacement Markdown body text."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        schema=SectionRevisionOutput, model=model, temperature=0.2,
        max_tokens=DEFAULT_MAX_TOKENS["generate"], stage="revision",
        run_attempt=run_attempt, audit_callback=audit_callback,
    )
    content = previous_content
    replacements = {key: value.strip() for key, value in response.sections.items()
                    if key in targets and value.strip()}
    for section in targets:
        body = replacements.get(section)
        if not body:
            raise ValueError(f"Section revision omitted required section: {section}")
        body = re.sub(r"(?m)^#{1,6}\s+[^\n]+\n+", "", body, count=1).strip()
        match = re.search(rf"(?m)^##\s+{re.escape(section)}\s*$", content)
        if match:
            next_match = re.search(r"(?m)^(?:##\s+|\*\*Glossary:\*\*)", content[match.end():])
            end = match.end() + next_match.start() if next_match else len(content)
            content = content[:match.start()] + f"## {section}\n\n{body}\n\n" + content[end:].lstrip()
        else:
            section_index = all_sections.index(section)
            insertion = len(content)
            for following in all_sections[section_index + 1:]:
                next_heading = re.search(rf"(?m)^##\s+{re.escape(following)}\s*$", content)
                if next_heading:
                    insertion = next_heading.start()
                    break
            glossary = re.search(r"(?m)^\*\*Glossary:\*\*", content)
            if glossary and glossary.start() < insertion:
                insertion = glossary.start()
            content = content[:insertion].rstrip() + f"\n\n## {section}\n\n{body}\n\n" + content[insertion:].lstrip()
    return content.strip()


def expand_inline_numbered_lists(markdown: str) -> str:
    """Put inline "1. ... 2. ... 3. ..." items on separate lines.

    Models sometimes flatten a numbered list into one line during a revision.
    This is markup repair only; the evaluator still requires three real steps.
    """
    lines = []
    for line in markdown.split("\n"):
        markers = list(re.finditer(r"(?<!\S)(\d{1,2})[.)]\s+", line))
        if len(markers) < 2:
            lines.append(line)
            continue
        rebuilt = line
        for match in reversed(markers[1:]):
            rebuilt = rebuilt[:match.start()].rstrip() + "\n" + rebuilt[match.start():]
        lines.extend(rebuilt.split("\n"))
    return "\n".join(lines)


def normalize_lesson_format(markdown: str, sections: List[str], topic: str) -> str:
    """Give model output a clean title and align its section labels to the plan."""
    markdown = markdown.strip()
    if not markdown:
        raise ValueError("The model returned an empty lesson.")
    markdown = expand_inline_numbered_lists(markdown)

    # Preserve section body text while making headings match the approved plan.
    section_matches = list(re.finditer(r"(?m)^##\s+.+$", markdown))
    if len(section_matches) < len(sections):
        all_heading_matches = list(re.finditer(r"(?m)^#{2,6}\s+.+$", markdown))
        if len(all_heading_matches) == len(sections):
            section_matches = all_heading_matches
    for index, match in reversed(list(enumerate(section_matches[:len(sections)]))):
        markdown = markdown[:match.start()] + f"## {sections[index]}" + markdown[match.end():]

    title_match = re.search(r"(?m)^#\s+.*$", markdown)
    if title_match:
        markdown = markdown[:title_match.start()] + f"# {topic}" + markdown[title_match.end():]
    else:
        markdown = f"# {topic}\n\n{markdown}"
    return markdown


def strip_preamble_to_first_heading(markdown: str) -> str:
    """Drop model preamble without consuming a heading's leading hash marks."""
    heading = re.search(r"(?m)^#{1,6}\s+[^\n]+", markdown)
    return markdown[heading.start():].strip() if heading else markdown.strip()


def extract_glossary(markdown: str) -> List[Dict[str, str]]:
    """Read the optional unheaded glossary block for persistence after thumbs-up."""
    match = re.search(r"(?im)^\*\*Glossary:\*\*\s*$([\s\S]*?)(?=^#{1,6}\s|\Z)", markdown)
    if not match:
        return []
    entries = []
    for term, definition in re.findall(r"(?m)^\s*-\s*\*\*(.+?):\*\*\s*(.+)$", match.group(1))[:5]:
        entries.append({"term": term.strip(), "definition": definition.strip()})
    return entries


@traceable(name="generate_lesson_draft")
def generate_lesson_draft(
    topic: str,
    plan: Dict[str, Any],
    source_pack: Dict[str, Any],
    memory_lessons: List[str],
    rejection_feedback: Optional[List[Dict[str, Any]]] = None,
    inject_error: Optional[str] = None,
    model: str = DEFAULT_GEN_MODEL,
    stream_callback=None,
    learner_profile: Optional[Dict[str, Any]] = None,
    semantic_context: str = "",
    revision_sections: Optional[List[str]] = None,
    revision_source_content: str = "",
    run_attempt: int = 1,
    audit_callback=None,
) -> DraftLesson:
    """Generate or regenerate full lesson markdown draft."""
    template = load_prompt_template("generate.md")
    source_context = build_source_context(source_pack)
    plan_context = json.dumps(plan, indent=2)
    topic_hints = load_topic_hints(topic) or "None."

    lessons_text = (
        "\n".join([f"- {l}" for l in memory_lessons])
        if memory_lessons
        else "No prior failure lessons recorded yet."
    )
    learner_profile = learner_profile or {}
    learner = (
        f"Education: {learner_profile.get('education_background', 'not specified')}; "
        f"expertise: {learner_profile.get('expertise_level', 'beginner')}; "
        f"learning needs: {learner_profile.get('target_audience', plan.get('target_audience', ''))}"
    )
    sections = plan.get("core_sections", [])

    if rejection_feedback:
        feedback_lines = []
        for item in rejection_feedback:
            stage = item.get("stage_failed", "evaluation")
            reasons = "; ".join(item.get("reasons", []))
            fix = item.get("suggested_fix", "")
            feedback_lines.append(f"Attempt {item.get('attempt')} failed in {stage}: {reasons}. Fix: {fix}")
        feedback_text = "\n".join(feedback_lines)
    else:
        feedback_text = "None. First attempt."

    prompt_content = (
        template.replace("{topic}", topic)
        .replace("{learner_profile}", learner)
        .replace("{plan_sections}", "\n".join(f"   - ## {section}" for section in sections))
        .replace("{plan_context}", plan_context)
        .replace("{source_context}", source_context)
        .replace("{memory_lessons}", lessons_text)
        .replace("{rejection_feedback}", feedback_text)
        .replace("{topic_hints}", topic_hints)
    )
    if semantic_context:
        prompt_content += "\n\nRELATED LEARNING CONTEXT (style and definitions only; facts must come from verified sources):\n" + semantic_context

    messages = [
        {"role": "system", "content": "You are an adaptive teacher. Match explanation depth to the supplied learner profile. Output only clean markdown."},
        {"role": "user", "content": prompt_content},
    ]

    used_revision_fallback = False
    if revision_sections:
        previous_content = revision_source_content
        if not previous_content:
            raise ValueError("Targeted revision requires the previous draft content.")
        reasons = rejection_feedback[-1].get("reasons", []) if rejection_feedback else []
        try:
            raw_markdown = revise_lesson_sections(topic, plan, source_pack, previous_content,
                                                   revision_sections, reasons, learner_profile, model,
                                                   run_attempt, audit_callback)
        except ValueError as exc:
            # A malformed/partial section-only response must not abort the run.
            # Fall back to a complete draft under the same frozen plan/sources
            # and attempt budget. This is still the revision attempt.
            used_revision_fallback = True
            fallback_messages = [
                messages[0],
                {"role": "user", "content": prompt_content +
                 "\n\nREVISION FALLBACK: The section-only revision was incomplete (" + str(exc) +
                 "). Rewrite the complete lesson using the same approved plan and verified sources. "
                 "Address the evaluator feedback above. Do not change the plan or add sections."},
            ]
            raw_markdown = call_llm(
                messages=fallback_messages, model=model, temperature=0.3,
                max_tokens=DEFAULT_MAX_TOKENS["generate"], stream_callback=stream_callback,
                stage="revision_fallback", run_attempt=run_attempt,
                audit_callback=audit_callback,
            )
    else:
        raw_markdown = call_llm(messages=messages, model=model, temperature=0.3,
                                max_tokens=DEFAULT_MAX_TOKENS["generate"],
                                stream_callback=stream_callback, stage="generate",
                                run_attempt=run_attempt, audit_callback=audit_callback)

    # Strip chain-of-thought preamble before first markdown header if present
    raw_markdown = strip_preamble_to_first_heading(raw_markdown)
    raw_markdown = normalize_lesson_format(raw_markdown, sections, plan.get("topic") or topic)
    if used_revision_fallback:
        # Keep the complete rewrite anchored to the exact plan headings.
        missing = [section for section in sections if not re.search(
            rf"(?m)^##\s+{re.escape(section)}\s*$", raw_markdown)]
        if missing:
            raise ValueError("Full revision fallback omitted planned sections: " + ", ".join(missing))

    # Optional error injection for verifying retry & self-healing loops
    if inject_error == "long_sentence":
        raw_markdown += (
            "\n\nIn addition to the aforementioned principles, the multi-layered neural architecture "
            "synthesizes highly intricate mathematical representations across non-linear latent spaces "
            "which subsequently interact with distributed high-dimensional vector embeddings simultaneously "
            "thereby establishing an extraordinarily complex contextual correlation matrix throughout the network."
        )
    elif inject_error == "missing_section":
        # Remove a required header
        if sections:
            raw_markdown = re.sub(rf"##\s*{re.escape(sections[-1])}.*?(?=##|$)", "", raw_markdown, flags=re.DOTALL | re.IGNORECASE)
    elif inject_error == "factual_jargon":
        process_section = sections[2] if len(sections) > 2 else (sections[-1] if sections else "")
        match = re.search(rf"(?m)^##\s+{re.escape(process_section)}\s*$", raw_markdown)
        if match:
            next_section = re.search(r"(?m)^##\s+", raw_markdown[match.end():])
            end = match.end() + next_section.start() if next_section else len(raw_markdown)
            bad_claim = "\n\nRAG retrains the model's weights every time a question is asked."
            raw_markdown = raw_markdown[:end].rstrip() + bad_claim + "\n\n" + raw_markdown[end:].lstrip()
    elif inject_error == "paraphrased_falsehood":
        # A reworded claim that the native substring matcher cannot catch.
        # The LLM judge should flag it via JUDGE_ACCURACY_GROUNDING with a
        # quoted source-pack line as evidence.
        intro_section = sections[0] if sections else ""
        match = re.search(rf"(?m)^##\s+{re.escape(intro_section)}\s*$", raw_markdown)
        if match:
            next_section = re.search(r"(?m)^##\s+", raw_markdown[match.end():])
            end = match.end() + next_section.start() if next_section else len(raw_markdown)
            bad_claim = "\n\nRAG permanently teaches the model every new document you add."
            raw_markdown = raw_markdown[:end].rstrip() + bad_claim + "\n\n" + raw_markdown[end:].lstrip()

    # Extract title from first markdown header
    title_match = re.search(r"^#\s+(.+)$", raw_markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else f"{topic} for Beginners"

    word_count = len(raw_markdown.split())
    cited_urls = re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", raw_markdown)

    return DraftLesson(
        title=title,
        content=raw_markdown,
        word_count=word_count,
        cited_urls=list(set(cited_urls)),
    )
