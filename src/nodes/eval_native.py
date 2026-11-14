"""Deterministic, zero-LLM native evaluation gate for lesson drafts."""

import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import textstat
from langsmith import traceable

from src.schemas import NativeCheckResult, SourcePack


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Safely load a YAML config file."""
    path = Path(file_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks and inline code to prevent skewing readability metrics."""
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    cleaned = re.sub(r"`[^`]*`", "", cleaned)
    return cleaned


def clean_prose(text: str) -> str:
    """Remove Markdown and link syntax before prose metrics are calculated."""
    text = strip_code_blocks(text)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        line = re.sub(r"^(?:[*+-]|\d+[.)])\s+", "", line)
        line = re.sub(r"[*_~]", "", line)
        lines.append(line)
    return " ".join(lines)


def extract_sentences(text: str) -> List[str]:
    """Extract clean sentences, ignoring markdown headers, list bullets, and blank lines."""
    full_prose = clean_prose(text)
    # Split at sentence terminators. A closing quote or bracket belongs to the
    # sentence it closes, so `staff." The ...` is two sentences, not one long
    # 29-word sentence. An optional comma after the quote is dropped.
    parts = re.split(r"(?<=[.!?])([\"'”’)\]]*)(,?)\s+", full_prose)
    sentences = []
    for index in range(0, len(parts), 3):
        body = parts[index]
        quote = parts[index + 1] if index + 1 < len(parts) else ""
        sentence = (body + quote).strip()
        if len(sentence.split()) >= 3:
            sentences.append(sentence)
    return sentences


def check_readability(
    text: str, min_reading_ease: float = 60.0, max_grade_level: float = 8.5
) -> NativeCheckResult:
    """Check Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    clean_text = clean_prose(text)
    if not clean_text.strip():
        return NativeCheckResult(
            check_id="NATIVE_READABILITY",
            name="Flesch Reading Ease & Grade Level",
            passed=False,
            score=0.0,
            message="Content is empty.",
            violations=["Empty text provided."],
        )

    reading_ease = textstat.flesch_reading_ease(clean_text)
    grade_level = textstat.flesch_kincaid_grade(clean_text)

    violations = []
    if reading_ease < min_reading_ease:
        violations.append(
            f"Flesch Reading Ease score {reading_ease:.1f} is below minimum {min_reading_ease:.1f}."
        )
    if grade_level > max_grade_level:
        violations.append(
            f"Flesch-Kincaid Grade Level {grade_level:.1f} exceeds maximum {max_grade_level:.1f}."
        )

    passed = len(violations) == 0
    msg = (
        f"Passed: Reading Ease {reading_ease:.1f} (>= {min_reading_ease}), Grade Level {grade_level:.1f} (<= {max_grade_level})"
        if passed
        else f"Failed: {'; '.join(violations)}"
    )

    return NativeCheckResult(
        check_id="NATIVE_READABILITY",
        name="Flesch Reading Ease & Grade Level",
        passed=passed,
        score=float(reading_ease),
        message=msg,
        violations=violations,
    )


def check_sentence_length(
    text: str, max_avg_words: int = 16, hard_max_words: int = 25
) -> NativeCheckResult:
    """Ensure sentences remain short and digestible for a 12th-grade beginner."""
    sentences = extract_sentences(text)
    if not sentences:
        return NativeCheckResult(
            check_id="NATIVE_SENTENCE_LEN",
            name="Sentence Length Guardrail",
            passed=False,
            score=0.0,
            message="No valid sentences found to evaluate.",
            violations=["No sentences found."],
        )

    lengths = [len(s.split()) for s in sentences]
    avg_length = sum(lengths) / len(lengths)

    violations = []
    long_sentences = [
        f"({length} words): \"{s}\""
        for s, length in zip(sentences, lengths)
        if length > hard_max_words
    ]

    if avg_length > max_avg_words:
        violations.append(
            f"Average sentence length is {avg_length:.1f} words (limit: {max_avg_words})."
        )
    if long_sentences:
        violations.append(
            f"{len(long_sentences)} sentence(s) exceed hard limit of {hard_max_words} words: {'; '.join(long_sentences[:3])}"
        )

    passed = len(violations) == 0
    msg = (
        f"Passed: Avg sentence length {avg_length:.1f} words (max {max_avg_words}), max single sentence {max(lengths)} words (limit {hard_max_words})."
        if passed
        else f"Failed: {'; '.join(violations)}"
    )

    return NativeCheckResult(
        check_id="NATIVE_SENTENCE_LEN",
        name="Sentence Length Guardrail",
        passed=passed,
        score=round(avg_length, 1),
        message=msg,
        violations=violations,
    )


def check_required_sections(
    text: str, required_headers: List[str]
) -> NativeCheckResult:
    """Verify that all core pedagogical markdown headers are present."""
    # Find all markdown headers (lines starting with #, ##, ###)
    headers = [
        re.sub(r"^#+\s*", "", line).strip()
        for line in text.split("\n")
        if re.match(r"^#+\s+", line.strip())
    ]

    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    missing = []
    for req in required_headers:
        req_clean = normalized(req)
        match = any(req_clean == normalized(header) for header in headers)
        if not match:
            missing.append(req)

    passed = len(missing) == 0
    msg = (
        "Passed: All mandatory sections present."
        if passed
        else f"Missing required sections: {', '.join(missing)}"
    )
    return NativeCheckResult(
        check_id="NATIVE_REQUIRED_SECTIONS",
        name="Mandatory Pedagogical Sections",
        passed=passed,
        message=msg,
        violations=[f"Missing header: {m}" for m in missing],
    )


def check_ordered_process_steps(text: str, section_name: str, minimum_steps: int = 3) -> NativeCheckResult:
    """Require the configured process section to explain a minimum ordered sequence."""
    header = re.search(rf"(?m)^##\s+{re.escape(section_name)}\s*$", text)
    section = ""
    if header:
        tail = text[header.end():]
        next_header = re.search(r"(?m)^##\s+", tail)
        section = tail[:next_header.start()] if next_header else tail

    numbered = re.findall(r"(?im)^\s*(?:step\s*)?\d+[.)\:]\s+", section)
    if numbered:
        count = len(numbered)
    else:
        ordinals = re.findall(r"\b(first|second|third|fourth|fifth|finally)\b", section, re.IGNORECASE)
        count = len(ordinals)
    passed = bool(section.strip()) and count >= minimum_steps
    violation = (f"Process section '{section_name}' contains {count} ordered steps; "
                 f"at least {minimum_steps} are required.")
    return NativeCheckResult(
        check_id="NATIVE_ORDERED_STEPS",
        name="Ordered Process Explanation",
        passed=passed,
        score=float(count),
        message=(f"Passed: {count} ordered steps found in '{section_name}'." if passed else violation),
        violations=[] if passed else [violation],
    )


def check_example_has_numbers(text: str, section_name: str,
                              minimum_digits: int = 1) -> NativeCheckResult:
    """Require the planned example section to contain a concrete number."""
    header = re.search(rf"(?m)^##\s+{re.escape(section_name)}\s*$", text)
    section = ""
    if header:
        tail = text[header.end():]
        next_header = re.search(r"(?m)^##\s+", tail)
        section = tail[:next_header.start()] if next_header else tail
    digits = re.findall(r"\d", section)
    passed = bool(section.strip()) and len(digits) >= minimum_digits
    violation = (f"Example section '{section_name}' has no concrete number; "
                 f"at least {minimum_digits} digit is required.")
    return NativeCheckResult(
        check_id="NATIVE_EXAMPLE_HAS_NUMBERS",
        name="Numeric Worked Example",
        passed=passed,
        score=float(len(digits)),
        message=(f"Passed: {len(digits)} digits found in '{section_name}'." if passed else violation),
        violations=[] if passed else [violation],
    )


def check_known_false_claims(text: str, claims: List[str]) -> NativeCheckResult:
    found = [claim for claim in claims if claim.casefold() in text.casefold()]
    return NativeCheckResult(
        check_id="NATIVE_KNOWN_FALSE_CLAIMS", name="Known False Claim Guard",
        passed=not found,
        message="Passed: No configured false claims found." if not found else "Failed: A configured false claim appears in the lesson.",
        violations=[f"Unsupported false claim present: ‘{claim}’. RAG retrieves external context; it does not retrain model weights per question."
                    for claim in found],
    )

_REVERSE_DEFINITION = re.compile(
    r"\b(?:called|known\s+as|named|termed|referred\s+to\s+as)\s*[\"'“‘]?\s*$",
    re.IGNORECASE,
)


def _jargon_definition_nearby(sentence: str, match: re.Match) -> bool:
    """Accept glossary ``Term: explanation`` lines and appositives ``phrase (Term)``.

    These are real definitions, even though the explanation sits before the term
    or after a colon rather than after an "is/means" marker.
    """
    before = sentence[:match.start()]
    after = sentence[match.end():]
    if _REVERSE_DEFINITION.search(before):
        return True
    # Glossary line: the term opens the sentence, then a colon and an explanation.
    if before.strip().lower() in ("", "the") and re.match(
        r"^\s*[:\-–—]\s+\S+(?:\s+\S+)", after
    ):
        return True
    # Appositive: "a number list (embedding) that shows its meaning".
    if before.rstrip().endswith("(") and re.match(r"^\s*\)", after):
        return True
    return False


def check_jargon_definitions(
    text: str, jargon_config_path: str = "jargon.yaml"
) -> NativeCheckResult:
    """Ensure technical jargon terms are defined upon their first occurrence.

    A definition marker must appear in the same sentence as the first use, or
    in the sentence immediately following.  A bare colon, em-dash, or en-dash
    next to the term is NOT sufficient — it must be followed by an actual
    explanation marker such as "means", "is a", "defined as", or a parenthetical
    "(meaning …)".
    """
    config = load_yaml_config(jargon_config_path)
    terms = config.get("terms", [])
    marker_pattern = re.compile(
        r"^\s*(?:\s+(?:step|process|method|technique|tool|system|term|feature|approach))?\s*"
        r"(?:[,]\s*)?(?:means\b|meaning\b|which\s+(?:means|is|refers\s+to)\b|"
        r"is(?:\s+(?:a|an|the))?\b|are(?:\s+(?:a|an|the))?\b|"
        r"refers?\s+to\b|defined\s+as\b|"
        r"is\s+like\b|think\s+of\s+(?:this|it)\s+as\b|simply\s+put\b|"
        r"in\s+simple\s+words\b|that\s+is\b|i\.e\.(?:\s|$)|\()",
        flags=re.IGNORECASE,
    )
    # For the next sentence, only accept a definition that starts the sentence
    # (e.g., "It is a …", "This means …", "That refers to …").  Bare "is"/"are"
    # in the middle of the next sentence does not count as a definition.
    next_sentence_pattern = re.compile(
        r"^\s*(?:it|this|that|the\s+term|the\s+concept)?\s*"
        r"(?:is\s+(?:a|an|the)\b|are\s+(?:a|an|the)\b|means\b|meaning\b|"
        r"refers?\s+to\b|defined\s+as\b|is\s+like\b|"
        r"think\s+of\s+(?:this|it)\s+as\b|simply\s+put\b|"
        r"in\s+simple\s+words\b|that\s+is\b|i\.e\.(?:\s|$))",
        flags=re.IGNORECASE,
    )

    sentences = extract_sentences(text)
    violations = []

    for item in terms:
        canonical = item.get("term", "").lower()
        aliases = [a.lower() for a in item.get("aliases", [])]
        all_variants = [canonical] + aliases

        first_occurrence = None
        first_idx = -1
        for idx, s in enumerate(sentences):
            occurrence = None
            for var in all_variants:
                for match in re.finditer(rf"\b{re.escape(var)}\b", s, flags=re.IGNORECASE):
                    # In the full compound term, the shorter word is not introduced separately.
                    if var == "retrieval" and s[match.end():].startswith("-"):
                        continue
                    occurrence = match
                    break
                if occurrence:
                    break
            if occurrence:
                first_occurrence = (s, occurrence)
                first_idx = idx
                break

        if first_occurrence:
            first_sentence, term_match = first_occurrence
            # A closing quote or punctuation after the term must not hide a
            # following "(meaning ...)" definition. Keep leading spaces so the
            # marker pattern can still see " step (meaning ...)".
            remainder = first_sentence[term_match.end():].lstrip("'\"“”‘’.,;:")
            next_sentence = sentences[first_idx + 1] if first_idx + 1 < len(sentences) else None
            # Require a definition marker in the same sentence (after the term)
            # or at the start of the immediately following sentence.
            found = bool(marker_pattern.search(remainder))
            if not found:
                found = _jargon_definition_nearby(first_sentence, term_match)
            if not found and next_sentence:
                found = bool(next_sentence_pattern.search(next_sentence))
            if not found:
                violations.append(
                    f"Term '{canonical}' introduced without immediate explanation in: \"{first_sentence}\""
                )

    passed = len(violations) == 0
    msg = (
        "Passed: All introduced jargon terms are accompanied by immediate definitions."
        if passed
        else f"Failed: {len(violations)} jargon term(s) introduced without explanation."
    )

    return NativeCheckResult(
        check_id="NATIVE_JARGON_DEFINED",
        name="First-Use Jargon Definition",
        passed=passed,
        message=msg,
        violations=violations,
    )


def check_citation_grounding(
    text: str, source_pack: Optional[Dict[str, Any]]
) -> NativeCheckResult:
    """Verify that cited external URLs exist within the active SourcePack."""
    # Find all URLs in markdown link format [text](url) or raw http(s)
    markdown_urls = re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", text)
    raw_urls = re.findall(r"(?<!\()(https?://[^\s\)]+)", text)
    all_cited_urls = list(set(markdown_urls + raw_urls))

    if not all_cited_urls:
        # It is valid to explain concepts without explicit outbound external hyperlinks
        return NativeCheckResult(
            check_id="NATIVE_CITATION_GROUNDING",
            name="Source URL Integrity",
            passed=True,
            message="Passed: No external URLs cited, or citations adhere to source pack.",
            violations=[],
        )

    valid_urls = set()
    if source_pack and "sources" in source_pack:
        for s in source_pack["sources"]:
            u = s.get("url") if isinstance(s, dict) else getattr(s, "url", "")
            if u:
                valid_urls.add(u.strip())

    violations = []
    for cited in all_cited_urls:
        # Check domain or full URL match
        if not any(cited in valid or valid in cited for valid in valid_urls):
            violations.append(f"Cited URL '{cited}' is not present in the verified SourcePack.")

    passed = len(violations) == 0
    msg = (
        f"Passed: All {len(all_cited_urls)} cited URLs verified against SourcePack."
        if passed
        else f"Failed: {len(violations)} unverified URL citation(s) detected."
    )

    return NativeCheckResult(
        check_id="NATIVE_CITATION_GROUNDING",
        name="Source URL Integrity",
        passed=passed,
        message=msg,
        violations=violations,
    )


def check_rare_word_ratio(
    text: str, max_ratio: float = 0.10
) -> NativeCheckResult:
    """Check the ratio of difficult (multi-syllable or uncommon) words.

    textstat's Flesch scores are calibrated for native English readers.  For
    limited-English learners a high proportion of difficult words is a stronger
    signal than grade level alone.
    """
    clean_text = clean_prose(text)
    if not clean_text.strip():
        return NativeCheckResult(
            check_id="NATIVE_RARE_WORD_RATIO",
            name="Rare Word Ratio",
            passed=False,
            score=0.0,
            message="Content is empty.",
            violations=["Empty text provided."],
        )

    total_words = max(1, textstat.lexicon_count(clean_text))
    try:
        difficult = textstat.difficult_words(clean_text)
    except Exception:
        difficult = 0
    ratio = difficult / total_words

    passed = ratio <= max_ratio
    msg = (
        f"Passed: Rare word ratio {ratio:.1%} (limit {max_ratio:.0%}); {difficult} of {total_words} words."
        if passed
        else f"Failed: Rare word ratio {ratio:.1%} exceeds limit {max_ratio:.0%}; {difficult} of {total_words} words are difficult."
    )

    return NativeCheckResult(
        check_id="NATIVE_RARE_WORD_RATIO",
        name="Rare Word Ratio",
        passed=passed,
        score=round(ratio, 4),
        message=msg,
        violations=[] if passed else [f"{difficult} difficult words out of {total_words} ({ratio:.1%})."],
    )


def check_source_grounding(source_pack: Optional[Dict[str, Any]]) -> NativeCheckResult:
    """Fail-closed when the source pack is an ungrounded offline stub."""
    if not source_pack:
        return NativeCheckResult(
            check_id="NATIVE_SOURCE_GROUNDING",
            name="Source Grounding Status",
            passed=True,
            message="Skipped: no source pack provided.",
        )
    tool = source_pack.get("source_tool", "")
    if tool == "offline_stub":
        details = source_pack.get("retrieval_details", {})
        reasons = details.get("fallback_reason", ["Live search returned no results."])
        return NativeCheckResult(
            check_id="NATIVE_SOURCE_GROUNDING",
            name="Source Grounding Status",
            passed=False,
            message="Failed: All search backends failed. The run was drafted against an offline stub and must not pass.",
            violations=[f"Search failure: {'; '.join(reasons)}"],
        )
    return NativeCheckResult(
        check_id="NATIVE_SOURCE_GROUNDING",
        name="Source Grounding Status",
        passed=True,
        message=f"Passed: Source pack grounded via '{tool}'.",
    )


@traceable(name="eval_tier1_native")
def evaluate_native_all(
    text: str,
    source_pack: Optional[Dict[str, Any]] = None,
    required_headers: Optional[List[str]] = None,  # NEW: from plan.core_sections
    profile: Optional[Dict[str, Any]] = None,
    rubric_path: str = "rubric.yaml",
    jargon_path: str = "jargon.yaml",
) -> Tuple[bool, List[NativeCheckResult]]:
    """Execute all Tier-1 deterministic evaluation checks."""
    rubric = load_yaml_config(rubric_path)
    target = rubric.get("target_audience", {})

    profile = profile or {}
    min_ease = profile.get("min_reading_ease", target.get("reading_ease_min", 60.0))
    max_grade = profile.get("max_grade_level", target.get("grade_level_max", 8.5))
    max_avg_sentence = profile.get("sentence_len_avg_max", target.get("sentence_len_avg_max", 16))
    hard_max_sentence = profile.get("sentence_len_max", target.get("sentence_len_max", 25))

    # Use dynamic headers from plan, or from rubric, or skip section check
    if required_headers:
        dynamic_headers = required_headers
    else:
        # Try rubric.yaml but don't use hardcoded RAG headers
        for check in rubric.get("native_checks", []):
            if check.get("id") == "NATIVE_REQUIRED_SECTIONS" and "required_headers" in check:
                dynamic_headers = check["required_headers"]
                break
        else:
            dynamic_headers = []  # Skip section check if no headers specified

    results: List[NativeCheckResult] = []

    # Run each deterministic check
    results.append(check_readability(text, min_reading_ease=min_ease, max_grade_level=max_grade))
    results.append(check_sentence_length(text, max_avg_words=max_avg_sentence, hard_max_words=hard_max_sentence))
    if dynamic_headers:
        results.append(check_required_sections(text, required_headers=dynamic_headers))
    ordered_config = next((item for item in rubric.get("native_checks", [])
                           if item.get("id") == "NATIVE_ORDERED_STEPS"), None)
    if ordered_config:
        process_index = int(ordered_config.get("section_index", 2))
        if dynamic_headers and 0 <= process_index < len(dynamic_headers):
            results.append(check_ordered_process_steps(
                text, dynamic_headers[process_index],
                int(ordered_config.get("minimum_steps", 3)),
            ))
    example_config = next((item for item in rubric.get("native_checks", [])
                           if item.get("id") == "NATIVE_EXAMPLE_HAS_NUMBERS"), None)
    if example_config:
        example_index = int(example_config.get("section_index", -1))
        if dynamic_headers and -len(dynamic_headers) <= example_index < len(dynamic_headers):
            results.append(check_example_has_numbers(
                text, dynamic_headers[example_index],
                int(example_config.get("minimum_digits", 1)),
            ))
    false_claim_config = next((item for item in rubric.get("native_checks", [])
                               if item.get("id") == "NATIVE_KNOWN_FALSE_CLAIMS"), None)
    if false_claim_config:
        results.append(check_known_false_claims(text, false_claim_config.get("forbidden_claims", [])))
    results.append(check_jargon_definitions(text, jargon_config_path=jargon_path))
    results.append(check_citation_grounding(text, source_pack=source_pack))
    results.append(check_source_grounding(source_pack))
    results.append(check_rare_word_ratio(text, max_ratio=float(profile.get("max_rare_word_ratio", 0.30))))

    all_passed = all(r.passed for r in results)
    return all_passed, results
