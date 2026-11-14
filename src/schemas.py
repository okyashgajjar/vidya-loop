"""Data schemas and types for the Self-Evaluating Lesson Content Generator."""

from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator

OLD_DEFAULT_CULTURAL_CONTEXT = (
    "Relatable Indian analogies (open-book board exams, kirana store ledgers, library index cards)"
)
DEFAULT_CULTURAL_CONTEXT = (
    "Everyday Indian examples (cricket scores, kirana shop bills, bus or train timetables, "
    "UPI payments, school marks). Use one only if it matches the concept exactly."
)


def _migrate_cultural_context(value: str) -> str:
    """Replace only the old stock default; keep any custom learner value."""
    return DEFAULT_CULTURAL_CONTEXT if value == OLD_DEFAULT_CULTURAL_CONTEXT else value


class UserProfile(BaseModel):
    """User-selectable learner profile with customizable constraints."""
    profile_id: str = "default"
    name: str = "12th Grade Indian Student"
    target_audience: str = "12th-grade graduate from India, limited English vocabulary, non-English-medium background"
    education_background: str = "12th-grade graduate"
    expertise_level: Literal["beginner", "intermediate", "expert"] = "beginner"
    max_grade_level: float = 8.5
    min_reading_ease: float = 60.0
    sentence_len_max: int = 25
    sentence_len_avg_max: int = 16
    max_rare_word_ratio: float = 0.30
    cultural_context: str = DEFAULT_CULTURAL_CONTEXT

    @field_validator("cultural_context")
    @classmethod
    def _migrate_old_default(cls, value: str) -> str:
        return _migrate_cultural_context(value)


class ChatMessage(BaseModel):
    """Single message in a conversation session."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    topic: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """One isolated learning conversation and its latest pipeline result."""
    session_id: str
    session_source_pack: Dict[str, Any] = Field(default_factory=dict)
    title: str = "New lesson"
    profile_id: str = "default"
    messages: List[ChatMessage] = Field(default_factory=list)
    latest_run: Dict[str, Any] = Field(default_factory=dict)
    source_pack: Dict[str, Any] = Field(default_factory=dict)


class LearnerProfile(BaseModel):
    """Profile describing the target learner persona and constraints."""
    target_audience: str = "12th-grade graduate from India, limited English vocabulary, non-English-medium background"
    max_grade_level: float = 8.5
    min_reading_ease: float = 60.0
    sentence_len_max: int = 25
    sentence_len_avg_max: int = 16
    max_rare_word_ratio: float = 0.30
    cultural_context: str = DEFAULT_CULTURAL_CONTEXT

    @field_validator("cultural_context")
    @classmethod
    def _migrate_old_default(cls, value: str) -> str:
        return _migrate_cultural_context(value)


class SourceItem(BaseModel):
    """An individual grounding search result."""
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    raw_content: Optional[str] = None


class SourcePack(BaseModel):
    """Pack of verified grounding sources for lesson synthesis."""
    topic: str
    source_tool: str  # "tavily", "ddgs", or "cache"
    sources: List[SourceItem] = Field(default_factory=list)
    fetched_at: str
    retrieval_version: int = 1
    retrieval_details: Dict[str, Any] = Field(default_factory=dict)


class LessonPlan(BaseModel):
    """Pedagogical outline generated dynamically per topic."""
    topic: str = ""
    target_audience: str = ""
    core_sections: List[str] = Field(default_factory=list)
    key_analogies: List[str] = Field(default_factory=list)
    must_cover_points: List[str] = Field(default_factory=list)
    avoid_pitfalls: List[str] = Field(default_factory=list)


class DraftLesson(BaseModel):
    """Synthesized lesson draft."""
    title: str
    content: str
    word_count: int = 0
    cited_urls: List[str] = Field(default_factory=list)


class NativeCheckResult(BaseModel):
    """Evaluation result for a deterministic, zero-LLM gate check."""
    check_id: str
    name: str
    passed: bool
    score: Optional[float] = None
    message: str
    violations: List[str] = Field(default_factory=list)


class JudgeCheckResult(BaseModel):
    """Evaluation result for an LLM-as-a-judge binary check."""
    check_id: str
    name: str
    passed: bool
    reason: str
    evidence_quote: Optional[str] = None
    source_quote: Optional[str] = None
    missing_points: List[str] = Field(default_factory=list)
    judge_error: bool = False  # True when the judge itself failed, not the lesson.


class EvalReport(BaseModel):
    """Combined report across native and judge evaluation tiers."""
    attempt: int
    native_passed: bool
    native_results: List[NativeCheckResult] = Field(default_factory=list)
    judge_passed: bool
    judge_results: List[JudgeCheckResult] = Field(default_factory=list)
    overall_passed: bool


class RejectionLogEntry(BaseModel):
    """Audit entry documenting why a generation attempt was rejected."""
    attempt: int
    stage_failed: str  # "native" or "judge"
    failed_check_ids: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    suggested_fix: str
    revised_sections: List[str] = Field(default_factory=list)
    source_evidence: List[Dict[str, str]] = Field(default_factory=list)
    revision_diff: str = ""


class SystemMemory(BaseModel):
    """Persistent memory across runs stored in memory.json."""
    system_version: str = "1.0"
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0
    failure_counters: Dict[str, int] = Field(default_factory=dict)
    failure_lessons: Dict[str, List[str]] = Field(default_factory=dict)
    feedback_reasons: List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    """LangGraph state representation passed through the pipeline."""
    topic: str
    learner_profile: Dict[str, Any]
    attempt: int                     # One-based total attempt number; MAX_ATTEMPTS includes the first draft.
    source_pack: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    draft: Optional[Dict[str, Any]]
    native_eval_passed: bool
    native_results: List[Dict[str, Any]]
    judge_eval_passed: bool
    judge_results: List[Dict[str, Any]]
    judge_unavailable: bool
    rejection_log: List[Dict[str, Any]]
    revision_sections: List[str]
    revision_source_content: str
    memory_lessons: List[str]
    final_status: str                # "passed", "failed", "in_progress"
    inject_error: Optional[str]      # For testing self-healing loops
    required_sections: List[str]     # plan.core_sections passed to eval
    must_cover_points: List[str]     # plan.must_cover_points passed to judge
    session_id: str
    session_source_pack: Dict[str, Any]
    profile_id: str
    chat_history: List[Dict[str, Any]]
    is_followup: bool                # True if follow-up on same topic
    semantic_context: Dict[str, Any]
