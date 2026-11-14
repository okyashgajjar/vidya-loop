"""Persistent memory management for failure lessons and human feedback.

Profile-scoped: each profile stores its own memory.json under profiles/{profile_id}/.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

from src.schemas import (
    DEFAULT_CULTURAL_CONTEXT,
    OLD_DEFAULT_CULTURAL_CONTEXT,
    SystemMemory,
    UserProfile,
)

PROFILES_DIR = Path("profiles")
MEMORY_FILE_PATH = "profiles/default/memory.json"

# Style checks are topic-independent: a readability failure on one topic
# teaches the same lesson for every topic.  Content checks (accuracy,
# coverage) are topic-specific and stay keyed by {topic}::{check_id}.
STYLE_CHECKS = frozenset({
    "NATIVE_READABILITY",
    "NATIVE_SENTENCE_LEN",
    "NATIVE_JARGON_DEFINED",
})

# Legacy root memory file (pre-profile migration)
_LEGACY_MEMORY_PATH = Path("memory.json")


def _ensure_profile_dir(profile_id: str) -> Path:
    """Create and return the profile directory."""
    profile_dir = PROFILES_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def _migrate_saved_profile(pfile: Path, data: Dict[str, Any]) -> None:
    """Persist the new cultural-context default, but never overwrite a custom value."""
    if data.get("cultural_context") != OLD_DEFAULT_CULTURAL_CONTEXT:
        return
    data["cultural_context"] = DEFAULT_CULTURAL_CONTEXT
    try:
        pfile.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Memory] Migrated cultural_context in {pfile}")
    except OSError:
        pass


def _migrate_legacy_memory() -> None:
    """One-time migration: move root memory.json into profiles/default/."""
    default_dir = _ensure_profile_dir("default")
    target = default_dir / "memory.json"
    if _LEGACY_MEMORY_PATH.exists() and not target.exists():
        shutil.copy2(str(_LEGACY_MEMORY_PATH), str(target))
        print("[Memory] Migrated legacy memory.json → profiles/default/memory.json")


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

def list_profiles() -> List[UserProfile]:
    """List all saved profiles. Always includes 'default'."""
    profiles: List[UserProfile] = []
    seen_ids: set = set()

    if PROFILES_DIR.exists():
        for pdir in sorted(PROFILES_DIR.iterdir()):
            pfile = pdir / "profile.json"
            if pdir.is_dir() and pfile.exists():
                try:
                    with open(pfile, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    _migrate_saved_profile(pfile, data)
                    prof = UserProfile(**data)
                    profiles.append(prof)
                    seen_ids.add(prof.profile_id)
                except Exception:
                    pass

    # Guarantee default always present
    if "default" not in seen_ids:
        default = UserProfile()
        create_profile(default)
        profiles.insert(0, default)

    return profiles


def create_profile(profile: UserProfile) -> None:
    """Save a new profile to profiles/{id}/profile.json."""
    profile_dir = _ensure_profile_dir(profile.profile_id)
    pfile = profile_dir / "profile.json"
    with open(pfile, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))


def load_profile(profile_id: str = "default") -> UserProfile:
    """Load profile settings. Returns default profile if not found."""
    pfile = PROFILES_DIR / profile_id / "profile.json"
    if pfile.exists():
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            _migrate_saved_profile(pfile, data)
            return UserProfile(**data)
        except Exception:
            pass
    return UserProfile(profile_id=profile_id)


# ---------------------------------------------------------------------------
# Memory load / save
# ---------------------------------------------------------------------------

def load_memory(profile_id: str = "default") -> SystemMemory:
    """Load profile-scoped memory from profiles/{profile_id}/memory.json."""
    # Migrate legacy root memory.json on first access to default profile
    if profile_id == "default":
        _migrate_legacy_memory()

    profile_dir = _ensure_profile_dir(profile_id)
    mem_path = profile_dir / "memory.json"

    if not mem_path.exists():
        initial = SystemMemory()
        save_memory(initial, profile_id)
        return initial

    try:
        with open(mem_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SystemMemory(**data)
    except Exception as e:
        print(f"[Memory] Failed to read {mem_path}: {e}. Initializing fresh memory.")
        return SystemMemory()


def save_memory(memory: SystemMemory, profile_id: str = "default") -> None:
    """Save memory to profile directory."""
    profile_dir = _ensure_profile_dir(profile_id)
    mem_path = profile_dir / "memory.json"
    try:
        with open(mem_path, "w", encoding="utf-8") as f:
            f.write(memory.model_dump_json(indent=2))
    except Exception as e:
        print(f"[Memory] Failed to write {mem_path}: {e}")


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def record_failures(
    failed_check_ids: List[str],
    profile_id: str = "default",
    topic: str = "",
    lessons_by_check: Dict[str, List[str]] | None = None,
) -> SystemMemory:
    """Record failures with topic-scoped lesson keys.

    Keys are stored as '{topic}::{check_id}' when a topic is provided,
    otherwise just '{check_id}'.
    """
    mem = load_memory(profile_id)
    for check_id in failed_check_ids:
        # Increment counter (bare check_id for aggregation)
        mem.failure_counters[check_id] = mem.failure_counters.get(check_id, 0) + 1

        # Style checks are stored per profile (topic-independent).
        # Content checks are stored per topic.
        if check_id in STYLE_CHECKS:
            key = f"__profile__::{check_id}"
        else:
            key = f"{topic}::{check_id}" if topic else check_id

        lessons = [str(line).replace("\n", " ").strip()[:280]
                   for line in (lessons_by_check or {}).get(check_id, []) if str(line).strip()]
        if not lessons:
            lessons = [f"Strictly adhere to {check_id} requirements on retry."]
        mem.failure_lessons[key] = list(dict.fromkeys(lessons))[:2]

    save_memory(mem, profile_id)
    return mem


def record_run_completion(
    success: bool,
    profile_id: str = "default",
) -> SystemMemory:
    """Update total and success run counts."""
    mem = load_memory(profile_id)
    mem.total_runs += 1
    if success:
        mem.successful_runs += 1
    else:
        mem.failed_runs += 1
    save_memory(mem, profile_id)
    return mem


def record_human_feedback(
    thumbs_up: bool,
    profile_id: str = "default",
) -> SystemMemory:
    """Record human thumbs up / thumbs down feedback from UI."""
    mem = load_memory(profile_id)
    if thumbs_up:
        mem.thumbs_up += 1
    else:
        mem.thumbs_down += 1
    save_memory(mem, profile_id)
    return mem


def record_feedback_reason(reason: str, profile_id: str = "default") -> SystemMemory:
    """Keep a single concise thumbs-down note without creating learned glossary entries."""
    text = " ".join(reason.split())[:280]
    if not text:
        return load_memory(profile_id)
    mem = load_memory(profile_id)
    mem.feedback_reasons.append(text)
    mem.feedback_reasons = mem.feedback_reasons[-100:]
    save_memory(mem, profile_id)
    return mem


def get_active_lessons(
    profile_id: str = "default",
    topic: str = "",
    max_lessons: int = 5,
) -> List[str]:
    """Return failure lessons for this topic plus topic-independent style lessons."""
    mem = load_memory(profile_id)

    lessons: List[str] = []

    # 1. Topic-specific lessons first (content checks: accuracy, coverage, etc.)
    if topic:
        topic_keys = [
            k for k in mem.failure_lessons
            if k.startswith(f"{topic}::")
        ]
        topic_keys.sort(
            key=lambda k: mem.failure_counters.get(k.split("::")[-1], 0),
            reverse=True,
        )
        for key in topic_keys:
            for lesson in mem.failure_lessons[key]:
                if lesson not in lessons:
                    lessons.append(lesson)
                if len(lessons) >= max_lessons:
                    return lessons

    # 2. Profile-scoped style lessons (readability, sentence length, jargon)
    #    These apply across all topics: fail on topic A → pass on topic B.
    style_keys = [
        k for k in mem.failure_lessons
        if k.startswith("__profile__::")
    ]
    style_keys.sort(
        key=lambda k: mem.failure_counters.get(k.split("::")[-1], 0),
        reverse=True,
    )
    for key in style_keys:
        for lesson in mem.failure_lessons[key]:
            if lesson not in lessons:
                lessons.append(lesson)
            if len(lessons) >= max_lessons:
                return lessons

    return lessons
