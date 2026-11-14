"""OpenAI-compatible model provider wrapper with resilient JSON repair."""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

T = TypeVar("T", bound=BaseModel)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
DEFAULT_PROVIDER_MODEL = os.getenv("NVIDIA_MODEL", "").strip() if LLM_PROVIDER == "nvidia" else os.getenv("OPENROUTER_MODEL", "").strip()
DEFAULT_GEN_MODEL = os.getenv("DRAFT_MODEL", os.getenv("GEN_MODEL", DEFAULT_PROVIDER_MODEL)).strip()
DEFAULT_FAST_MODEL = os.getenv("FAST_MODEL", "")
DEFAULT_PLAN_MODEL = os.getenv("PLAN_MODEL", DEFAULT_FAST_MODEL or DEFAULT_GEN_MODEL)
# The judge must NOT silently fall back to the draft model. If JUDGE_MODEL is
# unset, use the provider default but warn that the review is not independent.
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "").strip()
if not DEFAULT_JUDGE_MODEL:
    DEFAULT_JUDGE_MODEL = DEFAULT_PROVIDER_MODEL or DEFAULT_GEN_MODEL
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.90"))
DEFAULT_MAX_TOKENS = {
    "plan": int(os.getenv("PLAN_MAX_TOKENS", "1200")),
    "generate": int(os.getenv("DRAFT_MAX_TOKENS", "4096")),
    "judge": int(os.getenv("JUDGE_MAX_TOKENS", "1800")),
}

# Call usage statistics for the current run
USAGE_STATS = {
    "total_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "repair_attempts": 0,
    "last_model_used": None,
    "usage_incomplete": False,
    "calls_by_stage": {},
    "call_events": [],
}


def get_openrouter_client() -> OpenAI:
    """Create the selected OpenAI-compatible client, with optional LangSmith tracing."""
    if LLM_PROVIDER == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
        key_name = "NVIDIA_API_KEY"
    elif LLM_PROVIDER == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        key_name = "OPENROUTER_API_KEY"
    else:
        raise ValueError("LLM_PROVIDER must be 'nvidia' or 'openrouter'.")
    if not api_key:
        raise ValueError(f"{key_name} is not set in environment or .env file.")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Wrap client with LangSmith tracing if enabled
    if os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1") or os.getenv("LANGSMITH_API_KEY"):
        try:
            from langsmith.wrappers import wrap_openai
            return wrap_openai(client)
        except Exception as e:
            print(f"[LangSmith] wrap_openai warning: {e}")
            return client

    return client


def reset_usage_stats():
    """Reset counters at the start of a new generation run."""
    global USAGE_STATS
    USAGE_STATS = {
        "total_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "repair_attempts": 0,
        "last_model_used": None,
        "usage_incomplete": False,
        "calls_by_stage": {},
        "call_events": [],
    }


def _response_text(response: Any) -> str:
    """Safely extract text from an OpenAI-compatible completion response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        model_name = getattr(response, "model", None) or "unknown model"
        response_id = getattr(response, "id", None) or "no response id"
        raise RuntimeError(f"Provider returned zero completion choices for {model_name} ({response_id}).")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("Provider returned a completion choice without a message.")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(getattr(part, "text", "") or (part.get("text", "") if isinstance(part, dict) else ""))
            for part in content
        ).strip()
    else:
        text = ""
    finish_reason = getattr(choices[0], "finish_reason", None)
    if finish_reason != "stop":
        if finish_reason == "length":
            raise TruncatedGenerationError("Provider stopped at the output token limit (finish_reason=length).")
        raise RuntimeError(f"Provider did not complete the answer (finish_reason={finish_reason or 'missing'}).")
    if not text:
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise RuntimeError(f"Provider refused the request: {refusal}")
        finish_reason = getattr(choices[0], "finish_reason", None) or "not provided"
        raise RuntimeError(f"Provider returned an empty message (finish_reason={finish_reason}).")
    return text


class TruncatedGenerationError(RuntimeError):
    """Raised when a provider returns a response that did not finish normally."""


def classify_provider_error(error: BaseException, *, truncated: bool = False) -> tuple[str, Optional[int]]:
    status = getattr(error, "status_code", None)
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    if truncated or isinstance(error, TruncatedGenerationError):
        return "truncated", status
    if status == 429 or "ratelimit" in name or "rate limit" in message:
        return "rate_limit", status or 429
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return "auth", status
    if "timeout" in name or isinstance(error, TimeoutError):
        return "timeout", status
    if "connection" in name or "connect" in message or "dns" in message:
        return "connectivity", status
    if status is not None and 400 <= status < 500:
        return "bad_response", status
    if isinstance(error, (IndexError, KeyError, TypeError, ValueError)):
        return "bad_response", status
    return "bad_response", status


def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    max_retries: int = 2,
    top_p: Optional[float] = None,
    stream_callback=None,
    stage: str = "other",
    response_format: Optional[Dict[str, Any]] = None,
    run_attempt: int = 1,
    audit_callback=None,
) -> str:
    """Call the selected model provider with retry and usage tracking."""
    selected_model = model or DEFAULT_GEN_MODEL
    if not selected_model:
        raise RuntimeError("No model configured. Set DRAFT_MODEL/GEN_MODEL and JUDGE_MODEL in .env.")
    client = get_openrouter_client()
    sampling_top_p = DEFAULT_TOP_P if top_p is None else top_p
    sampling_options = {"top_p": sampling_top_p}
    if LLM_PROVIDER != "nvidia":
        sampling_options["temperature"] = temperature

    last_error = None
    def begin_call():
        USAGE_STATS["total_calls"] += 1
        stats = USAGE_STATS.setdefault("calls_by_stage", {}).setdefault(
            stage, {"calls": 0, "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        stats["calls"] += 1
        USAGE_STATS["last_model_used"] = selected_model
        return time.perf_counter()

    def finish_call(started, result=None):
        stats = USAGE_STATS["calls_by_stage"][stage]
        stats["latency_ms"] += int((time.perf_counter() - started) * 1000)
        usage = getattr(result, "usage", None) if result is not None else None
        if usage:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            USAGE_STATS["total_prompt_tokens"] += prompt_tokens
            USAGE_STATS["total_completion_tokens"] += completion_tokens
        else:
            USAGE_STATS["usage_incomplete"] = True

    def emit(event: Dict[str, Any]) -> None:
        USAGE_STATS.setdefault("call_events", []).append(event)
        if audit_callback:
            try:
                audit_callback(event)
            except Exception as exc:
                print(f"[LLM] Audit callback failed: {type(exc).__name__}")

    def create(stream, provider_attempt):
        started = begin_call()
        request_event = {
            "stage": stage, "model": selected_model, "attempt": run_attempt,
            "provider_attempt": provider_attempt, "stream": bool(stream),
        }
        if stream_callback:
            stream_callback("\x00RESET\x00")
        try:
            options = dict(model=selected_model, messages=messages, **sampling_options,
                           max_tokens=max_tokens, stream=stream)
            if response_format:
                options["response_format"] = response_format
            if LLM_PROVIDER == "nvidia":
                options["extra_body"] = {"chat_template_kwargs": {
                    "enable_thinking": os.getenv("NVIDIA_ENABLE_THINKING", "false").strip().lower() in {"1", "true", "yes"}
                }}
            result = client.chat.completions.create(**options)
            return result, started, request_event
        except Exception as exc:
            finish_call(started)
            error_class, status = classify_provider_error(exc)
            emit({**request_event, "status": "error", "latency_ms": int((time.perf_counter() - started) * 1000),
                  "prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None,
                  "finish_reason": None, "http_status": status, "error_class": error_class})
            raise

    def complete_event(started, request_event, response=None, *, finish_reason=None, error=None):
        finish_call(started, response)
        usage = getattr(response, "usage", None) if response is not None else None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        error_class, status = classify_provider_error(error, truncated=isinstance(error, TruncatedGenerationError)) if error else (None, 200 if response is not None else None)
        emit({**request_event, "status": "error" if error else "success",
              "latency_ms": int((time.perf_counter() - started) * 1000),
              "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
              "reasoning_tokens": reasoning_tokens, "finish_reason": finish_reason,
              "http_status": status, "error_class": error_class})

    def one_request(stream: bool, provider_attempt: int) -> str:
        response, started, request_event = create(stream, provider_attempt)
        if not stream:
            choices = getattr(response, "choices", None) or []
            finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
            try:
                content = _response_text(response)
            except Exception as exc:
                complete_event(started, request_event, response, finish_reason=finish_reason, error=exc)
                raise
            complete_event(started, request_event, response, finish_reason=finish_reason)
            return content

        parts: List[str] = []
        finish_reason = None
        stream_usage = None
        try:
            if not hasattr(response, "__iter__"):
                raise RuntimeError("Provider returned a non-iterable stream.")
            for chunk in response:
                if getattr(chunk, "usage", None):
                    stream_usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    parts.append(piece)
                    stream_callback(piece)
            if finish_reason != "stop":
                if finish_reason == "length":
                    raise TruncatedGenerationError("Stream stopped at the output token limit (finish_reason=length).")
                raise TruncatedGenerationError(f"Stream ended without finish_reason=stop (got {finish_reason or 'none'}).")
            content = "".join(parts).strip()
            if not content:
                raise RuntimeError("Provider completed the stream without user-facing text.")
            complete_event(started, request_event, stream_usage, finish_reason=finish_reason)
            return content
        except Exception as exc:
            complete_event(started, request_event, stream_usage, finish_reason=finish_reason, error=exc)
            raise

    for attempt in range(1, max_retries + 1):
        try:
            if stream_callback:
                try:
                    return one_request(True, attempt)
                except Exception as stream_error:
                    # Replace any partial stream with the complete non-streaming response.
                    try:
                        content = one_request(False, attempt)
                        stream_callback(content)
                        return content
                    except Exception as fallback_error:
                        last_error = fallback_error
                        if isinstance(stream_error, TruncatedGenerationError):
                            last_error = stream_error if isinstance(fallback_error, TruncatedGenerationError) else fallback_error
                        raise last_error
            return one_request(False, attempt)

        except Exception as e:
            last_error = e
            print(f"[LLM] {stage} call to {selected_model} attempt {attempt} failed ({classify_provider_error(e)[0]}): {e}")
            if attempt < max_retries:
                time.sleep(2.0 * attempt)

    error_class, http_status = classify_provider_error(last_error or RuntimeError("unknown provider error"))
    status_text = f", HTTP {http_status}" if http_status is not None else ""
    raise RuntimeError(
        f"LLM provider call failed after {max_retries} attempts [{error_class}{status_text}]: {last_error}"
    ) from last_error


def extract_json_from_text(text: str) -> Optional[Any]:
    """Tolerantly extract and parse JSON from raw LLM output."""
    if not text:
        return None

    # Step 1: Check for fenced ```json ... ``` blocks
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else text.strip()

    # Step 2: Attempt direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Step 3: Find outermost { ... } or [ ... ]
    first_brace = candidate.find("{")
    last_brace = candidate.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_slice = candidate[first_brace : last_brace + 1]
        try:
            return json.loads(json_slice)
        except json.JSONDecodeError:
            # Clean common trailing commas before closing braces/brackets
            cleaned = re.sub(r",\s*([\}\]])", r"\1", json_slice)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    first_bracket = candidate.find("[")
    last_bracket = candidate.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        json_slice = candidate[first_bracket : last_bracket + 1]
        try:
            return json.loads(json_slice)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([\}\]])", r"\1", json_slice)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    return None


def coerce_to_schema(data: Any, schema: Type[T]) -> Optional[T]:
    """Gracefully coerce parsed JSON (whether dict or list) into a target Pydantic schema."""
    if data is None:
        return None

    if isinstance(data, dict):
        try:
            return schema.model_validate(data)
        except ValidationError:
            pass

    elif isinstance(data, list):
        # Case 1: List with a single dict element
        if len(data) == 1 and isinstance(data[0], dict):
            try:
                return schema.model_validate(data[0])
            except ValidationError:
                pass

        # Case 2: List of strings passed where schema expects a list field (e.g. core_sections)
        if all(isinstance(x, str) for x in data):
            # Check for core_sections or any list field
            for field_name in ["core_sections", "items", "sections", "points"]:
                if field_name in schema.model_fields:
                    try:
                        return schema.model_validate({field_name: data})
                    except ValidationError:
                        pass

        # Case 3: Scan items in list for first matching dict
        for item in data:
            if isinstance(item, dict):
                try:
                    return schema.model_validate(item)
                except ValidationError:
                    pass

    return None


def call_llm_json(
    messages: List[Dict[str, str]],
    schema: Type[T],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    stage: str = "other",
    repair: bool = False,
    run_attempt: int = 1,
    audit_callback=None,
) -> T:
    """Call LLM and strictly validate output against a Pydantic schema with tolerant coercion and repair."""
    raw_text = call_llm(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens,
                        stage=stage, response_format={"type": "json_object"}, run_attempt=run_attempt,
                        audit_callback=audit_callback)
    parsed = extract_json_from_text(raw_text)

    # First attempt: Coerce parsed JSON into schema
    validated = coerce_to_schema(parsed, schema)
    if validated is not None:
        return validated

    if not repair:
        raise ValueError(f"Provider returned invalid {schema.__name__} JSON; repair call disabled.")

    # Trigger single lightweight repair attempt
    USAGE_STATS["repair_attempts"] += 1
    print(f"[LLM:JSON] Triggering repair attempt for schema {schema.__name__}...")

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    repair_prompt = [
        {
            "role": "system",
            "content": (
                "You are a strict JSON repair engine. Format the following data into a valid JSON object matching the schema below. "
                "Output ONLY a single valid JSON object ({...}). Do NOT output a raw list/array ([...]). No markdown comments."
            ),
        },
        {
            "role": "user",
            "content": f"TARGET SCHEMA:\n{schema_json}\n\nMALFORMED OUTPUT TO REPAIR:\n{raw_text}",
        },
    ]

    repaired_text = call_llm(
        messages=repair_prompt,
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        stage=f"{stage}_repair",
        response_format={"type": "json_object"},
        run_attempt=run_attempt,
        audit_callback=audit_callback,
    )
    repaired_parsed = extract_json_from_text(repaired_text)
    repaired_validated = coerce_to_schema(repaired_parsed, schema)
    if repaired_validated is not None:
        return repaired_validated

    raise ValueError(f"Failed to parse or repair valid JSON for schema {schema.__name__}. Raw text was:\n{raw_text[:300]}")
