"""ADK model callback demo agent.

This example demonstrates model-level callbacks:
    1. before_model_callback can inspect a request before model cost is spent.
    2. before_model_callback can return LlmResponse to enforce a budget limit.
    3. after_model_callback can record rough response cost telemetry.
"""

from __future__ import annotations

import os
from typing import Optional, Union

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LiteLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


AUDIT_NOTE_PREFIX = "[after_model_callback estimated this response at about "
AUDIT_NOTE_SUFFIX = " output tokens.]"


def build_model(provider: str, model_name: str) -> Union[str, BaseLlm]:
    """Return an ADK model spec for the given provider+model."""
    p = provider.strip().lower()
    if p == "gemini":
        return model_name
    if p == "openai":
        return LiteLlm(model=f"openai/{model_name}")
    if p == "anthropic":
        return LiteLlm(model=f"anthropic/{model_name}")
    if p == "ollama":
        return LiteLlm(model=f"ollama/{model_name}")
    raise ValueError(
        f"Unknown provider: {provider!r}. "
        "Expected one of: gemini, openai, anthropic, ollama."
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


def _has_audit_note(content: types.Content | None) -> bool:
    if not content or not content.parts:
        return False
    return any(AUDIT_NOTE_PREFIX in (part.text or "") for part in content.parts)


def _content_text_without_audit_note(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""

    clean_chunks = []
    for part in content.parts:
        text = part.text or ""
        if AUDIT_NOTE_PREFIX in text:
            lines = [line for line in text.splitlines() if AUDIT_NOTE_PREFIX not in line]
            text = "\n".join(lines)
        clean_chunks.append(text)
    return "".join(clean_chunks)


def _system_instruction_text(llm_request: LlmRequest) -> str:
    instruction = llm_request.config.system_instruction
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, types.Content):
        return _content_text(instruction)
    return str(instruction)


def _request_text(llm_request: LlmRequest) -> str:
    content_texts = [_content_text(content) for content in llm_request.contents]
    content_texts.append(_system_instruction_text(llm_request))
    return "\n".join(text for text in content_texts if text)


def _estimate_tokens(text: str) -> int:
    # Cheap approximation for demos: English text is often around 4 chars/token.
    return max(1, (len(text) + 3) // 4) if text else 0


def _latest_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents):
        if content.role == "user":
            text = _content_text(content).strip()
            if text:
                return text
    return ""


def _record_callback_event(
    callback_context: CallbackContext,
    event: dict[str, object],
) -> None:
    events = list(callback_context.state.get("callback_events", []))
    events.append(event)
    callback_context.state["callback_events"] = events[-20:]


def enforce_model_budget(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Skip expensive model calls before they happen."""
    user_text = _latest_user_text(llm_request)
    request_text = _request_text(llm_request)
    prompt_chars = len(request_text)
    estimated_prompt_tokens = _estimate_tokens(request_text)
    max_prompt_chars = _env_int("MODEL_CALLBACK_MAX_PROMPT_CHARS", 700)
    max_prompt_tokens = _env_int("MODEL_CALLBACK_MAX_PROMPT_TOKENS", 220)
    max_model_calls = _env_int("MODEL_CALLBACK_MAX_MODEL_CALLS_PER_SESSION", 2)
    allowed_calls = int(callback_context.state.get("allowed_model_calls", 0))

    callback_context.state["before_model_calls"] = (
        int(callback_context.state.get("before_model_calls", 0)) + 1
    )
    callback_context.state["last_before_model_message"] = user_text
    callback_context.state["last_prompt_chars"] = prompt_chars
    callback_context.state["last_estimated_prompt_tokens"] = estimated_prompt_tokens

    budget_reasons = []
    if prompt_chars > max_prompt_chars:
        budget_reasons.append(
            f"prompt characters {prompt_chars} exceeded limit {max_prompt_chars}"
        )
    if estimated_prompt_tokens > max_prompt_tokens:
        budget_reasons.append(
            "estimated prompt tokens "
            f"{estimated_prompt_tokens} exceeded limit {max_prompt_tokens}"
        )
    if allowed_calls >= max_model_calls:
        budget_reasons.append(
            f"session model calls {allowed_calls} reached limit {max_model_calls}"
        )

    if budget_reasons:
        skipped_calls = int(callback_context.state.get("skipped_model_calls", 0)) + 1
        callback_context.state["skipped_model_calls"] = skipped_calls
        _record_callback_event(
            callback_context,
            {
                "callback": "before_model_callback",
                "action": "blocked_for_budget",
                "prompt_chars": prompt_chars,
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "reasons": budget_reasons,
            },
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "The model call was skipped by the cost-control "
                            "callback before any LLM request was sent.\n\n"
                            "Reason: " + "; ".join(budget_reasons) + "."
                        )
                    )
                ],
            ),
            custom_metadata={
                "skipped_by": "before_model_callback",
                "reason": "budget_limit",
            },
        )

    callback_context.state["allowed_model_calls"] = allowed_calls + 1
    llm_request.append_instructions(
        [
            "This request passed the model-cost callback budget check. "
            "Answer concisely to keep output token cost low."
        ]
    )
    _record_callback_event(
        callback_context,
        {
            "callback": "before_model_callback",
            "action": "allowed_by_budget",
            "prompt_chars": prompt_chars,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "allowed_model_call_number": allowed_calls + 1,
        },
    )
    return None


def record_model_cost_telemetry(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Record rough response cost telemetry after a real model call."""
    has_audit_note = _has_audit_note(llm_response.content)
    response_text = _content_text_without_audit_note(llm_response.content).strip()
    response_chars = len(response_text)
    estimated_response_tokens = _estimate_tokens(response_text)

    callback_context.state["after_model_calls"] = (
        int(callback_context.state.get("after_model_calls", 0)) + 1
    )
    callback_context.state["last_after_model_preview"] = response_text[:160]
    callback_context.state["last_response_chars"] = response_chars
    callback_context.state["last_estimated_response_tokens"] = estimated_response_tokens
    _record_callback_event(
        callback_context,
        {
            "callback": "after_model_callback",
            "action": "recorded_response_cost_telemetry",
            "response_chars": response_chars,
            "estimated_response_tokens": estimated_response_tokens,
            "audit_note_already_present": has_audit_note,
        },
    )

    if (
        not _env_bool("MODEL_CALLBACK_APPEND_AUDIT_NOTE", True)
        or llm_response.partial
        or not llm_response.content
        or not llm_response.content.parts
    ):
        return None

    if has_audit_note:
        return None

    llm_response.content.parts.append(
        types.Part.from_text(
            text=(
                "\n\n"
                f"{AUDIT_NOTE_PREFIX}{estimated_response_tokens}"
                f"{AUDIT_NOTE_SUFFIX}"
            )
        )
    )
    return llm_response


MODEL_CALLBACK_PROVIDER = os.getenv("MODEL_CALLBACK_PROVIDER", "gemini")
MODEL_CALLBACK_MODEL = os.getenv("MODEL_CALLBACK_MODEL", "gemini-2.5-flash")


root_agent = Agent(
    name="model_callback_demo_agent",
    model=build_model(MODEL_CALLBACK_PROVIDER, MODEL_CALLBACK_MODEL),
    description=(
        "Demonstrates ADK model callbacks by enforcing a simple prompt and "
        "session budget before model calls are sent."
    ),
    instruction=(
        "You are a concise assistant demonstrating ADK model cost control. "
        "Explain callback behavior plainly and avoid long answers."
    ),
    before_model_callback=enforce_model_budget,
    after_model_callback=record_model_cost_telemetry,
    output_key="last_model_callback_reply",
)
