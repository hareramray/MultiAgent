"""Session demo agent.

This example demonstrates two ADK session behaviors:
    1. conversation history is tied to app_name + user_id + session_id
    2. tools can persist structured data in tool_context.state

Run with ADK Web:
    adk web

Run from the CLI:
    python -m session_agent.session_demo --session-id demo "Remember my name is Ada."
    python -m session_agent.session_demo --session-id demo "What do you remember?"
"""

from __future__ import annotations

import os
from typing import Any, Union

from google.adk.agents import Agent
from google.adk.models import LiteLlm
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import ToolContext


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


def _normalize_key(key: str) -> str:
    return "_".join(key.strip().lower().split())


def remember_fact(key: str, value: str, tool_context: ToolContext) -> dict[str, Any]:
    """Save one durable fact in the current ADK session state."""
    normalized_key = _normalize_key(key)
    clean_value = value.strip()

    if not normalized_key:
        return {"status": "error", "message": "A non-empty key is required."}
    if not clean_value:
        return {"status": "error", "message": "A non-empty value is required."}

    facts = dict(tool_context.state.get("facts", {}))
    facts[normalized_key] = clean_value
    tool_context.state["facts"] = facts

    return {
        "status": "saved",
        "key": normalized_key,
        "value": clean_value,
        "fact_count": len(facts),
    }


def recall_facts(tool_context: ToolContext) -> dict[str, Any]:
    """Return all facts saved in the current ADK session state."""
    facts = dict(tool_context.state.get("facts", {}))
    return {"facts": facts, "fact_count": len(facts)}


def forget_fact(key: str, tool_context: ToolContext) -> dict[str, Any]:
    """Remove one fact from the current ADK session state."""
    normalized_key = _normalize_key(key)
    facts = dict(tool_context.state.get("facts", {}))

    if normalized_key not in facts:
        return {
            "status": "not_found",
            "key": normalized_key,
            "remaining_facts": facts,
        }

    removed_value = facts.pop(normalized_key)
    tool_context.state["facts"] = facts

    return {
        "status": "forgotten",
        "key": normalized_key,
        "removed_value": removed_value,
        "fact_count": len(facts),
    }


SESSION_PROVIDER = os.getenv("SESSION_PROVIDER", "gemini")
SESSION_MODEL = os.getenv("SESSION_MODEL", "gemini-2.0-flash")


root_agent = Agent(
    name="session_state_demo_agent",
    model=build_model(SESSION_PROVIDER, SESSION_MODEL),
    description=(
        "Demonstrates ADK sessions by remembering and recalling facts within "
        "a stable session id."
    ),
    instruction=(
        "You are a concise assistant for demonstrating ADK session behavior.\n\n"
        "Use the provided tools to manage session state:\n"
        "- When the user shares a stable preference, profile detail, goal, or "
        "task that should be remembered, call remember_fact.\n"
        "- When the user asks what you remember, calls back to earlier details, "
        "or asks for a personalized answer, call recall_facts before replying.\n"
        "- When the user asks you to remove a saved detail, call forget_fact.\n\n"
        "Keep replies short and make it clear when something was saved, recalled, "
        "or removed. Do not store secrets, API keys, passwords, or highly "
        "sensitive personal data. Session facts are scoped to the current ADK "
        "app_name, user_id, and session_id."
    ),
    tools=[remember_fact, recall_facts, forget_fact],
    output_key="last_session_reply",
)
