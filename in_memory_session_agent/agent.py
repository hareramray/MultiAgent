"""Agent used by the InMemorySessionService demo."""

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


def save_session_note(key: str, value: str, tool_context: ToolContext) -> dict[str, Any]:
    """Save a note in the current in-memory session state."""
    normalized_key = "_".join(key.strip().lower().split())
    clean_value = value.strip()

    if not normalized_key:
        return {"status": "error", "message": "A key is required."}
    if not clean_value:
        return {"status": "error", "message": "A value is required."}

    notes = dict(tool_context.state.get("notes", {}))
    notes[normalized_key] = clean_value
    tool_context.state["notes"] = notes

    return {"status": "saved", "notes": notes}


def read_session_notes(tool_context: ToolContext) -> dict[str, Any]:
    """Read notes from the current in-memory session state."""
    notes = dict(tool_context.state.get("notes", {}))
    return {"notes": notes, "note_count": len(notes)}


IN_MEMORY_SESSION_PROVIDER = os.getenv("IN_MEMORY_SESSION_PROVIDER", "gemini")
IN_MEMORY_SESSION_MODEL = os.getenv(
    "IN_MEMORY_SESSION_MODEL",
    "gemini-2.5-flash",
)


root_agent = Agent(
    name="in_memory_session_demo_agent",
    model=build_model(IN_MEMORY_SESSION_PROVIDER, IN_MEMORY_SESSION_MODEL),
    description=(
        "Demonstrates ADK InMemorySessionService by storing notes in session "
        "state during a single process."
    ),
    instruction=(
        "You are a concise assistant demonstrating ADK in-memory sessions.\n\n"
        "Use save_session_note when the user asks you to remember a stable "
        "detail for this demo. Use read_session_notes before answering when "
        "the user asks what you remember or refers to earlier details.\n\n"
        "Keep replies short. Mention that this data is in-memory only when it "
        "helps clarify the demo."
    ),
    tools=[save_session_note, read_session_notes],
    output_key="last_in_memory_reply",
)
