"""Sequential pipeline: researcher -> writer -> reviewer.

Each sub-agent's model is independently configurable via environment variables,
so the same pipeline can mix OpenAI, Anthropic, Gemini, and Ollama models.

Env vars (with defaults):
    RESEARCH_PROVIDER  / RESEARCH_MODEL   (default: gemini / gemini-2.0-flash)
    WRITER_PROVIDER    / WRITER_MODEL     (default: openai / gpt-4o-mini)
    REVIEWER_PROVIDER  / REVIEWER_MODEL   (default: anthropic / claude-sonnet-4-6)

Provider keys also expected per-provider:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)
"""

from __future__ import annotations

import os
from typing import Union

from google.adk.agents import Agent, SequentialAgent
from google.adk.models import LiteLlm
from google.adk.models.base_llm import BaseLlm


def build_model(provider: str, model_name: str) -> Union[str, BaseLlm]:
    """Return an ADK model spec for the given provider+model.

    Gemini models are returned as a bare string so ADK's registry resolves them
    natively; all other providers route through LiteLlm.
    """
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


RESEARCH_PROVIDER = os.getenv("RESEARCH_PROVIDER", "gemini")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gemini-2.0-flash")

WRITER_PROVIDER = os.getenv("WRITER_PROVIDER", "openai")
WRITER_MODEL = os.getenv("WRITER_MODEL", "gpt-4o-mini")

REVIEWER_PROVIDER = os.getenv("REVIEWER_PROVIDER", "anthropic")
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "claude-sonnet-4-6")


researcher = Agent(
    name="researcher",
    model=build_model(RESEARCH_PROVIDER, RESEARCH_MODEL),
    description="Gathers key facts about the user's topic.",
    instruction=(
        "You are a research assistant. Given the user's topic, produce 3-5 concise "
        "bullet points of factual, relevant information about it. "
        "Output only the bullet list, nothing else."
    ),
    output_key="research_notes",
)

writer = Agent(
    name="writer",
    model=build_model(WRITER_PROVIDER, WRITER_MODEL),
    description="Drafts a short article from the research notes.",
    instruction=(
        "You are a writer. Use the research notes below to write a clear, engaging "
        "2-paragraph article on the topic.\n\n"
        "Research notes:\n{research_notes}\n\n"
        "Output the article only."
    ),
    output_key="draft_article",
)

reviewer = Agent(
    name="reviewer",
    model=build_model(REVIEWER_PROVIDER, REVIEWER_MODEL),
    description="Edits the draft for clarity, grammar, and tone.",
    instruction=(
        "You are an editor. Review the draft below for clarity, grammar, and tone. "
        "Return the final polished article only, with no commentary.\n\n"
        "Draft:\n{draft_article}"
    ),
    output_key="final_article",
)

root_agent = SequentialAgent(
    name="research_writer_pipeline",
    description="Researches a topic, drafts an article, and edits the result.",
    sub_agents=[researcher, writer, reviewer],
)
