"""Parallel resume / document analyzer.

Three reviewer agents independently score the same resume against the same job
description, using different LLM providers (OpenAI, Anthropic, Gemini by
default). A synthesizer agent then merges the three rubrics into a single
scorecard with per-criterion agreement signals.

The pipeline is:
    document_extractor
        -> ParallelAgent(openai_reviewer, anthropic_reviewer, gemini_reviewer)
        -> synthesizer

The document_extractor uses native Gemini so ADK Web uploads are read once and
converted to plain text. Reviewer and synthesizer calls then strip raw upload
parts from their model requests and use the extracted text from state. This
avoids LiteLLM/OpenAI errors when ADK forwards a file_id without usable MIME
metadata.

Each reviewer model is independently configurable via env vars so any reviewer
can be swapped to a different provider.

Env vars (with defaults):
    DOCUMENT_EXTRACTOR_MODEL                              (gemini-2.5-flash)
    OPENAI_REVIEWER_PROVIDER     / OPENAI_REVIEWER_MODEL     (openai / gpt-4o-mini)
    ANTHROPIC_REVIEWER_PROVIDER  / ANTHROPIC_REVIEWER_MODEL  (anthropic / claude-sonnet-4-6)
    GEMINI_REVIEWER_PROVIDER     / GEMINI_REVIEWER_MODEL     (gemini / gemini-2.0-flash)
    SYNTHESIZER_PROVIDER         / SYNTHESIZER_MODEL         (anthropic / claude-sonnet-4-6)

Provider keys expected per provider:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY)

Input format:
    Paste the job description and upload the resume/document in ADK Web,
    or paste both in the same user turn.
"""

from __future__ import annotations

import os
from typing import Union

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.models import LiteLlm
from google.adk.models.base_llm import BaseLlm
from google.genai import types


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


def strip_uploaded_files_from_request(callback_context, llm_request):
    """Remove ADK Web upload parts after the extractor has read them."""
    sanitized_contents = []

    for content in llm_request.contents:
        kept_parts = []
        removed_files = []

        for part in content.parts or []:
            if part.file_data:
                removed_files.append(
                    part.file_data.display_name
                    or part.file_data.file_uri
                    or "uploaded file"
                )
                continue

            if part.inline_data:
                display_name = getattr(part.inline_data, "display_name", None)
                removed_files.append(display_name or "uploaded inline data")
                continue

            kept_parts.append(part)

        if removed_files:
            kept_parts.append(
                types.Part.from_text(
                    text=(
                        "[Uploaded file omitted from this model request: "
                        + ", ".join(removed_files)
                        + ". Use the extracted document payload in the "
                        "instructions/context instead.]"
                    )
                )
            )

        if kept_parts:
            content.parts = kept_parts
            sanitized_contents.append(content)

    llm_request.contents = sanitized_contents
    return None


DOCUMENT_EXTRACTOR_MODEL = os.getenv("DOCUMENT_EXTRACTOR_MODEL", "gemini-2.5-flash")

OPENAI_REVIEWER_PROVIDER = os.getenv("OPENAI_REVIEWER_PROVIDER", "openai")
OPENAI_REVIEWER_MODEL = os.getenv("OPENAI_REVIEWER_MODEL", "gpt-4o-mini")

ANTHROPIC_REVIEWER_PROVIDER = os.getenv("ANTHROPIC_REVIEWER_PROVIDER", "anthropic")
ANTHROPIC_REVIEWER_MODEL = os.getenv(
    "ANTHROPIC_REVIEWER_MODEL", "claude-sonnet-4-6"
)

GEMINI_REVIEWER_PROVIDER = os.getenv("GEMINI_REVIEWER_PROVIDER", "gemini")
GEMINI_REVIEWER_MODEL = os.getenv("GEMINI_REVIEWER_MODEL", "gemini-2.0-flash")

SYNTHESIZER_PROVIDER = os.getenv("SYNTHESIZER_PROVIDER", "anthropic")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "claude-sonnet-4-6")


document_extractor = Agent(
    name="document_extractor",
    model=DOCUMENT_EXTRACTOR_MODEL,
    description="Extracts plain text from ADK Web uploads and user text.",
    instruction=(
        "You extract hiring-analysis input from the user's message and any "
        "uploaded files. Return ONLY a JSON object with exactly these keys:\n\n"
        "{\n"
        '  "job_description": <full job description text, or empty string>,\n'
        '  "resume": <full resume/document text, or empty string>,\n'
        '  "notes": [<short note>, ...]\n'
        "}\n\n"
        "Preserve names, titles, dates, companies, education, skills, metrics, "
        "and requirements exactly as they appear. Do not score, summarize, or "
        "add facts. If the user pasted the job description but uploaded the "
        "resume, combine both into the JSON. If either side is missing, leave "
        "that field empty and add a brief note."
    ),
    output_key="document_payload",
)


REVIEWER_INSTRUCTION = (
    "You are an experienced technical recruiter scoring a resume against a "
    "job description. Use this extracted document payload as the source of "
    "truth:\n\n"
    "{document_payload}\n\n"
    "Score independently and objectively. Do not hedge. Return ONLY a JSON "
    "object with exactly this schema (no markdown fences, no commentary):\n\n"
    "{\n"
    '  "skills_match": <int 0-10>,\n'
    '  "experience_relevance": <int 0-10>,\n'
    '  "education_fit": <int 0-10>,\n'
    '  "soft_skills": <int 0-10>,\n'
    '  "overall_score": <int 0-10>,\n'
    '  "recommendation": "Strong Hire" | "Hire" | "Maybe" | "No Hire",\n'
    '  "key_strengths": [<string>, <string>, <string>],\n'
    '  "key_gaps": [<string>, <string>, <string>],\n'
    '  "one_line_summary": <string>\n'
    "}\n\n"
    "Be specific in strengths and gaps - cite actual resume content, not "
    "generic praise. If a criterion cannot be assessed from the resume, score "
    "it 0 and list it under key_gaps."
)


openai_reviewer = Agent(
    name="openai_reviewer",
    model=build_model(OPENAI_REVIEWER_PROVIDER, OPENAI_REVIEWER_MODEL),
    description="Scores the resume against the JD from a generalist perspective.",
    instruction=REVIEWER_INSTRUCTION,
    before_model_callback=strip_uploaded_files_from_request,
    output_key="openai_rubric",
)

anthropic_reviewer = Agent(
    name="anthropic_reviewer",
    model=build_model(ANTHROPIC_REVIEWER_PROVIDER, ANTHROPIC_REVIEWER_MODEL),
    description="Scores the resume against the JD with emphasis on reasoning.",
    instruction=REVIEWER_INSTRUCTION,
    before_model_callback=strip_uploaded_files_from_request,
    output_key="anthropic_rubric",
)

gemini_reviewer = Agent(
    name="gemini_reviewer",
    model=build_model(GEMINI_REVIEWER_PROVIDER, GEMINI_REVIEWER_MODEL),
    description="Scores the resume against the JD with emphasis on recency/grounding.",
    instruction=REVIEWER_INSTRUCTION,
    before_model_callback=strip_uploaded_files_from_request,
    output_key="gemini_rubric",
)

parallel_review = ParallelAgent(
    name="parallel_review",
    description="Runs three reviewer LLMs concurrently on the extracted JD/resume.",
    sub_agents=[openai_reviewer, anthropic_reviewer, gemini_reviewer],
)

synthesizer = Agent(
    name="synthesizer",
    model=build_model(SYNTHESIZER_PROVIDER, SYNTHESIZER_MODEL),
    description="Merges three independent rubrics into a final consensus scorecard.",
    instruction=(
        "You are a hiring manager consolidating three independent recruiter "
        "scorecards into one final assessment. Each scorecard is a JSON object.\n\n"
        "OpenAI reviewer:\n{openai_rubric}\n\n"
        "Anthropic reviewer:\n{anthropic_rubric}\n\n"
        "Gemini reviewer:\n{gemini_rubric}\n\n"
        "Produce a markdown report with these sections:\n\n"
        "## Final Scorecard\n"
        "A table with columns: Criterion | OpenAI | Anthropic | Gemini | Average | Agreement.\n"
        "Agreement = 'High' if the three scores are within 2 points, "
        "'Medium' if within 4, otherwise 'Low'.\n"
        "Criteria rows: Skills Match, Experience Relevance, Education Fit, "
        "Soft Skills, Overall.\n\n"
        "## Recommendation\n"
        "State the consensus recommendation. If reviewers disagree on the "
        "recommendation label, say so and explain the split.\n\n"
        "## Consolidated Strengths\n"
        "Bullet list. Merge duplicates across reviewers; mark items with (3/3), "
        "(2/3), or (1/3) to show how many reviewers raised them.\n\n"
        "## Consolidated Gaps\n"
        "Same format as Strengths.\n\n"
        "## Flags for Human Review\n"
        "Bullet any criteria with Low agreement, or any gap raised by only one "
        "reviewer that seems material. If nothing notable, write 'None'."
    ),
    before_model_callback=strip_uploaded_files_from_request,
    output_key="final_scorecard",
)

root_agent = SequentialAgent(
    name="resume_analyzer_pipeline",
    description=(
        "Resume / document analyzer: extract uploaded content, run three LLM "
        "reviewers in parallel, then synthesize one consensus scorecard."
    ),
    sub_agents=[document_extractor, parallel_review, synthesizer],
)
