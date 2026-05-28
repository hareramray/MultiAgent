"""CLI runner for the ADK model callback demo."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types


APP_NAME = "model_callback_demo_app"
DEFAULT_USER_ID = "demo_user"
DEFAULT_SESSION_ID = "model_callback_session"
DEFAULT_MESSAGE = (
    "Summarize this deliberately oversized demo prompt for model cost control: "
    + ("budget control callback " * 40)
)


def load_local_env() -> None:
    """Load model_callback_agent/.env when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(Path(__file__).with_name(".env"))


def content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


def print_session_state(session: Session | None) -> None:
    print("\n--- Callback State ---")
    if session is None:
        print("Session: None")
        return

    interesting_keys = [
        "before_model_calls",
        "after_model_calls",
        "allowed_model_calls",
        "skipped_model_calls",
        "last_before_model_message",
        "last_prompt_chars",
        "last_estimated_prompt_tokens",
        "last_after_model_preview",
        "last_response_chars",
        "last_estimated_response_tokens",
        "callback_events",
        "last_model_callback_reply",
    ]
    for key in interesting_keys:
        if key in session.state:
            print(f"{key}: {session.state[key]}")


async def run_turn(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
) -> str:
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        if event.is_final_response():
            final_text = content_text(event.content)

    return final_text


async def run_demo(args: argparse.Namespace) -> None:
    load_local_env()
    from .agent import root_agent

    message = " ".join(args.message).strip() or DEFAULT_MESSAGE

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
        state={"callback_events": []},
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    print(f"User: {message}")
    response = await run_turn(
        runner=runner,
        user_id=args.user_id,
        session_id=args.session_id,
        message=message,
    )
    print(f"Assistant: {response}")

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    if not args.hide_state:
        print_session_state(session)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ADK model callback demo agent."
    )
    parser.add_argument(
        "message",
        nargs="*",
        help=(
            "User message. If omitted, an oversized prompt is used so the "
            "cost-control callback skips the real model call."
        ),
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument(
        "--hide-state",
        action="store_true",
        help="Hide callback state after the turn.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_demo(parse_args()))


if __name__ == "__main__":
    main()
