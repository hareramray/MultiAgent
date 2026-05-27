"""Run two agent turns with InMemorySessionService.

Because this service stores data only in process memory, both turns run inside
one Python process. Running this script again starts with an empty service.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types


APP_NAME = "in_memory_session_agent_app"
DEFAULT_USER_ID = "demo_user"
DEFAULT_SESSION_ID = "demo_session"


def load_local_env() -> None:
    """Load in_memory_session_agent/.env when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(Path(__file__).with_name(".env"))


def content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


def print_session_summary(label: str, session: Session | None) -> None:
    print(f"\n--- {label} ---")
    if session is None:
        print("Session: None")
        return

    print(f"Session id: {session.id}")
    print(f"State:      {session.state}")
    print(f"Events:     {len(session.events)}")


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

    session_service = InMemorySessionService()
    created_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
        state={"notes": {}, "storage": "in-memory"},
    )
    print_session_summary("Created Session", created_session)

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    for index, message in enumerate(args.turn, start=1):
        print(f"\nUser turn {index}: {message}")
        response = await run_turn(
            runner=runner,
            user_id=args.user_id,
            session_id=args.session_id,
            message=message,
        )
        print(f"Assistant turn {index}: {response}")

        current_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=args.user_id,
            session_id=args.session_id,
        )
        print_session_summary(f"Session After Turn {index}", current_session)

    list_response = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=args.user_id,
    )
    print("\nKnown in-memory session ids:")
    for session in list_response.sessions:
        print(f"- {session.id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent with ADK InMemorySessionService."
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument(
        "--turn",
        action="append",
        default=None,
        help="User turn. Pass multiple --turn values to show same-process state.",
    )

    args = parser.parse_args()
    if args.turn is None:
        args.turn = [
            "Remember that my demo goal is learning ADK sessions.",
            "What do you remember from this session?",
        ]
    return args


def main() -> None:
    asyncio.run(run_demo(parse_args()))


if __name__ == "__main__":
    main()
