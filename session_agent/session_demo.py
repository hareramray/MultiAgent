"""CLI runner for the ADK session demo agent."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types


APP_NAME = "session_demo_app"
DEFAULT_USER_ID = "demo_user"
DEFAULT_SESSION_ID = "demo_session"
DEFAULT_DB_PATH = "session.db"


def load_local_env() -> None:
    """Load session_agent/.env when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(Path(__file__).with_name(".env"))


def content_text(content: types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


async def run_message(args: argparse.Namespace) -> None:
    load_local_env()
    from .agent import root_agent

    message = " ".join(args.message).strip()
    if not message:
        message = input("You: ").strip()
    if not message:
        raise SystemExit("A message is required.")

    session_service = SqliteSessionService(db_path=args.db_path)
    existing_session = session_service.get_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
    )

    if existing_session is None:
        session_service.create_session(
            app_name=APP_NAME,
            user_id=args.user_id,
            session_id=args.session_id,
            state={"facts": {}},
        )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=args.user_id,
        session_id=args.session_id,
        new_message=user_content,
    ):
        if event.is_final_response():
            final_text = content_text(event.content)

    print(f"Assistant: {final_text}")

    if args.show_state:
        session = session_service.get_session(
            app_name=APP_NAME,
            user_id=args.user_id,
            session_id=args.session_id,
        )
        print(f"Session state: {session.state if session else {}}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ADK session demo with a persistent SQLite session."
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="User message. If omitted, the script prompts for one.",
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print the stored ADK session state after the turn.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_message(parse_args()))


if __name__ == "__main__":
    main()
