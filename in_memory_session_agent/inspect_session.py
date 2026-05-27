"""Inspect ADK Session objects created by InMemorySessionService.

This script mirrors the core idea from:
https://adk.dev/sessions/session/

It does not call an LLM or require API keys.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from typing import Iterable

from google.adk.sessions import InMemorySessionService, Session


APP_NAME = "in_memory_session_demo"
DEFAULT_USER_ID = "demo_user"
DEFAULT_SESSION_ID = "demo_session"


def format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def describe_session(title: str, session: Session | None) -> None:
    print(f"\n--- {title} ---")

    if session is None:
        print("Session: None")
        return

    print(f"Type:             {type(session).__name__}")
    print(f"ID:               {session.id}")
    print(f"Application name: {session.app_name}")
    print(f"User ID:          {session.user_id}")
    print(f"State:            {session.state}")
    print(f"Events:           {len(session.events)}")
    print(f"Last update:      {format_timestamp(session.last_update_time)}")


def describe_session_ids(title: str, sessions: Iterable[Session]) -> None:
    print(f"\n--- {title} ---")
    for session in sessions:
        print(f"- {session.id}: user={session.user_id}, state={session.state}")


async def run_demo(args: argparse.Namespace) -> None:
    session_service = InMemorySessionService()

    created_session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
        state={
            "initial_key": "initial_value",
            "storage": "process memory only",
        },
    )

    describe_session("Created Session", created_session)
    print(f"Is Session instance: {isinstance(created_session, Session)}")

    retrieved_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    describe_session("Retrieved Same Session", retrieved_session)

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=f"{args.session_id}_second",
        state={"initial_key": "second_value"},
    )
    list_response = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=args.user_id,
    )
    describe_session_ids("Sessions For User", list_response.sessions)

    await session_service.delete_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    deleted_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=args.user_id,
        session_id=args.session_id,
    )
    describe_session("After Delete", deleted_session)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate InMemorySessionService and Session objects."
    )
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_demo(parse_args()))


if __name__ == "__main__":
    main()
