# InMemorySessionService Demo

This folder demonstrates the ADK `Session` object and
`InMemorySessionService` from the ADK sessions documentation:

https://adk.dev/sessions/session/

`InMemorySessionService` stores sessions in process memory only. It is useful
for local examples and tests, but sessions are lost when the Python process
exits.

## Inspect Session Without an LLM

This script creates, retrieves, lists, and deletes `Session` objects. It does
not require API keys.

```powershell
python -m in_memory_session_agent.inspect_session
```

It prints:

- `Session.id`
- `Session.app_name`
- `Session.user_id`
- `Session.state`
- `Session.events`
- `Session.last_update_time`

## Run an Agent With In-Memory Sessions

Copy `.env.example` to `.env` and set the key for the provider you use, then:

```powershell
python -m in_memory_session_agent.two_turn_demo
```

The script runs two turns in one process so you can see state and events update
inside the same in-memory session.

Custom turns:

```powershell
python -m in_memory_session_agent.two_turn_demo `
  --turn "Remember that my preferred format is bullets." `
  --turn "What format should you use?"
```
