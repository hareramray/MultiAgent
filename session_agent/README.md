# ADK Session Demo

This folder demonstrates ADK sessions with a small agent that can remember,
recall, and forget facts inside a stable session.

The session key is the combination of:

- `app_name`
- `user_id`
- `session_id`

Using the same values keeps the conversation history and `tool_context.state`.
Changing the session id starts a separate session.

## ADK Web

```powershell
adk web
```

Select `session_agent`, then try:

```text
Remember that my name is Ada and I prefer concise answers.
What do you remember about me?
Forget my name.
What do you remember now?
```

## CLI Demo

```powershell
python -m session_agent.session_demo --session-id demo "Remember that my name is Ada."
python -m session_agent.session_demo --session-id demo "What do you remember?" --show-state
python -m session_agent.session_demo --session-id other-demo "What do you remember?" --show-state
```

The first two commands share state because they use the same `session_id`.
The third command uses a different session.
