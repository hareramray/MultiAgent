# ADK Model Cost Control Callback Demo

This folder demonstrates ADK model callbacks from:

https://adk.dev/callbacks/

The demo uses:

- `before_model_callback` to estimate request size before spending model cost.
- `before_model_callback` to return an `LlmResponse` and skip the real LLM call
  when the prompt or session exceeds the configured budget.
- `after_model_callback` to record rough response-size telemetry and optionally
  append a visible cost note.

## Run Without an API Key

The default prompt is deliberately oversized, so the callback returns a local
`LlmResponse` before ADK calls the model.

```powershell
python -m model_callback_agent.callback_demo
```

## Run With a Real Model Call

Copy `.env.example` to `.env`, set the key for your provider, then pass a
short message that stays within the budget.

```powershell
python -m model_callback_agent.callback_demo "Explain model callbacks in one sentence."
```

You should see prompt estimates, model-call counts, and `callback_events`
printed from session state.

## Budget Settings

Edit `.env` to change the demo thresholds:

```powershell
MODEL_CALLBACK_MAX_PROMPT_CHARS=700
MODEL_CALLBACK_MAX_PROMPT_TOKENS=220
MODEL_CALLBACK_MAX_MODEL_CALLS_PER_SESSION=2
MODEL_CALLBACK_APPEND_AUDIT_NOTE=true
```

## ADK Web

```powershell
adk web
```

Select `model_callback_agent`, then try:

```text
Explain model callbacks in one sentence.
Summarize this long text: budget budget budget budget budget budget budget...
```
