# Phase 0 · InferAI DeepSeek V4 Pro · thinking disabled · 2026-09-29

This one-episode smoke used τ-bench Retail task `73`, seed `42`, pinned τ-bench commit `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, and the same frozen prompt hashes and 24-request cap as the V4 Flash baseline. All four roles used InferAI model `deepseek-v4-pro`; thinking was disabled for every role. The successful provider responses confirm this model ID is accepted by the configured endpoint.

The agent asked the customer to authenticate, received the email, then emitted explanatory text together with a `find_user_id_by_email` tool call. The pinned τ-bench half-duplex validator rejects this format, so the episode ended at the first tool action with `agent_error` and native reward `0.0`. This reproduces the V4 Flash thinking-disabled failure at the same protocol boundary. The native reviewer did not flag the orchestration error. One episode does not establish a general model-capability comparison.

All six provider requests succeeded (14,775 prompt tokens, 272 completion tokens). LiteLLM reported no price mapping for this model; the native cost value is `0.0`, so the provider-billed amount is unknown.

`native-simulation.json` retains visible messages, tool calls, usage, and evaluation. No provider reasoning fields were present. No API key or credential value is included. This CLI run did not produce an `events.jsonl`.
