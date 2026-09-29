# Phase 0 · InferAI Qwen 3.7 Max · thinking disabled · 2026-09-29

This one-episode smoke used τ-bench Retail task `73`, seed `42`, pinned τ-bench commit `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, the same frozen prompt hashes, and a 24-request cap as the V4 Flash baseline. All four roles used `qwen3.7-max` through the InferAI OpenAI-compatible endpoint with thinking disabled. The successful provider responses confirm that the endpoint accepts this model ID.

After asking for and receiving the customer's email, the agent emitted explanatory text together with a `find_user_id_by_email` tool call. The pinned τ-bench half-duplex validator rejects this format, so the episode ended at the first tool action with `agent_error` and native reward `0.0`. This is the same failure point observed with V4 Flash and V4 Pro under disabled thinking. Similar outcomes across model IDs make the shared prompt/protocol integration worth investigating, but these single episodes cannot establish the cause. The native reviewer did not flag the orchestration error.

All six provider requests succeeded (14,809 prompt tokens, 295 completion tokens). LiteLLM reported no price mapping for this model; the native cost value is `0.0`, so the provider-billed amount is unknown.

`native-simulation.json` retains visible messages, tool calls, usage, and τ-bench evaluation. No provider reasoning fields were present. No API key or credential value is included. This CLI run did not produce an `events.jsonl`.
