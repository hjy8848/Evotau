# Phase 0 · InferAI DeepSeek V4 Flash · 2026-09-29

This directory archives the first provider-enabled EvoTau Phase 0 smoke run. It is intended for inspection and failure analysis; this episode is **not a successful task completion or research result**.

The run used τ-bench Retail task `73`, pinned τ-bench commit `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, and InferAI model `deepseek-v4-flash` with thinking disabled. All four configured roles used the same provider model.

The provider accepted all 6 requests (budget cap 24), reporting 14,761 prompt tokens and 271 completion tokens. The episode ended with `agent_error` and native reward `0.0`: the assistant response included explanatory text and a tool call in one turn, which the pinned τ-bench communication validator rejects. The task therefore did not finish. This is an integration observation, not a verified policy failure or study finding.

The native record's cost value is `0.0`; this is not confirmation of the amount billed by InferAI.

`config.yaml`, `manifest.json`, `phase0-result.json`, `native-simulation.json`, and `events.jsonl` are unmodified copies of the corresponding files from the local run folder. `analysis-summary.json` provides a compact machine-readable index. Synthetic customer details in the trajectory are from the τ-bench fixture. No API key or credential value is included.
