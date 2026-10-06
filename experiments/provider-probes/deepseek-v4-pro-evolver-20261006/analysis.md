# DeepSeek V4 Pro provider capability probe

## Decision: Stage A failed; stop

Probe 1 sent one small OpenAI-compatible Chat Completions request through the repository's existing `tau2.generate()` / LiteLLM path to `https://inferaiapi.com/v1`, requesting `openai/deepseek-v4-pro`. InferAI/LiteLLM returned `NotFoundError`: `Model "deepseek-v4-pro" is not supported by any configured account in this group`.

- Latency: 2.403 seconds.
- Calls / retries: one generate invocation; `num_retries=0`.
- Requested reasoning parameters: `extra_body.thinking.type=enabled`, `reasoning_effort=high`.
- The provider returned no completion, so JSON parsing, usage, and reasoning activation could not be verified. The numeric HTTP status was not exposed by the captured LiteLLM exception.
- The request used the macOS Keychain entry named `openai-gpt-api-key`; no secret is included in these artifacts. The provider error indicates that the model is not available to an account in that key's current group. This points to model-group access/routing, not to SkillMemory behavior. It does not establish whether another InferAI group supports the model.

Probe 2 was not sent. The archived 20-task context was not sent or modified. Stage B config was not created; no Customer, Service, evaluator, E, V, H, or alternating generation was started. No model fallback or retry occurred.

The first local harness attempt is retained separately under `deepseek-v4-pro-evolver-20261006-local-harness-error/`; its telemetry confirms zero provider calls. The actual provider rejection is recorded in `small/` in this directory.
