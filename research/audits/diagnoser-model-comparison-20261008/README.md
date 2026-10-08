# Representative InferAI Diagnoser model comparison

User authorized testing several alternative models after V4 Pro timed out.
Each model received exactly the same compressed E20 Diagnoser context and the
unchanged strict Diagnoser prompt, through existing EvoTau provider dispatch.
Model membership was refreshed from both InferAI credential-group `/v1/models`
endpoints before testing. Listing access does not imply successful inference.

| Model | Result | End-to-end latency | Validation |
|---|---|---|---|
| openai/glm-5.2 | HTTP 504 | 123.19s | No visible JSON; schema not reached |
| openai/kimi-k2.7 | HTTP 504 | 120.15s | No visible JSON; schema not reached |
| openai/qwen3.7-max | HTTP 504 | 120.14s | No visible JSON; schema not reached |
| openai/gpt-6.1-sol | HTTP 504 | 120.57s | No visible JSON; schema not reached |

Four provider requests; zero retries and zero native episodes. No formal run
configuration, task, model, evaluator or fitness score was replaced. A pre-request
observer setup AttributeError was corrected before the first request, with a
zero-call startup failure record retained.

GLM/Kimi/Qwen used existing chat completion dispatch with `max_tokens=65536`;
no temperature, thinking or reasoning override was sent, so provider defaults
apply. GPT used the existing GPT-group Responses adapter with reasoning high,
no temperature and no added output allowance override. Keys stayed process-local
and are absent from these artifacts. Actual wire arguments are recorded in
`actual-provider-http.jsonl`. Every failure was HTTP504; GPT is classified by
its adapter as RuntimeError, whereas chat failures are LiteLLM Timeout exceptions.

All requests failed near the same 120-second interval, across chat and Responses.
This is consistent with a shared upstream timeout boundary; it does not establish
whether input processing, reasoning/output time, load or another upstream issue
caused the timeout. No returned usage was available, so billed tokens/cost remain
unknown, not zero. No diagnosis quality or repair effectiveness was measured.

`context.json` stores the single shared context. `calls/*/input-binding.json`
stores each original input metadata/digest, alongside immutable failure evidence.
Full original per-call inputs remain in the local diagnostics directory.
`report.json` and the budget/wire records retain all four outcomes. The runner
refuses to run again when a report already exists; no automatic recovery/retry.

Formal experiments remain stopped. No fallback model was promoted from this test.
