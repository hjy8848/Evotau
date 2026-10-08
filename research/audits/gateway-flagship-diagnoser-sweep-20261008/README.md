# Gateway flagship Diagnoser sweep

User authorized testing strong models on the independent LiteLLM gateway.
Scope: flagship text/reasoning/coding families, omitting duplicate dated aliases
and image/audio/embedding models. This is not an independently verified ranking
of model strength. Exact IDs were confirmed in refreshed gateway `/v1/models`.

Two isolated batches: 15 flagship routes (peak2) and three standard GPT routes
(peak1, after the first batch completed). Existing four-model results are reused,
not rerun. Exactly **18 new HTTP requests**, **4 prior records**, zero native
episodes and zero retries. Maximum global request concurrency2. All requests
used the same ~30k-token compressed Diagnoser context and unchanged strict prompt.
Output allowance65536, client timeout180s, no tools, no temperature/reasoning
override. OpenAI-compatible client prefix is added outside the exact gateway ID;
wire request IDs and responses are retained. Provider defaults apply.

| Gateway ID | Category | Seconds | Provenance |
|---|---|---:|---|
| `anthropic/claude-opus-4-7` | BLOCKED | 7.08 | new |
| `anthropic/claude-opus-4-8` | BLOCKED | 7.08 | new |
| `anthropic/claude-opus-4-6` | BLOCKED | 0.07 | new |
| `anthropic/claude-fable-5` | BLOCKED | 0.08 | new |
| `anthropic/claude-sonnet-4-6` | BLOCKED | 0.06 | new |
| `anthropic/claude-sonnet-5` | BLOCKED | 0.09 | new |
| `openai/gpt-5.6-sol` | BLOCKED | 0.07 | new |
| `openai/gpt-5.6-terra` | BLOCKED | 0.07 | new |
| `openai/gpt-5.5-pro` | NO_DEPLOYMENT | 0.28 | new |
| `openai/gpt-5.4-pro` | NO_DEPLOYMENT | 0.26 | new |
| `openai/gpt-5.3-codex` | NO_DEPLOYMENT | 0.13 | new |
| `openai/o3-pro` | NO_DEPLOYMENT | 0.14 | new |
| `dashscope/qwen3-max` | CLIENT_TIMEOUT_180S | 180.02 | new |
| `dashscope/qwen3-coder-plus` | CLIENT_TIMEOUT_180S | 180.03 | new |
| `DeepSeek-V4-Pro` | INVALID_MODEL | 0.38 | new |
| `openai/gpt-5.6` | BLOCKED | 6.88 | new |
| `openai/gpt-5.5` | BLOCKED | 0.07 | new |
| `openai/gpt-5.4` | BLOCKED | 0.10 | new |
| `GLM-5.2` | CLIENT_TIMEOUT_180S | 182.74 | prior, reused |
| `Kimi-K2.6` | CLIENT_TIMEOUT_180S | 180.04 | prior, reused |
| `MiniMax-M2.7` | CONTRACT_REJECTED | 25.98 | prior, reused |
| `DeepSeek-V4-Flash` | CLIENT_TIMEOUT_180S | 180.02 | prior, reused |

BLOCKED: server reported model blocked. NO_DEPLOYMENT: HTTP429 explicitly says
no deployments available; this is not evidence that request concurrency exceeded
RPM. INVALID_MODEL: V4 Pro is listed but inference returned404/invalid model.
CLIENT_TIMEOUT_180S: client waiting deadline; no HTTP status observed, not504.
CONTRACT_REJECTED: MiniMax's previously returned HTTP200/stop JSON labels native
failed task98 as a protected success; strict validation rejects it unchanged.

No tested route passed the full diagnosis contract. Only the prior MiniMax
request returned a complete parseable diagnosis. No outputs were repaired and
no formal run was resumed or reconfigured. No mutation quality, gate outcome
or evolutionary effectiveness was measured. Timed-out/rejected network requests
without usage metadata have unknown billed tokens/cost, not zero cost.

Only models actually tested are covered; this does not prove every model offered
by the gateway is unavailable. Listed aliases can be blocked or lack a deployment.
The current practical blockers include access/deployment problems as well as
long-request deadlines; changing context length cannot fix a blocked model.

Shared input is ../diagnoser-model-comparison-20261008/context.json. Equality
was asserted for every saved per-call input. Batch report/usage/wire logs and
per-call input bindings plus immutable failure artifacts are stored separately.
The scripts refuse to duplicate requests if a report already exists. Ruff and
whitespace checks passed; no algorithm code changed.
