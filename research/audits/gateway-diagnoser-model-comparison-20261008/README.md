# Local LiteLLM gateway: representative Diagnoser requests

Four user-authorized requests used exactly the same compressed E20 context and
unchanged strict Diagnoser prompt as the preceding InferAI comparison. The
independent gateway was `http://10.130.138.46:8010/v1`; Keychain credentials were
process-local and proxy variables were removed. `/v1/models` confirmed the exact
bare gateway model names. `openai/` is the local client's protocol prefix; wire
requests use the listed bare names, not an invented dashscope route.

| Model | Result | End-to-end seconds |
|---|---|---:|
| GLM-5.2 | Client timeout; no HTTP response | 182.74 |
| Kimi-K2.6 | Client timeout; no HTTP response | 180.04 |
| MiniMax-M2.7 | HTTP200, stop, parseable JSON; strict contract rejected | 25.98 |
| DeepSeek-V4-Flash | Client timeout; no HTTP response | 180.02 |

MiniMax returned 937 visible characters and 4,297 reasoning characters. Provider
usage reports 30,342 prompt and 1,095 completion tokens, and zero reasoning
*token* count despite nonempty reasoning *text*. Preserve this metadata discrepancy;
do not equate a zero reported reasoning-token count with no reasoning.
Its diagnosis incorrectly put task98 (a native failure) in protected_success_task_ids.
Strict validation stopped it with EvolverSchemaError; output was not rewritten.
The response's root-cause explanation is a model claim, not an established finding.
No mutation, screen, gate or promotion followed this standalone test.

Requests had max_tokens65536 and a 180-second client timeout, without automatic
retries or parallel calls. Wire args were model/max_tokens; no temperature override.
For Flash, the harness also passed thinking_mode enabled / reasoning_effort high
as local model args, but neither appears in the captured wire payload. This test
therefore does NOT verify an explicitly enabled Flash thinking mode; the captured
request and any provider defaults are authoritative. Do not treat it as a verified
thinking-mode comparison or promote its args directly to a formal config.

A client timeout is not an observed HTTP504 and does not prove permanent model
unavailability. The three timed-out requests had unavailable provider usage;
billed cost is unknown, not zero. The exact 180-second boundary is imposed by this
probe's client, unlike the previously observed InferAI server's 120-second504.

Four provider calls, zero native episodes, zero retries. Formal experiment remains
stopped. No runtime algorithm, benchmark data, frozen run or evaluator was changed.
Source context is `../diagnoser-model-comparison-20261008/context.json`; equality
was asserted against all four original per-call inputs. Per-call input bindings,
raw parsed MiniMax output, schema-error evidence and transport failures are retained
under calls/. Original full inputs remain in experiments/diagnostics.

Ruff and whitespace checks passed. These are live provider availability/contract
evidence only; no research-effectiveness result was obtained.
