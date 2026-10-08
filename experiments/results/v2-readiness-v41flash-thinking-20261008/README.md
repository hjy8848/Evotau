# V4.1 Flash thinking-enabled replacement — representative preflight failed

User explicitly requested V4.1 Flash with thinking enabled to replace Pro.
InferAI's current DeepSeek-group `/v1/models` lists `deepseek-v4.1-flash`; the
public model-plaza listing also includes it. These are access/listing evidence,
not a guarantee that inference will succeed.

Only the Evolver model is changed. Qwen Customer/Service/native evaluator and
Activator, native policy/backend/tools, V2 prompts, tasks, seeds and gate remain
unchanged. Runtime source SHA is preserved in the probe plan. The independent
[`config`](../../../configs/v2-readiness-v41flash-thinking-e5.yaml) is disabled
after the failed tests; the E20/V16/H40 formal draft remains untouched.

Actual wire: `model=deepseek-v4.1-flash`, `thinking.type=enabled`,
`max_tokens=65536`, no tools, zero automatic retries. `reasoning_effort=high` is
requested through the existing client, but is not present on the actual wire;
high effort is not independently verified.

| Request | Context characters | Outcome | Client elapsed |
|---|---:|---|---:|
| Tiny connectivity JSON | Small | HTTP200, valid JSON, stop; 41 reported reasoning tokens | 8.37s |
| Diagnoser | 181,114 | HTTP502, no usable output | 29.81s |
| Mutator | 181,708 | HTTP502, no usable output | 0.21s |
| Skill Validator | 9,242 | HTTP502, no usable output | 0.12s |
| Customer Validator | 13,568 | HTTP502, no usable output | 0.30s |

All four representative contexts have exactly the same hashes as the earlier Pro
audit. Mutator uses the same explicitly declared readiness-only cluster fixture;
it is not the output of a successful live Diagnoser. The helper's failed report
returns exit code 2. The four requests are different independent stage checks,
not repeated retries of a failed stage. Short connectivity passed; representative
readiness did not. Validators also failed, so length alone is not established as
the cause. Provider routing/availability/root cause remains unknown.

Five inference requests total: one successful, four failed. Only the successful
request reports usage: 245 prompt tokens, 47 completion tokens. Failed-call usage
is unavailable, not zero cost. No baseline cache was imported, no native episode
was started, no skill was promoted, and no formal E/V/H experiment was launched.
The initial standalone connectivity script had an import-path initialization
error before dispatch; it was corrected with zero provider retries and retained
as separate initialization evidence.

See [`summary.json`](summary.json), the two probe directories, and
[`sha256-index.json`](sha256-index.json) for sanitized request/status/output
evidence. The tested enabled configuration is retained separately from the current
disabled config; it was used only to select the provider args for these no-tools
probes, never to start native simulation.
