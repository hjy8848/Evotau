# GPT6 representative readiness follow-up — transport succeeds, schema fails

The user explicitly requested the previously specified `openai/gpt-6.1-sol`.
All probes use the existing InferAI GPT-group Keychain binding and direct
`/v1/responses` adapter. Actual wire args are `model: gpt-6.1-sol` and
`reasoning: {effort: high}`. No tools, temperature, output allowance, silent
model fallback, automatic retry or output repair was added. Provider-reported
reasoning usage is recorded; backend weights and effort implementation are not
independently verified.

| Request | Context characters | Transport / visible JSON | Typed stage result | Client elapsed |
|---|---:|---|---|---:|
| Connectivity | Small | HTTP200, completed, legal JSON | Passed | about 3.1s |
| Diagnoser | 181,114 | HTTP200, completed, legal JSON | **Failed: unknown recommended mutation** | 67.67s |
| Mutator | 181,708 | HTTP200, completed, legal JSON | Passed | 40.21s |
| Skill Validator | 9,242 | HTTP200, completed, legal JSON | Passed; semantic rejection | 14.85s |
| Customer Validator | 13,568 | HTTP200, completed, legal JSON | Passed; semantic rejection | 18.35s |

The same four frozen context hashes were used in the Pro and V4.1 Flash probes.
All V2 prompts and runtime source hashes stayed unchanged. Only the diagnostic
helper's credential routing/observer changed to use the repository's existing
GPT route explicitly, with regression coverage. No evolution algorithm changed.

Diagnoser emitted labels such as `action_scope_validation` and
`replacement_mapping_contract_clarification`. The strict schema instead allows
`add,narrow_trigger,expand_trigger,rewrite_guidance,split,delete,no_op`.
The original Diagnoser prompt does not list that enum. This is a directly
observed prompt/schema contract gap, not a timeout or truncated JSON. The model's
root-cause analysis remains unverified inference; no native backend or evaluator
was changed based on it. Raw visible output, parsed JSON and typed failure are
preserved separately. A provider ledger success denotes a completed response,
not acceptance by the stage schema.

The four stage probes are independent checks. Mutator uses a declared fixture,
not the failed Diagnoser's output. Validators test declared fixtures and their
negative booleans are legitimate semantic rejections. These checks do not show a
connected diagnosis → mutation → screen → gate completion or an effective skill.
Preflight exits 2, and the independent config remains disabled.

Five inference requests total; all report usage: **133,589 prompt tokens** and
**3,933 completion tokens**. The four representative probes take about **141s**
sequentially. No native episodes, accepted skills or formal experiments were
started. No held-out task content was newly loaded. Offline tests: **227 passed**,
5 dependency warnings; config dry validation starts no real requests.

See [`summary.json`](summary.json), sanitized request/output directories,
[`config.yaml`](config.yaml), and [`sha256-index.json`](sha256-index.json).
Runtime source remains frozen; a prospective prompt-contract fix requires separate
provenance rather than silently altering this model-only comparison.
