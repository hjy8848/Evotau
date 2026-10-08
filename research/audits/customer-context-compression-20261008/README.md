# Customer Evolver context compression — offline audit

The archived E20 Customer input is rebuilt deterministically without any provider calls.

| Metric | Before | After |
|---|---:|---:|
| Estimated input tokens including system prompt | 117,580 | 28,439 |
| E tasks visible | 20 | 20 |
| Full native results modified | 0 | 0 |

Reduction: **75.81%**, measured with `cl100k_base` as a proxy, not billed DeepSeek tokens.
Overview including unchanged full Customer scenarios and traceability metadata: **10,343 tokens**.
Detailed evidence: **16,220 tokens**. The overall 20k–30k target is met on this archive;
the aspirational 3k–5k overview and 10k–15k evidence targets are not individually met.

## Frozen evidence and selection

The uncompressed source input remains at the `source_input` path in measurement.json;
its original file digest is recorded. compressed-input.json is an offline comparison,
not a live request. All task IDs, seeds, scenarios, native results and trajectory refs
are retained. All task tool paths, initial/latest user prefixes and explicit JSON tool
errors are listed, without inventing failure causes.

Four detailed cases are selected deterministically by outcome stratum and task ID:
failed tasks **35, 54, 98**, and successful task **106**. The success control is an
outcome-stratified identifier choice, not evidence that it matches every failure family.
No model performance search is used to choose cases.

Every user message and state-changing tool-call parameter in selected cases is exact.
The adjacent action-result block is conservatively retained when old projections lack
ToolMessage IDs. Bulky Retail order results keep exact JSON fields including status,
order ID, cancellation/exchange/return fields and errors. All returned `items` are retained. Omitted `address`,
`fulfillments` and `payment_history` fields are explicitly listed; originals remain on
disk. Read-only results may be exact-prefix excerpts. Other omitted message indices
are listed. Original projection/message/content digests make these projections auditable.
An allowance overflow for required user/action evidence raises an explicit local error;
it cannot silently remove required evidence.

Each of the three observed failure cases passes the mechanical reconstruction checks:
all user messages/action parameters and retained action-result fields match the source.
This verifies evidence fidelity, **not** causal diagnostic quality. Missing inventory
or payment-history details may still matter for a later diagnosis; an Evolver must not
invent them. Diagnoser/Mutator inputs are unchanged in this first step.

## Integration and validation

Only V2 generation Customer Evolver and final E-only fresh Customer generation use the
new builder. Customer Validator receives the same full original scenarios. Service
Diagnoser, Mutator, Skill Validator, Screen, Gate, evaluator, tasks, native runtime,
fitness/cache identity and budgets remain unchanged. H content is not loaded.

Validation: **44 tests passed** (new context tests plus V2 deterministic and native
integration tests), one dependency deprecation warning; Ruff and diff checks pass.
Provider calls for this work: **0**. No experiment resumed, model changed, or live
request sent. There is no measured wall-clock/provider quality or total-cost saving.
Native Qwen episode cost remains the principal separate cost concern.

Code changes create a new runtime source fingerprint. The old frozen experiment is
preserved and cannot simply be resumed under changed code; any future authorized run
needs a separate audited continuation that verifies native episode compatibility.

Reproduce from repository root:

```bash
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python research/audits/customer-context-compression-20261008/measure.py
```
