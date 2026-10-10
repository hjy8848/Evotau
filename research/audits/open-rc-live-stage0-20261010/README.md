# Open-RC real launch: free audit and authorized Pair Acquisition

Baseline: `03e80ed2c282a8b5b22fdb799f293fb18a7cda7c`, initial working tree clean.
The local checkout is `codex/airline-failure-analyst`; the delivery target is
`codex/alternating-evolution-refactor`. A new remote read timed out; the prior
push verified the target at 03e80ed. The next commit/push records this launch work.

## Actual Stage 0 validation (not historical test claims)

- Full pytest executed this turn: **503 passed, 6 dependency warnings, 42.55s**.
- New acquisition launcher tests: **7 passed**, no real requests.
- Four existing configs: dry validation passed, pinned Airline E10/source hashes
  verified by native loader; V/H content was not loaded by these four dry checks.
- Acquisition config: separate E10/V20, G1/C1/P2, max_steps=200, H disabled.
  Its dry launch loads E/V, never H, and checks source hashes/authorization.
- Existing H result metadata may be present in old full result JSON files read
  to access their generation section. No H task content/trajectory or H score
  was used to select Pair/tasks or estimate cost.
- Ruff and diff whitespace checks passed. A second full run saved in `pytest-final.log`: **509 passed, 6 warnings, 43.78s** before the final wire-guard addition; all **7** final launcher tests then passed (the baseline 503 tests are unchanged).

The initial audit JSON predates the budget reply. The user then explicitly
replied **“无上限”**, superseding the unfilled monetary budget and finite-budget
restriction in the attached task. See `budget-authorization.json`. Actual monetary
cost is still unknown for the private Qwen gateway; it will not be invented.
No request/output-token/spend caps are added under this authorization.

## Repair Pair prerequisite

Four completed real Airline generation records from two unique runs were scanned.
All had Service ID `f0a9e404e39ab085` before and after (empty memory).
No deployed, formally promoted real Airline Pair was found. Unpromoted proposals,
Retail results and Synthetic artifacts are not used as a Pair.

Therefore the authorized chargeable next stage is **Pair Acquisition**, not A/B/C:

- Same declared gateway Qwen3.7-plus Agent/Customer/Evaluator/Activator and
  official `openai/deepseek-flash` thinking/high Evolver as the templates.
- Existing Customer Skill v1 and Service Analyst/Mutator, unchanged native scoring,
  unchanged frozen V-primary gate, no score imports and no gate relaxation.
- One generation, one Customer proposal, existing three Service proposals and
  crossover budget. E10 repair / independent V20 with two gate seeds / H closed.
- A/A is uncalibrated; this is exploratory acquisition, not calibrated publication
  evidence. Promotion must still satisfy the frozen gate.
- Two minimal no-tools JSON preflights are separately charged/logged. They verify
  routing/basic visible JSON only. Actual Analyst/Mutator/Validator calls must
  establish representative-context reliability; short probes cannot do that.
- A completed acquisition without promotion is a genuine prerequisite failure:
  stop, save negative result, do not relabel ordinary search as Open-RC.
- HTTP/JSON/runtime errors stop with evidence. Only the existing frozen, bounded
  format-recovery protocol may run; no model switching or blind replay.

## Cost planning — not a currency quote or guaranteed bound

Using **16 actual E-only Airline episode telemetry records** at the same declared
runtime/model/max_steps, mean usage was 22.0625 calls, 122,012.9375 prompt tokens,
3,998.1875 completion tokens per episode. The p90 was 35 calls / 233,999 prompt /
7,037 completion. These mix native and candidate memory conditions and do not
measure a new Open-RC treatment. Concurrent global budget deltas were NOT used;
per-episode attribution was used.

For the proposed E3/K2/R2 comparison, cold unique native conditions are at most 81
(72 feedback + 9 incumbents; candidate fitness is reused only if exactly identical
within its own manifest). Planning projections for runtime alone:

| Scenario | Calls | Prompt tokens | Completion tokens |
| --- | ---: | ---: | ---: |
| historical mean | 1,787 | 9,883,048 | 323,853 |
| historical p90 | 2,835 | 18,953,919 | 569,997 |
| historical sample max | 2,997 | 20,059,164 | 735,318 |

These are NOT hard bounds. Longer dialogs/nonempty memory can exceed them.
Six generators + six static validations + up to 90 per-cell trajectory validations
+ six discovery reviews = 108 Evolver calls before bounded format recovery;
if all requests undergo two allowed recoveries, up to 324 attempts. Invalid
candidates may skip downstream execution. Provider usage of these new stages is
not yet measured. Probes, A/A diagnostics, Pair acquisition/revalidation, failed
attempts and any approved recovery must be charged separately.

Current official Flash price reference:
https://api-docs.deepseek.com/quick_start/pricing/
Peak cache-miss input/output: **USD 0.30 / 1.20 per million tokens**;
off-peak: **USD 0.15 / 0.60**. Cache discounts are not assumed in estimates.
Use actual per-call price/time/cache status; conversion and payment fees remain
unknown. Gateway Qwen price is not established, so total RMB cost is **N/A**.
A/B/C outcomes/discoveries and unit-discovery cost remain **N/A** until actually run.

The G2 alternating executor changes Service and cannot fairly execute the frozen
Pair A/B/C comparison. A dedicated read-only frozen-Pair search entry remains
necessary before Stage 2. The draft preregistration is not an executable manifest.

## Execution

```bash
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
/Users/spring/RSI/Evotau/.venv/bin/python \
experiments/execution/run-open-rc-acquisition.py \
--config configs/airline-open-rc-pair-acquisition-g1-c1-p2.yaml \
--tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
--execute --approve-unbounded-requests
```

Unique run directory:
`experiments/runs/evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010`.
`preflight/api-usage-live.json` is separate from the main run usage; report their
sum, not only the main counter. Raw visible preflight output is persisted BEFORE
schema validation; unresolved preflight intent never triggers automatic replay.
Source provenance, native episodes and gate stages are immutable under this run.

## Initial real preflight and wire correction

The initial run at commit `95e2e44` sent **one real Qwen request**, HTTP200,
31 input/5 output tokens, valid JSON and finish_reason=stop. Its DeepSeek attempt
was blocked before HTTP by the thinking/high wire check. No native episode ran.
The ledger conservatively has two attempts (one successful remote request, one
local pre-network failure). Complete evidence is in `initial-real-preflight/`.
No uncertain remote request is replayed.

An offline mock of the exact τ generate/LiteLLM chain found that `thinking=enabled`
was present but `reasoning_effort=high` was dropped. τ enables LiteLLM drop_params;
this model alias is unrecognized as an OpenAI reasoning model. The official API
supports high: https://api-docs.deepseek.com/guides/thinking_mode/.
EvoTau now also passes the declared effort in extra_body ONLY for the official
api.deepseek.com route. Top-level setting is retained, Qwen and other providers
are unchanged. Actual wire mock asserts explicit high and enabled thinking.
This fixes transmission of the original declared setting; it does not change a
prompt, search rule, model, task, gate or native τ source.

After this fix: **512 tests passed, 6 dependency warnings, 42.93s**; focused native
wire/budget/launcher checks: **38 passed**; Ruff passed. See `pytest-wirefix.log`.

Use the independent continuation config:
`configs/airline-open-rc-pair-acquisition-g1-c1-p2-wirefix1.yaml`.
It has a new run directory, manifest/source identity and checkpoints. The only
compatible import is the already complete Qwen no-tools probe (exact model/args,
original response/request references and full usage ledger). There are no episode
scores to import. Parent DS intent is not imported; its pre-network failure is
retained in costs. Stage0/Stage1 still cannot be called an Open-RC H1/H2 result.
