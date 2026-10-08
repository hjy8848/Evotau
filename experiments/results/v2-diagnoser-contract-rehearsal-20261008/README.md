# V2 Diagnoser contract correction and end-to-end rehearsal

Baseline: `f65dd4d3ef38123485138e06002d4a66c7df092e`.
Initial correction and frozen live runtime: `3f6ac10`.
Additional case-reference contract and frozen follow-up runtime: `1292988`.

## Scope and contract

The Diagnoser prompt now separates natural-language root causes from closed
repair surfaces and mutation operations. Allowed surfaces are `skill`,
`tool_boundary`, `runtime_protocol`, `stochastic_or_weak`. Allowed mutation
operations are `add`, `narrow_trigger`, `expand_trigger`, `rewrite_guidance`,
`split`, `delete`, `no_op`. Invented mechanism/category names are explicitly
forbidden in the operation array; non-skill diagnoses may leave it empty.

The strict operation check is preserved, with explicit JSON string-array
validation. The original GPT6 output from the baseline still fails; there is no
alias mapping or silent repair. Mutator and Crossover now validate their existing
proposal schema at the provider boundary, before a completed stage is published.
A malformed provider proposal therefore fails closed instead of being classified
as a low-quality research candidate. Semantic rejection, invalid applicability,
NO_OP and ordinary scored benchmark failure keep their existing behavior.

The first live rehearsal also exposed an unstated Mutator field contract:
`expected_fixes` contained natural-language prose instead of task IDs. Its
original output and rejection remain preserved. The prompt now explicitly states
that `expected_fixes` / `protected_cases_at_risk` contain observed task ID strings,
with explanations/counts in `analysis`. Provider-boundary validation rejects
unsupported case references before stage publication; it does not translate prose
or repair output. Crossover case references use the supplied cluster/parent
effect evidence. Candidate fitness and acceptance logic are unchanged.

No τ-bench task, policy, tools, backend, evaluator, metrics, selection rule,
Screen or Gate was changed. No model fallback or automatic retry was added.

## Deterministic verification

Final full test suite: **261 passed**, five dependency warnings, 18.05 seconds. Ruff
checks pass. New tests exercise every legal enum, invented labels, invalid array
types, non-skill empty recommendations, and the original recorded GPT6 rejection.

An explicit strict-provider orchestration test completes Customer → Customer
Validator → Diagnoser → Mutator → Skill Validator → Screen → Gate → Archive.
Replay issues no additional provider-boundary calls or deterministic episodes,
and frozen journal bytes remain identical. This test uses scripted research
outputs and deterministic episode outcomes; it is not live efficacy evidence.

Pinned native offline integration exercises UserSimulator, LLMAgent, native
Retail tools/backend/evaluator and Activator with local scripted completions.
Five native episodes, 17 local completion calls, no network calls, H not loaded;
completed replay adds zero calls. The exported test command also checks failures
from malformed JSON, schema validation, provider timeout, budget exhaustion and
native episode exceptions. Execution state records the failing stage and error
type. Completed evidence remains byte-identical on explicit recovery.
An invalid cached diagnosis remains invalid on resume, with zero new calls.

An additional integration test replays the exact live GPT proposal and structured
stage outputs, without modifying the proposal, through deterministic paired
episode fixtures to Screen → Gate → Archive. This establishes legal-proposal
plumbing, not native accuracy. Its synthetic test outcomes are never copied to a
real run, used as fitness or represented as native evaluator results.

## Frozen live configuration

`configs/v2-gpt61sol-contract-e3-rehearsal.yaml`: E=`98,8,66`, G=1, Customer
candidates=1, Service proposals per cluster=1, maximum clusters=1, crossover off,
P=2, steps=32, screen/gate seeds=`1,2`, fitness seed=1, finite request cap=600,
V/H disabled. E uses previously observed diagnostic failures and a passing control;
it is an engineering coverage panel, not a representative research sample.
No previous configuration's score or episode artifact is imported.

Follow-up config: `configs/v2-gpt61sol-contract-e3-rehearsal-v2.yaml` has a new
directory and cap533; the first rehearsal used67 requests. The combined native
rehearsal allocation is600, not600 per retry. Actual requests across both runs
plus the separately bounded three-request GPT diagnostic total76, below600.
Neither config nor frozen native source was modified during its execution.
After all probes stopped, the three GPT diagnostic requests were explicitly
charged to the follow-up's cumulative budget via the existing `absorb_usage`
mechanism, with source hashes and counters recorded. This imports cost telemetry
only, no episode, score, proposal or search stage. The follow-up ledger is now9
calls (six native-attempt calls plus three diagnostic calls); together with the
first67, it allows at most600 aggregate calls and leaves524. The original failed
six-call ledger is retained separately in the archive. No calls were issued
during accounting or ledger restoration.

Native Agent/Customer/Evaluator/Activator use `openai/qwen3.7-plus`, temperature0,
InferAI chat completions, thinking disabled. Evolver uses `openai/gpt-6.1-sol`,
InferAI Responses, GPT-group Keychain, `reasoning_effort=high`, no temperature.
Credential values are never included in artifacts. Refreshing `/v1/models`
confirmed both exact IDs available to their respective groups; listing evidence
does not guarantee inference completion or independently identify backend weights.

Launch / explicit resume under this frozen configuration:

```sh
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
HTTPS_PROXY=http://127.0.0.1:65533 HTTP_PROXY=http://127.0.0.1:65533 \
NO_PROXY=localhost,127.0.0.1 \
/Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-gpt61sol-contract-e3-rehearsal-v2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --approved-request-cap 533
```

This command is for the independently authorized small rehearsal only. A future
formal E20/G2 run requires its own frozen config, independent validation/test
plan, finite confirmed budget and explicit authorization. Finite-panel acceptance
cannot establish population generalization or a 5% population harm bound.

## Live outcome and readiness judgment

| Check | Live result | Missing coverage |
|---|---|---|
| First E3 rehearsal (`3f6ac10`) | Three native episodes; Customer → Validator → Diagnoser → Mutator → Archive/checkpoint completed; Customer semantically rejected; Diagnoser enum valid; Mutator case-reference rejection | Skill Validator, native Screen/Gate |
| Follow-up E3 (`1292988`) | Stopped fail-closed at `customer_incumbent`: Qwen HTTP429; six calls, three successes and three failures, zero completed episodes | All later native-connected stages |
| Three-request GPT chain (`1292988`) | Diagnoser → actual Mutator proposal → actual Skill Validator completed; legal observed task IDs; semantic acceptance=true | Native Screen/Gate, fitness, efficacy |

The GPT chain is a provider diagnostic on historical raw native evidence, with
declared source/context hashes. It imports no old score into a new search condition,
runs no episode, computes no accuracy and never substitutes a fixture for fitness.
Its real proposal passes the separate deterministic Gate integration test above.
It is **not** a successful native end-to-end rehearsal. The failed Qwen run is
preserved as failed, not relabeled complete; 429 retries and model fallbacks:0.

First-run completed replay issued zero additional requests and preserved27
completed journal/episode artifacts byte-for-byte. After the failed follow-up,
budget restoration issued zero calls: attempts6, failures3, in_flight0, reserved0.
There were no completed stages/episodes in that failed attempt to replay; live
completed-stage recovery after an error is therefore not claimed. The deterministic
failure-injection tests provide that coverage.

Consumption: **76 requests**,73 completed responses,3 HTTP429 failures;
**452,001 reported prompt tokens**, **12,760 reported completion tokens**.
Failed429 usage is unavailable for three calls, not zero billable cost.
First live run298.53s, failed follow-up19.92s, connected GPT probes101.09s;
about419.54s active execution in total, excluding preparation and completed replay.
Native completed episodes:3; Reviewer/Customer Judge/Service Judge calls:0.

**Formal readiness: not established.** The contract fixes and deterministic
stage/recovery workflow are verified; live GPT structure and semantics are verified.
The remaining immediate blocker is Qwen runtime availability/rate limiting and
production-path native Skill Validator → Screen → Gate coverage. No skill repair
effect or H generalization has been verified. Formal E/V/H statistical limitations,
an explicitly confirmed finite formal budget and formal launch authorization remain
as recorded in the earlier readiness audit. H has not been loaded.

Do not resume automatically. Once native provider availability is confirmed, an
explicit frozen follow-up resume uses the launch command above with config
`v2-gpt61sol-contract-e3-rehearsal-v2.yaml` and cap533. Completed compatible artifacts
are reused by the runtime; failed attempts are not scored. A hard-killed in-flight
call still requires manual ledger reconciliation rather than discarding possible
cost. No full E20/G2 experiment was launched.

Sanitized immutable evidence and SHA index:
`experiments/results/v2-diagnoser-contract-rehearsal-20261008/`.
