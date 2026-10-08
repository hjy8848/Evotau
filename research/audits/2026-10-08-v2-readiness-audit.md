# V2 pre-formal readiness audit — 2026-10-08

Baseline: `cae257557e302e843b0d29de1018badd058f1ed0`, branch
`codex/alternating-evolution-refactor`. Engineering implementation and independent
live rehearsal are frozen at `ea99eebb6a049ea8c5050f79981812923987c46b`.
This report is an audit, not an authorization to start a formal experiment.

**Current verdict: NO-GO for a full formal E/V/H launch.** Representative InferAI
DeepSeek V4 Pro requests still fail with upstream 504. The independent E3/V3 live
rehearsal is being measured separately; neither an HTTP 200 nor a completed
generation alone proves diagnosis → mutation → paired screen → gate coverage.

## 1. Evidence and boundaries

| Evidence | Verified | Does not establish |
|---|---|---|
| Engineering suite | 226 tests passed after readiness importer checks; Ruff and diff checks passed | Provider reliability or evolutionary effectiveness |
| Dry validation | E/V/H membership, exclusions, disjointness; E/V only loaded before freeze; disabled formal draft | Any real rollout or provider success |
| Representative Pro requests | Four actual V2 stages per route, real saved E20 evidence, complete structured output checks | A successful diagnosis in either route, or full V2 end-to-end success |
| Live rehearsal | E3 completes without a repair cluster; independent E5 fails closed at Diagnoser 504; H disabled, 1,600 cumulative cap | Required mutation/screen/gate coverage; generalization; engineering cases are not a research sample |
| Research effect | Not yet validated by this audit | Accepted beneficial skill, independent H gain, reproducible method-level improvement |

The native policy, tools, backend, evaluator, task facts, V2 prompts and selection
rules remain intact. No automatic model/endpoint switch, retries, skipped task,
fabricated reward, missing-JSON-brace repair or lowered gate criterion was used.

## 2. Finite panel interpretation and independent evaluation design

`finite_panel_paired` is a preservation/superiority check on frozen observed
task×seed cells. Every task must have the same repeated paired seed coverage.
Current Customer repair must improve; historical/native opponents must preserve.
An ACCEPTED verdict carries `population_risk_certified: false`.

The original V3 and H5 supply three and five task blocks. Four repeated seeds do
not make these twelve and twenty independent tasks. Even under optimistic
independent Bernoulli task-risk assumptions, zero observed regressions imply the
following one-sided 95% upper bounds:

| Task blocks | Zero-event upper bound | Approximate 80%-power detectable paired gain |
|---:|---:|---:|
| 3 | 63.2% | 72.3 percentage points |
| 5 | 45.1% | 56.0 percentage points |
| 16 | 17.1% | 31.3 percentage points |
| 40 | 7.2% | 19.8 percentage points |

The last column is a rough normal planning approximation with two-sided alpha
0.05 and discordance q=0.2; these are assumptions, not measured power. It is
especially imprecise for tiny panels. Repeated seeds reduce within-task noise;
they do not increase the number of task blocks. Task-risk bounds and cell-level
harmfulness have different denominators and must not be interchanged. The
zero-event calculation is already optimistic because not every task necessarily
has an old passing cell. Exact binomial intervals are described in the
[SciPy binomtest documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Unadjusted zero-event 5% risk certification needs at least 59 independent blocks
using a one-sided exact interval, or 73 using the gate's unadjusted two-sided
Wilson bound; multiplicity increases that requirement. There are 74 train tasks,
two exclusions, and E20: at most 52 disjoint train tasks remain for V. Therefore
the pinned Retail data cannot support a credible 5% population-risk certificate
after fixing E20, even if all eligible V tasks show zero breaks.

The proposed **research pilot** is E20 / V16 / H40, G2, four paired gate/endpoint
seeds `[1,2,3,4]`. `configs/v2-formal-readiness-DRAFT.yaml` is disabled and not
approved. V16 is sampled deterministically with seed 20261008 from numerically
sorted allowed train IDs excluding E and 46/47. Selection uses no rewards or
trajectories. H is the entire official test split, using split metadata only at
planning time. Existing E is unchanged. The reproducible metadata-only planner
and its JSON report are adjacent to this report.

Study protocol:

1. E drives search. V is disjoint from E, but adaptive repeated use makes V
   selection data; do not report its final score as an unbiased test effect.
2. Freeze one final ST, its SkillMemory, provider arguments, selected Customer,
   and E-only fresh challenge before unsealing H content. Do not select a model,
   strategy, seed or run after seeing H results. Native and fresh-adaptive H are
   predeclared separately.
3. Primary endpoint: native H task-block paired accuracy difference ST−S0.
   Secondary: fresh-adaptive H difference, conditional on that one E-generated
   fresh challenge. Report all 40 tasks, every paired seed, fixes and breaks,
   max-step terminations, raw accuracy and pass^k. If ST equals S0, reuse identical
   cached conditions and explicitly report no Service update.
4. For exploratory uncertainty, average seed differences within each task, then
   resample whole task blocks with matched S0/ST indices and all of that task's
   seeds. Never bootstrap rollout cells as independent tasks. Use a fixed
   resampling seed and report a 95% paired interval; handle zero-variance data
   explicitly rather than trusting a NaN BCa interval. If making inferential
   claims for both endpoints, predeclare a two-test correction. The
   [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
   explains paired resampling and degenerate interval behavior.
5. A full fixed test split supports a statement about this benchmark and observed
   provider conditions, not a random population of real customers or new domains.
   Public benchmark exposure and previous inspections also limit any claim of a
   pristine unseen set. Local saved episode records contain no test-split runs;
   that is a local artifact observation, not a global exposure certificate.
6. To study the evolution *method*, plan later independent search-seed replicates
   (at least 3–5), native-S0 and relevant ablation comparisons, with a separately
   frozen testing protocol. Those runs are not authorized by this audit. One
   G2 optimization trajectory cannot establish method-level effectiveness.

## 3. Real representative Pro requests

These requests use actual `V2Providers` → `LLMAlternatingEvolvers` → native
`generate()` → LiteLLM → InferAI, not an `{"ok":true}` connectivity surrogate.
No tools, no automatic retries. Saved real E20 Diagnoser context SHA-256:
`2e8f3b0b690f1438b07fca3b3e87b625c319faea686ed31d01dd5cc742957013`.

| Stage | Context characters | Original route | Explicit streaming route |
|---|---:|---|---|
| Diagnoser | 181,114 | 504 / 122.35 s | 504 before response headers / 121.82 s |
| Mutator | 181,708 | Complete legal mutation / 93.99 s | 504 before response headers / 120.20 s |
| Skill Validator | 9,242 | Complete legal schema / 10.20 s | Complete legal schema / 21.02 s |
| Customer Validator | 13,568 | Complete legal schema / 103.55 s | Complete legal schema / 115.43 s |

The mutation probe uses real evidence plus an explicitly declared readiness-only
root-cause fixture; it is not a falsely successful Diagnoser result. The Skill
Validator fixture is an earlier skill, not a promoted candidate. Both Customer
Validator responses reject objective preservation; valid `false` is a legitimate
semantic rejection, not a provider failure. The original route consumed 4 calls
(3 successful, 1 failed), 61,723 reported prompt tokens and 16,486 reported
completion tokens. Streaming consumed 4 calls (2 successful, 2 failed), 5,691
reported prompt tokens and 8,538 reported completion tokens. Failed calls have
unavailable usage; these token totals are lower bounds, not zero-cost failures.

The representative probes preceded the final implementation commit; their plans,
input hashes, actual wire logs and reports are retained unchanged. Their plans
do not contain a runtime-source hash, so they are not claimed to be bitwise-frozen
`ea99eeb` replays. The live E3/E5 rehearsal manifests do record the runtime-source
hash and commit. Subsequent edits are audit helpers/tests/docs only, with the
runtime source hash unchanged. Failed representative reports remain `failed`;
the helper has additionally been corrected to return nonzero for future failed
audits, so a shell/CI job cannot infer readiness from its earlier exit code 0.

Actual Pro wire arguments: model `deepseek-v4-pro`, `thinking.type: enabled`,
`max_tokens: 65536`, no tools; streaming variant additionally `stream: true` and
usage streaming. Although configuration requests `reasoning_effort: high`, the
installed client does not put that field on the Pro wire. High effort is therefore
not independently verified. No GPT call or fallback was made in this audit.

Inference from the recorded pre-header timing: streaming alone does not resolve
the roughly 120-second gateway failure for these representative requests. The
provider's queue/model/edge root cause is not established. A longer local timeout
cannot recover an HTTP 504 response already issued by the gateway. A reliable
route/model change needs its own approved config, credentials and provenance,
then the same representative tests; it must not silently reinterpret old stages.

## 4. Exception, output, recovery and budget controls

- Native benchmark reward 0 and native max-steps termination are scored and the
  panel continues. Unknown/incomplete outcomes are never imputed as failures.
  Provider, JSON and runtime exceptions stop promotion/evolution and preserve the
  attempt. A well-formed semantically invalid candidate is a recorded rejection.
- Validator flags must be JSON booleans, with required fields/reason; `"true"`,
  `1` and missing fields fail closed. Non-stream structured results require an
  actual stop finish reason. Stream results require a completed visible answer,
  one terminal `stop`, no tools, and full stream consumption. Interrupted or
  length-limited JSON is not repaired or scored.
- Raw visible Evolver output, SHA, and parse transformation are persisted. The
  existing complete Markdown-fence removal is now explicit in the artifact
  (`complete_markdown_fence_only`); semantic/bracket repair remains false.
  Missing usage remains unavailable, not a client-estimated token count.
- Shared RequestBudget reserves under a lock and writes pending usage before
  network dispatch. Parallel workers cannot exceed the cap. Evolution, V, fresh
  challenge and H share the same cumulative ledger. The launch wrapper requires
  an explicitly matching `--approved-request-cap`; connectivity probes are off
  by default. Enabling optional probes adds a separate capped two-call check and
  must be disclosed.
- Completed episodes are keyed by the actual condition and hash-verified. Stage
  journal inputs/outputs and generation/checkpoint state are frozen. A provider
  error leaves completed sibling episodes and earlier stages reusable; changed
  model, task, source hash or semantics cannot silently bind an old checkpoint.
- Graceful pause uses a stop-before-next-episode file; active calls can finish.
  On hard kill, pending calls may have been billed: the restored ledger refuses
  unresolved in-flight/reserved entries. Explicit reconciliation is required;
  **there is no automatic hard-crash reconciliation tool yet**. This preserves
  accounting integrity but is a remaining operational recovery limitation.
- Non-stream and stream engineering paths are tested, including a mocked HTTP
  SSE call through actual `generate()`; mocked HTTP is engineering evidence,
  distinctly separate from the real failed representative tests above.

## 5. Draft budget and runbook

The pilot draft proposes **150,000 cumulative provider requests**, not a dollar
cap. It is **not confirmed** unless the user explicitly accepts it; the draft
remains `real_provider_enabled: false` and `launch_authorized: false`.
No monetary estimate is claimed without verified provider pricing.

For planning, at most four Service candidates per generation (three mutations
plus one crossover), up to four opponent conditions, and four gate seeds produce
a deliberately conservative allocation of 4,720 episodes: 40 Customer fitness,
320 screen, 400 full-E repair and 1,280 V opponent episodes per generation, plus
640 final H episodes over G2. This overcounts cached old/screen conditions and
opponents impossible in the earliest generation. Semantic/screen/repair rejection
and identical S0/ST typically reduce the count substantially. At an assumed
30 calls per episode this is 141,600 rollout calls, plus Evolver/validators;
this is a scenario, not a prediction or guaranteed completion under the cap.
`max_steps=32` is not an API-call cap: Activator/evaluator add calls. The request
cap must stop honestly if exhausted. Do not reset budget on resume.

### Authorized small rehearsal

```sh
cd /Users/spring/RSI/Evotau-alternating
export PYTHONPATH=src
export TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074
export HTTPS_PROXY=http://127.0.0.1:65533
export HTTP_PROXY=http://127.0.0.1:65533
export NO_PROXY=localhost,127.0.0.1
/Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-readiness-pro-rehearsal.yaml \
  --tau2-data-dir "$TAU2_DATA_DIR" \
  --approved-request-cap 1600 \
  --stop-before-next-episode-file /tmp/evotau-readiness-stop-next
```

Pause with `touch /tmp/evotau-readiness-stop-next`, wait for terminal paused state
and ledger `in_flight: 0`, then resume by removing that file and rerunning the
identical command from the frozen runtime checkout. Do not kill the process merely
because a call has not produced output yet. Do not repeat already committed
conditions. Provider failures require an explicit reviewed resume, not a retry
loop; no resume was automatically triggered by this audit.
The pause file blocks new **episode** dispatch, not an already running provider
call or every intermediate pre-rollout Evolver stage. It is a graceful boundary
pause, not an immediate cancellation mechanism.

### Formal preparation only — no launch command is authorized

```sh
/Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/plan-v2-readiness.py \
  --config configs/v2-formal-readiness-DRAFT.yaml \
  --split-metadata /Users/spring/.cache/evotau/tau2-data-b7ea9074/tau2/domains/retail/split_tasks.json
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-formal-readiness-DRAFT.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 --dry-run
```

Only after representative reliability is resolved, live production-path gate
coverage is demonstrated, budget is confirmed and formal launch is explicitly
authorized should a separately named enabled formal config be frozen. Its launch
uses the same wrapper, a confirmed matching cap and a dedicated pause file.
This audit does not provide an enabled final config while a blocker remains.

## 6. Live rehearsal outcome

**Required full-flow rehearsal did not pass. No formal E/V/H experiment started.**

| Run | Result | Actual coverage |
|---|---|---|
| E3: 21,16,66; V3 configured; H off | Completed G1; 3/3 native success; Customer semantically rejected; Pro diagnosis has no clusters; Service unchanged | Native rollout/backend/evaluator, Customer generation/validation/selection, diagnosis, checkpoint. No mutation, screen or gate |
| Independent E5: 98,8,21,29,66; V3 configured; H off | Native 4/5; Customer semantically rejected; Diagnoser HTTP504 after 120.14 s; G1 uncommitted | Benchmark failure continues, provider failure stops, completed evidence retained. No mutation, screen or gate |

E5 uses prior observed failure cases and passing controls for engineering coverage,
not a random fitness sample. Two already-completed baseline conditions (21 and
66, seed1) are imported with byte-identical native simulation/score/telemetry;
only activation manifest bindings are transformed, with source/target hashes
recorded. No Customer/diagnosis/search stage is imported across the changed E
panel. All 59 earlier E3 requests are charged to the E5 cumulative ledger,
including the E16 condition not in E5. Import itself issued zero requests.

The E5 Diagnoser input contains 158,888 context characters. The failure has no
diagnosis result, no mutation candidate, no screen/gate decision and no generation
checkpoint; these are missing coverage, not zero improvements. Four earlier stage
artifacts and all five native condition records remain intact. Source and cache
bindings, cumulative budget restoration and stage hashes were checked after the
failure with zero provider requests; all five cached conditions load, with
`in_flight=0`, `reserved=0`. The failed call was not retried.

E3 exact completed-run replay also issued zero requests: all 60 immutable episode
and journal files remained byte-identical, and cumulative usage stayed at 59 calls.
This is stronger than merely asserting that a checkpoint exists, but it does not
prove a hard-killed pending request can be automatically reconciled.

Actual consumption, without double-counting imported E3 usage:

- E3: 59 calls; 278.16 seconds initial execution. Completed replay: zero additional
  calls, about 0.11 seconds in `run_from_config`.
- E5: 120 cumulative calls including those 59; **61 new requests**, 427.82 seconds
  execution. 119 successes, one provider failure; five cached native conditions,
  three newly executed native episodes.
- Representative original/stream probes: eight additional requests, five successes,
  three failures; no native episodes.
- Whole live audit: **128 requests**, 124 successes, four failures; **784,330
  reported prompt tokens**, **66,684 reported completion tokens**. Usage is
  unavailable on four failed requests. Six unique completed native episodes were
  executed; copied records are not counted as additional episodes.
- Reviewer, Customer Judge, Service Judge calls: **0**. Native Qwen calls: 114;
  Pro calls: 14. Native wire thinking-disabled checks passed. Pro thinking-enabled
  calls retain visible structured output when available; high effort remains
  unverified on the actual Pro wire.
- Active measured rehearsal time is about 706 seconds; representative probe call
  time sums to about 709 seconds. Neither is a forecast for a full formal run.
  Token/request limits do not establish a monetary cost: installed price mappings
  for these provider IDs are unavailable, and any native cost-zero placeholder
  must not be interpreted as free inference.

Sanitized immutable evidence, JSON summary and SHA-256 index are in
`experiments/results/v2-readiness-audit-20261008/`.

Remaining launch blockers: representative long Pro diagnosis reliability,
production-path live mutation → screen → gate coverage, and explicit formal
budget/launch authorization. The E20/V16/H40 configuration is a **disabled draft**,
not a release-approved final run config. No model or endpoint fallback was taken.

## 7. Explicitly authorized V4.1 Flash follow-up

After the Pro audit, the user explicitly requested `deepseek-v4.1-flash` with
thinking enabled as an alternative Evolver. The new independent configuration
keeps native Qwen roles and every V2 prompt/task/seed/gate unchanged. Short
connectivity succeeded (HTTP200, legal JSON, 41 reported reasoning tokens), but
the exact same four representative contexts each returned HTTP502. Diagnoser
took 29.81 seconds; Mutator and both Validators failed in under one second.
Length alone is not established as the cause. No native episode or full formal
experiment was started; the alternative config is disabled after the failed
preflight. Five additional inference requests belong to this separate follow-up,
not the 128-request audit total above. Raw sanitized evidence and interpretation
are in [the V4.1 Flash archive](../../experiments/results/v2-readiness-v41flash-thinking-20261008/README.md).
