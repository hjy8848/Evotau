# EvoTau V2 E20/V3/H5/G2 Launch Readiness

2026-10-08 · baseline `bbf939853a50fa2e0abd0a77b87a14ac0e4b7017` · branch `codex/alternating-evolution-refactor`

**Decision: engineering release PASS; budget approval pending. No new provider request, E3 rehearsal, or formal experiment was launched during this review. After explicit approval of the 100,000-request ceiling, use the frozen command below to start the full experiment directly. No additional live canary is required. Zero Skill promotions is a valid outcome.**

## Frozen release

- Runtime/code commit: `83b13c6ee664f747f5a8f436516d185338633a91`.
- Config: `configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1.yaml`.
- Frozen manifest: `research/audits/v2-release-20261008/frozen-manifest.json`.
- Manifest SHA-256: `64628f0788f36ce0ee5f004bba46433e94d657e2e8c13861ff879d01ef8d5c93`.
- Runtime source SHA-256: `c894614d9134f71a0d3709a3cb1e2338f8c2f4870ddfa5551eb8f9583a914568`.
- Config file SHA-256: `991cac98907d3ee1a9646dcc77bca33f83712be5b2d50b00c437121b56e2740c`.
- Config payload SHA-256: `b8a5bb52dd3d9768614fabe4371a4c301f9d1dd836a4fd48d3b108463a41df3b`.
- The frozen command binds this manifest before inference and verifies source/config digests. Launcher, transport observer/pacer, and recovery CLI hashes are also embedded in the config and checked by the launcher. Subsequent report-only commits do not change the frozen runtime.
- Repository-wide clean flag is false because unrelated `experiments/.DS_Store` was left untouched; it is not a runtime source. Runtime files are committed and independently hashed.
- Independent run: `experiments/runs/evotau-v2-qwen37plus-gpt61sol-retail-e20-v3-h5-g2-p1-release-20261008`.
- Checkpoint: `experiments/checkpoints/evotau-v2-qwen37plus-gpt61sol-retail-e20-v3-h5-g2-p1-release-20261008.json`.
- No old episode score, proposal, generation or mixed-model checkpoint is imported.

| Setting | Frozen value |
|---|---|
| E | 66, 92, 29, 67, 106, 22, 69, 98, 93, 88, 4, 21, 8, 54, 107, 48, 52, 80, 35, 16 |
| V | 44, 25, 78 |
| H | 74, 55, 18, 12, 97 |
| Excluded | 46, 47 |
| Runtime Agent / Customer / Evaluator / Activator | `openai/qwen3.7-plus` |
| Evolver, including semantic validators | `openai/gpt-6.1-sol` |
| GPT args | `api_base=https://inferaiapi.com/v1`, `api_protocol=responses`, `api_key_env=INFERAI_API_KEY`, `reasoning_effort=high`; no temperature/output token cap |
| Qwen args | Existing chat adapter, `temperature=0.0`, `thinking_mode=disabled`; wire translation `thinking.type=disabled` |
| Generations / Customer candidates | 2 / 1 per generation |
| Episode concurrency / max_steps | 1 / 32 |
| Fitness seed | 1, fixed across Customer/Service/generations |
| Screen / Gate seeds | [1, 2] / [1, 2, 3, 4] |
| Service proposals | 3 per cluster; maximum1 cluster; optional1 crossover: upper4 per generation, no forced proposals |
| Replay | Up to2 historical semantic-valid Customers plus native Customer |
| Gate | Existing `finite_panel_paired`, zero paired success/stuck regressions, no relaxed correctness |
| V/H | Enabled; H objects/trajectories remain sealed until evolution and the fresh decision complete |
| Native benchmark | τ-bench `b7ea9074c1cba482b30687fecdb5c8425fd6f619`, package1.0.1; policy/tools/backend/evaluator/metrics unchanged |
| Request cap | **100,000 total attempted dispatches**, including failed/recovery requests; pending human approval |
| Transport pacing | Existing shared Chinese-group30 requests/60.1s; no automatic retries/model fallback |

## Confirmed blockers fixed

1. **Parsed but schema-invalid Evolver cache poisoned resume.** Stage contracts now record immutable `schema-error.json` with input/output digests and raise `EvolverSchemaError` with a diagnostics reference before journal publication. Ordinary resume revalidates and stops on the same invalid response with zero additional calls. An explicit operator authorization can exclude that exact invalid response from reuse and permit a replacement with the unchanged prompt, context and model. Original visible text, parsed JSON, fees and error remain. A new schema error requires its own authorization. Valid semantic rejection is never treated as schema failure.
2. **Diagnoser outcome-label error could be published as a valid stage.** Native failed/passing case labels are checked at the provider boundary before the diagnosis is frozen. Nested mutation payload/operation types, draft activation signatures and safe stage identifiers are strictly checked; illegal schema is not coerced. A structurally legal but inapplicable/budget-exceeding mutation remains an explicit pre-rollout research rejection.
3. **Gen1/stagnation could crash on `gate=None`.** Rejected Screen candidates have nullable Gate; history handles it safely. No selection/acceptance rule changed.
4. **Archive parent/crossover exposed raw V gate episodes to Evolvers.** Parent inputs now contain E-derived mutation/effect evidence only. Raw V gate records and metrics stay in reporting/archive artifacts, outside Evolver prompts. Promotion acceptance signals are necessarily adaptive V feedback; this is disclosed below. Customer hidden scenarios and Customer semantic-validator free text are not added to Service inputs.
5. **Fresh semantic rejection permanently prevented finalization.** Freeze one E-only fresh proposal and its semantic validation. Use it only if all strict flags pass and its text differs from evolution incumbent/initial/archive strategies. On rejection or duplicate text, mark fresh-adaptive unavailable and run native H only; never substitute an archived Customer or claim a fresh score. Replay reuses the same rejected decision rather than regenerating until it passes. Provider/schema errors still stop. This is a frozen missing-endpoint protocol, not semantic relaxation.
6. **Native evaluation publication crash could strand paid completed episodes.** New immutable native completion bundles preserve the exact EpisodeRecord, usage telemetry and activation trace. Cache restoration republishes missing artifacts without calling the provider or evaluator. A task-only completion key prevents reading H activation/trajectory bundles during E/V restoration. Generation publication is likewise reconstructed from immutable generation commits.
7. **Concurrent launch/resume could race on accounting/artifacts.** An OS writer lock on the frozen checkpoint rejects a second writer; recovery operations take the same lock. Pause is checked before new Evolver dispatch as well as native episode dispatch.
8. **Report falsely claimed V evaluation when only E Gate ran.** `validation_evaluated` now requires actual V opponent decisions. Finite-panel promotion is labeled `finite_panel_validation_gated`; E decision time is excluded from V timing. Numerical gate conditions are unchanged.
9. **Old formal config was mixed-model/unbounded/P4/E-only.** New serial pure-Qwen/GPT E20/V3/H5/G2 config has a finite cap, independent paths and pinned provenance. Formal launcher rejects its optional separate connectivity probes, keeping all experiment dispatches inside the run budget.

## Stage-by-stage control-flow review

| Stage | Input/output contract and legal branch | Failure/cache/recovery |
|---|---|---|
| Customer | Full E task scenarios/visible trajectories/current strategies → one structured candidate | JSON/schema/network failure stops. Completed recorded call recovered if journal publication was interrupted. |
| Customer Semantic Validator | Candidate + underlying E scenarios → strict boolean flags/reason | False flag: candidate accuracy=null, no rollout, incumbent remains eligible. Invalid flag type: schema error, not task failure. |
| Customer selection | Complete incumbent/candidate E seed1 records | Strict lower accuracy wins; tie preserves incumbent. Unknown outcome never enters accuracy. |
| Diagnoser | E visible trajectories/tools/native outcomes/policy/memory/E mutation history; no hidden scenario or raw V/H | Strict surfaces/types/observed case IDs/native labels. Non-skill clusters are valid; no skill fabrication. |
| Mutator / crossover | One cluster/E controls/accepted memory/local parent effects → strict structural mutation | Schema failure stops; NO_OP, semantic duplicate, absent target or applicability violation is recorded rejection with no fitness imputation. IDs/high watermark recover deterministically. |
| Skill Semantic Validator | Proposed reusable memory + native policy → strict flags | False flag rejects before evaluation. JSON/schema/network errors stop. |
| Screen | Paired target/protected/prior-fix/clean E tasks × [1,2] | Every baseline-passing cell is protected, including another seed of a target task. Require target fix, zero broken cells/hard regression, non-increased stuck count. Ordinary native failure scores normally. |
| E Gate | Full fixed E × [1,2,3,4], same current Customer and old/proposed Service | Require observed strict superiority with paired preservation. If rejected/inconclusive, no V promotion check; record honest status. Screen/full-E overlapping conditions reuse native cache. |
| V Gate / opponent replay | Independent task IDs V3 × [1,2,3,4], same cells per old/new/current/historical/native condition | Existing preservation objective; zero passing→failing/new-stuck cells, no hard regression. All required Gates must ACCEPT. Raw V evidence never becomes repair examples. |
| Archive | Candidate ID/effect/E scope/comparison key/decision/lineage | Screen/full-E fitness scales remain distinguished; archived candidate is not deployed. Promotion only via complete gates; rejected local fixes may inform E-derived history. |
| Generation commit | Selected Customer/Service, exact final E records, history/archive/watermark/provenance | Immutable digest-bound generation commit; atomic checkpoint. Missing publication/checkpoint recovered from commit. Gen1 rebuilds completed prior generation without calls. |
| Fresh Customer | Completed E evidence and final Service, before H loading | One fresh attempt with frozen strict semantic/distinct-text rule. Rejection→fresh unavailable; no regeneration or non-fresh fallback. Schema/network failure→stop/resume protocol. |
| H | Native and, if available, frozen fresh Customer × S0/ST ×5 tasks ×4 seeds | Neither scores nor H content reach Evolvers. Same manifest/task/seed/strategies/native args form cache key. If S0==ST, reuse endpoints explicitly. Interrupted H resumes completed conditions. |
| Finalization | Generation commits/native endpoint records/cumulative budget/attempt history | Full report includes rejected candidates, unavailable fresh endpoint, zero promotions and actual API roles. No benchmark exception converted to a failing task. No partial run labeled complete. |

**Data isolation:** dry validation loaded23 E/V tasks, not H. IDs/splits/exclusions were verified against pinned split metadata and selected task schema/source blob fingerprints. H files are lexically skipped before unsealing. Validation is distinct from E but is repeatedly used for adaptive promotion; it is not an untouched final test. Existing H5 has been used in earlier project experiments, so it is not a new unseen research sample. This release imports none of those results.

## Statistical limits: what this experiment can say

`finite_panel_paired` resolves V3/min_tasks8 by explicitly checking observed cells, rather than lowering a population sample threshold. `min_tasks=8` remains the bootstrap policy threshold; it does not certify V3. Gates report `inference_scope=frozen_tasks_observed_seeds_only`, `population_risk_certified=false`. Bootstrap/Wilson quantities are descriptive here, not the promotion's population guarantee.

- E20 supports an accuracy/repair curve on the selected training tasks; selection and repeated search make it unsuitable as an unbiased generalization estimate.
- V3 with4 seeds has12 observed cells but only3 task blocks. H5 with4 seeds has20 observed cells but only5 task blocks. Repeated seeds reduce within-task noise, not increase independent task count.
- Even in the optimistic independent-task Bernoulli model, zero harmful tasks among3/5 gives one-sided95% risk upper bounds about63.2%/45.1%; actual at-risk successful tasks may be fewer. This cannot establish a≤5% population regression risk.
- With only5 paired task blocks, a two-sided exact sign test's smallest possible p-value is0.0625 (all5 nonzero differences same direction); for3 it is0.25. Task means/seeds are not new independent sign trials. Small positive H deltas are exploratory, not a powered significance claim.
- Report H native/fresh paired differences, cell outcomes, pass^1/pass^2 and missing endpoints; do not treat20 cells as20 independent new tasks or finite-panel acceptance as overall generalization.
- A later research-confirmation study needs preregistered independent runs/seeds and substantially more held-out task blocks, preferably untouched additional domains/tasks. Broader claims must be scoped to available data; this review does not change this run's fixed panels or invent new tasks.

## Cost and request ceiling

Reproducible planning: `experiments/execution/plan-v2-release.py`; full evidence list and estimates: `research/audits/2026-10-08-v2-release-cost-plan.json`. The sample has144 unique archived real Qwen episodes: mean19.92 calls/episode, p90=24, max27; mean104,863 input and4,406 output tokens/episode. The sample includes empty-catalog/V1 conditions; nonempty V2 memory adds Activator calls. A50% call uplift is a planning scenario, not measured formal performance.

Path allocation accounts for one Customer candidate/generation, up to3 mutations+1 crossover, shared Screen/full-E condition cache, up to3/4 opponents in Gen0/Gen1, four Gate/H seeds and endpoint reuse. It is **not E3 multiplied by20/3**.

| Branch | Episode allocation | Calls from mean through +50% activation scenario | Input-token planning range | Output-token planning range |
|---|---:|---:|---:|---:|
| No skill; Customer rejected; unchanged Gen1 reused; fresh available | 60 | 1,235–1,833 | 8.3M–10.8M | 0.36M–0.74M |
| Valid Customer candidates; no Service promotion | 100 | 2,032–3,028 | 12.5M–16.7M | 0.54M–1.16M |
| One Screen/full-Gate proposal per generation, all replay checks | 648 allocation | 12,946–19,399 | 70.0M–97.6M | 3.0M–7.0M |
| All candidate paths allocated | 1,380 allocation | 27,525–41,268 | 146.7M–205.5M | 6.2M–14.8M |

The largest row is a conservative episode allocation, not a predicted count: Gen0 Customer40 + union Screen/full-E400 + V180; Gen1 40+400+240; H≤80. Rejections/cache reuse reduce it. For the66-call/episode allocation (32 runtime +32 Activator +2 evaluator) plus40 GPT calls, it reaches91,120 requests. This is not a proved native API ceiling: evaluator assertion counts and provider/runtime behavior can vary. **100,000 is the hard global request ceiling**, giving allocation headroom without unlimited requests.

Token columns combine historical per-episode mean/p90 with a2M input/100k output GPT allowance; they are scenarios, not statistical intervals or token caps. Larger skill context/reasoning or long responses can exceed them. Allow roughly10M–300M input /0.4M–20M output tokens for planning, with substantial uncertainty. No current credential-group monetary price is verified, so a currency cost is deliberately not invented. A request ceiling is not a hard currency/token ceiling; actual usage remains recorded.

Serial pacing supplies a lower throughput limit: at least≈2s per Chinese-model dispatch at sustained30/60.1s throughput. At≈4–7s actual latency, branches can take hours to days; GPT reasoning adds time. The archived7m48s E3 result is connectivity evidence, not a formal-runtime prediction.

**Approval required:** explicitly approve100,000 attempted requests, or choose another cap and regenerate the frozen config/manifest before launch. The cap cannot be raised in-place after dispatch. Exhaustion ends that frozen run with preserved progress; any changed-budget continuation requires separate authorization/config/provenance and cannot be presented as this completed frozen run. No output-token truncation or model switch is added.

## Failure, pause and resume protocol

| Condition | Frozen handling |
|---|---|
| Ordinary benchmark task failure/max_steps with native boolean reward | Continue; genuine native failure participates in fitness. |
| Legitimate semantic/structural/NO_OP/Screen/Gate rejection | Continue; record rejection, no fabricated episode/accuracy for unrun candidates. |
| HTTP429/timeout/502/504/empty/incomplete provider output/native execution error | Stop, preserve error chain/stage/partial evidence and charged attempts. Later explicit same-command resume reruns only incomplete conditions; completed stages/episodes are reused. No automatic retry loop. |
| Valid JSON, illegal stage schema/labels | Stop with immutable parsed/raw output and `schema-error.json`. Plain resume still fails without cost; operator-reviewed bound authorization permits unchanged-input replacement. No silent output repair. |
| Budget exhaustion | Stop before next dispatch; same-cap resume cannot buy more requests. Checkpoint/archive remain valid; report remains incomplete. |
| Graceful PAUSE signal | Finish active episode, then stop before next native/Evolver dispatch. Completed cached conditions may be reconstructed without calls. Remove signal and invoke identical command to resume. |
| Ctrl-C | Record pause and potentially incomplete episode; same frozen resume preserves finished conditions. Prefer the PAUSE signal so the in-flight request/episode can finish. |
| Hard kill with unresolved persisted in_flight | Default restore refuses uncertain accounting. After confirming process is dead, explicitly charge possibly billed calls as interrupted/unknown transport attempts, retain before/after accounting audit, leave tokens unknown, and resume. Never count these as benchmark failures. |
| Corrupt/different manifest/source/config/cache/digest | Terminal for this frozen run until original exact evidence/environment is restored. Do not edit output/digest to bypass checks or import changed-condition scores. |
| Second writer | Rejected by checkpoint lock; no concurrent resume on the same artifacts. |

Requests are durably reserved before external dispatch. Partial native episodes without a completion commit may need their unfinished condition rerun; their earlier costs/evidence remain and are not scored twice. Distributed HTTP has no exactly-once guarantee: a lost response may have been billed; the conservative reconciliation preserves that uncertainty. Operator-authorized schema recovery may change which sampled output is used; all original/ replacement attempts are retained and must be disclosed with results. Do not selectively retry valid but disappointing candidates.

Completed provider JSON is reused if stage publication was interrupted; completed native bundles and generation commits recover publication/checkpoint without inference. Budget/episode cache condition keys remain bound to the full manifest/native args/task/seed/Customer/Service. Original tests also cover reservation accounting and stable panel order.

## Actual verification and evidence boundaries

**Engineering executed:** full offline suite **281 passed,5 dependency deprecation warnings,23.93s**; changed-file Ruff passed; `git diff --check` passed. Logs: `research/audits/v2-release-20261008/tests.log`. Ordinary and frozen-manifest dry validation passed, loaded exactly E20/V3, not H; `real_requests_started=false`. Six local G2/V/Fresh/H orchestrator cases cover fresh accepted/rejected × no interruption/Gen1 timeout/H timeout, exact completed-stage reuse and final report. Additional tests cover nullable Gate/history, parent V isolation, explicit schema retry, invalid labels, strict nested schema, paired seed/cell protection, writer exclusion, pause, hard-kill accounting, completion publication, Gen publication and H bundle sealing. Native offline integration executes pinned τ-bench tools/backend/evaluator using deterministic local completions; fault injection includes malformed JSON/schema/timeout/budget/native error. Synthetic acceptance/fitness in local tests is explicitly test data, not live research accuracy.

An initial combined-test invocation exposed test-order dependence from τ-bench import-time DATA_DIR when the environment was not set before imports. The final full command below sets the pinned data directory before process start; the formal launcher also does so. The native package was not changed.

```bash
TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python -m pytest -q --tb=short
```

**Live evidence already available, not rerun:** prior serial E3 completed62/62 HTTP200, zero429,7m48s, actual Qwen thinking-off; GPT Responses reasoning high completed representative Diagnoser/Mutator/Skill Validator requests in earlier archives. Prior raw failures remain documented. The serial episode run had no skill proposal/promotion, and earlier real GPT proposal→Gate replay used synthetic outcomes. This review made **zero new real provider requests**.

**Not yet established:** full live G2+V+Fresh+H reliability, live skill-promoting Gate effectiveness, formal accuracy curves, superiority on H, provider effort-tier authenticity, permanent availability, or population generalization. These are residual experiment risks, not a reason to add another E3 rehearsal. HTTP errors or no promotions remain valid recorded outcomes when handled as above.

## Frozen launch and recovery commands

Run from `/Users/spring/RSI/Evotau-alternating`. The Python runtime below already has the pinned native package. The wrapper reads Chinese-model credentials from Keychain into process-local `OPENAI_API_KEY`; the existing GPT Responses adapter uses the GPT group via `INFERAI_API_KEY`/Keychain. No keys are stored in config/Git.

**Only after explicit budget authorization**, start (or use the identical command to resume):

```bash
PYTHONPATH=src \
TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
HTTPS_PROXY=http://127.0.0.1:65533 \
HTTP_PROXY=http://127.0.0.1:65533 \
NO_PROXY=localhost,127.0.0.1 \
/Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1.yaml \
  --frozen-manifest research/audits/v2-release-20261008/frozen-manifest.json \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --approved-request-cap 100000 \
  --stop-before-next-episode-file experiments/runs/evotau-v2-qwen37plus-gpt61sol-retail-e20-v3-h5-g2-p1-release-20261008/PAUSE
```

Pause after active work completes:

```bash
touch experiments/runs/evotau-v2-qwen37plus-gpt61sol-retail-e20-v3-h5-g2-p1-release-20261008/PAUSE
```

Resume: remove that exact PAUSE file, then invoke the unchanged launch command. No checkpoint edits, no task deletion, no old-result import.

For an illegal cached **schema** response only, after examining the recorded `schema-error.json` and confirming the process has exited:

```bash
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/recover-v2-release.py \
  --config configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1.yaml \
  --authorize-schema-retry CALL_DIRECTORY_ID \
  --reason 'Operator reviewed this exact invalid schema; authorize unchanged-input replacement'
```

This makes no provider call. It writes a digest-bound authorization alongside the original error. Resume with the unchanged launch command; valid rejected candidates have no schema authorization path.

For hard-kill unresolved accounting only, after confirming no active process remains:

```bash
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/recover-v2-release.py \
  --config configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1.yaml \
  --charge-interrupted-requests \
  --reason 'Process confirmed dead; conservatively charge all persisted possibly billed requests'
```

Before/after usage is retained in `recovery/*-accounting.json`; unknown tokens are not guessed. Then resume unchanged. Source/config/fingerprint mismatch is not handled by either operation.

Final outputs will include `alternating-result.json`, every `generation-*.json`, immutable `evolution-v2/` decisions, `evolver-calls/` visible/parsed/failure/recovery evidence, native episode completion/activation/provider logs, checkpoint, `heldout-endpoint-evaluation.json`, cumulative API roles and actual request statistics. Report unavailable fresh scores explicitly; report rejected mutations and zero promotions as observed results.
