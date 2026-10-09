# Airline / V-primary protocol (2026-10-09)

No live API request was made during implementation. This is a new algorithm and domain,
not a continuation or a claimed improvement over Retail. Historical Retail artifacts stay
untouched; replay their frozen code at `3469fa5` (or the recorded earlier commit), never
bind the new sources to an old manifest.

## Frozen data and domain

The pinned upstream revision remains `b7ea9074c1cba482b30687fecdb5c8425fd6f619`.
Its Airline split has **30 train / 20 test**, so disjoint E20/V20 is impossible.
The template prioritizes V20, uses E10 and all H20. Selection is `random.Random(20261009)`
over numerically sorted train IDs, E first10, V remaining20. No rewards or trajectories
were consulted. Retail exclusions46/47 do not apply to different Airline tasks.

- E: `1,0,39,49,11,27,34,12,7,15`
- V: `41,20,17,40,23,33,4,10,47,28,36,46,3,14,5,21,43,9,42,38`
- H: `2,6,8,13,16,18,19,22,24,25,26,29,30,31,32,35,37,44,45,48`

`manifest.domain` chooses task files, native environment and policy. Unsupported domains
fail explicitly. Airline fingerprints additionally cover its environment/data_model/utils;
no fallback, tools/backend/task/evaluator edits. The selected-record streaming loader
lexically skips H bodies during evolution; source hashing reads bytes but does not decode
H scenarios. Native DB loading remains upstream behavior.

Airline reservation/flights/fares/passenger/payment/baggage write results remain exact.
The existing Retail order projection never applies to booking results. Decisive evidence
that exceeds the frozen allowance raises an error, rather than being silently truncated.
Customer preservation prompt explicitly includes reservations/flights/fares/payment constraints.
Service input contains E outcomes and observable evidence, not scenarios or reference answers.
V raw records, task goals and rejection details never enter Mutator/crossover prompts.
Archive parents contain E effects only; deployed memory identity is necessarily observable.

## Exact promotion differences

| Stage | Historical `direct_skill_evolution_v1` | New `direct_skill_v_validation_v2` |
|---|---|---|
| Structure/semantic validation | required | unchanged |
| E screen | target fix, zero passing-cell regressions, no stuck increase | target fix, observed protected-cell regressions <=max(1cell,20% of passing cells), stuck delta <=10%, no recorded hard regression |
| Full E | paired superiority with preservation prerequisite | paired attribution only; no E statistical veto |
| Current Customer V | preservation | superiority with lower gain bound >2 percentage points |
| Archived/native Customer V | preservation | preservation |
| Risk | legacy method, including strict finite-cell option | task-block bootstrap for gain; explicit observed-V harm <=15%, stuck delta <=5%, absolute stuck <=20%, zero recorded severe violations |
| Deployment | all required gates ACCEPTED | same conjunction; INCONCLUSIVE retains research candidate, never deploys |

**Numbers above are provisional, not calibrated or authorized launch values.**
`calibration_confirmed:false` blocks formal evolution before any provider call.
They must be calibrated against A/A evidence, reviewed and frozen in a new independent
config/manifest before formal launch. The implementation does not lower thresholds until
something passes.

Both V and H use paired seeds `[1,2,3,4]`; each task remains one bootstrap block.
Current V also requires positive net gain in at least50% of seed schedules. Current
superiority requires a positive corrected task-block lower bound above the configured
minimum; one extra observed pass is insufficient. Historical/native preservation has zero
accuracy margin. Gross lower mean accuracy or recorded hard regression is REJECTED;
insufficient tasks/repeats/confidence is INCONCLUSIVE. Frozen Bonferroni looks include the
maximum candidates (3direct+1conditional crossover), G2 and possible4opponents:32looks.
Actual V decisions are counted in generation `validation_looks_evaluated`.

The Wilson task-risk bound is retained and reported, but observed-panel risk acceptance
**does not certify that bound**. With V20 and32looks, a5% population bound is not credible
even with zero observed regressions. `population_risk_certified:false` and
`adaptive_validation_panel_task_block_evidence` make that distinction explicit.
V is reused adaptively, not an untouched generalization set. H never selects candidates.
H native/fresh S0/ST scorecards report Pass1/Pass2/Pass4, full metrics, endpoint flips and
identical-S0 reuse. A legally rejected fresh Customer stays unavailable; it is not relabeled.

Candidate retained != promoted != deployed. Original unknown/error fail-closed behavior,
finite cumulative budget, immutable journals and manifest/source checks remain active.
Ordinary native failures are scores; network/JSON/runtime errors are not fabricated failures.
Observed hard-policy/protocol counts reject regressions. These counts are not an exhaustive
Airline policy-violation oracle: native tools enforce constraints and native evaluator scores
its task criteria; no new unrestricted policy Judge has been added. This limits any claim
about unobserved severe business violations and requires trajectory audit during calibration.

## Independent baseline / A-A

`python -m evotau.aa_diagnostic` or `experiments/execution/run-airline-baseline.py`.
The wrapper reads the existing InferAI key internally, with existing pacing and sanitized
wire metadata. No Evolver key/request is needed. No auto retries or model substitution.
Without `--execute` it only validates and prints the episode plan.

Each repetition has a different experiment ID/manifest/output/cache namespace, while keeping
identical empty Service memory, models, task, seed, customer strategy, policy and max_steps.
A new repetition does not hit baseline caches. Resume reuses completed episodes **within**
the same repetition; it does not rerun them merely to call the experiment independent.
One cumulative finite budget covers every repetition, including failed calls.
A/A output binds tasks/seeds/repetitions/config; changed conditions cannot resume.
OS writer lock prevents duplicate dispatch. `--stop-before-next-episode-file` pauses before
the next episode; remove the file and repeat the same command to resume. In-flight calls
are preserved by existing accounting and require explicit reconciliation if forcibly killed.

Output:
- `diagnostic-config.json`, `api-usage-live.json`, `aa-report.json`, failure evidence if any;
- replicate manifests, full native-simulation.json, provider-calls.jsonl, sanitized HTTP log,
  activation traces, prompt hashes and native reward_info;
- paired success flips, first Customer equality, first differing native message index (plus behavioral comparison ignoring generated tool-call IDs),
  termination reasons, reward_info and skill-activation IDs.

This baseline tool starts with empty memory (no Activator dispatch). It quantifies ordinary
run variation; it does not experimentally isolate Activator-only overhead. Existing candidate
activation traces can identify selected guidance and native-prompt hashes, but ordinary paired
rollouts cannot prove causal Skill benefit. No fixed-Customer causal replay is claimed.
Selected diagnostic subsets cannot estimate all-benchmark noise rates.

## Next recommended paid action — not executed

One Airline E baseline:40episodes (10tasks×4seeds), cap20,000provider requests. This is a
request cap, not a dollar quote or a promise all episodes will finish. Longer max_steps200
is a newly frozen Airline condition, not evidence of improvement over Retail32.

```sh
cd /Users/spring/RSI/Evotau-alternating
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python experiments/execution/run-airline-baseline.py \
  --config configs/airline-baseline-e10.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  --output experiments/diagnostics/airline-baseline-e10-20261009 \
  --repetitions 1 --seeds 1 2 3 4 --approved-request-cap 20000 --execute
```

Use a new independent directory and `--repetitions 2` for A/A (80episodes). Select/approve
an adequate finite cap in its own config first. Do not reuse a completed40episode directory
with changed repetitions. For calibration, compare both runs independently, not baseline
cache versus itself.

Formal template: `configs/airline-direct-skill-v-primary-e10-v20-h20.yaml`.
Proposed request cap200,000 is **unconfirmed**, formal launch blocked. Conservative no-cache
maximum: G2×4candidates×(40screen+80fullE+640Vreplay) +40Customer +20selectedE +320H
= **6,460 episode executions**. Baseline/opponent cache reuse normally reduces this heavily;
NO_OP/rejections reduce it further. Conditional crossover need not occur. A native200-step
episode can approach200 runtime calls; candidate Activator adds up to roughly100 calls,
plus native evaluation and search/validators. Thus the no-cache worst case exceeds1.9million
requests; cap200,000 does not guarantee completion. Measure Airline baseline before adopting
an expected call/token range or monetary budget. Retail's archived94,523tokens per baseline
cell is not an Airline cost estimate. Provider token/cost uncertainty and unknown billed usage
remain visible, never treated as zero.

The eventual formal command uses the existing official-DeepSeek Evolver wrapper and a newly
calibrated/approved config, with `--approved-request-cap` matching its frozen cap. The current
uncalibrated template must not be used for live evolution.

## Verification / limitations

Tests cover native Airline environment/tools/policy/task IDs, E/V/H isolation, no Retail fallback,
V accepted/rejected/inconclusive paths, finite regressions/hard/stuck rejection, corrected seed
coverage, G2 journal resume without model redispatch, native Airline evaluator/cache recovery,
independent A/A cache namespaces and frozen-condition refusal. All provider boundaries in these
tests are deterministic/local. Retail suite remains included. Live Airline quality, live A/A noise,
provider availability and real cost have not been measured.

Actual offline validation on2026-10-09:

```sh
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python -m pytest -q
# 304 passed, 6 dependency deprecation warnings, 32.65s
/Users/spring/RSI/Evotau/.venv/bin/ruff check src tests experiments/execution/run-airline-baseline.py
# All checks passed
git diff --check
# passed
```

Baseline planning dry-run printed40episodes and `real_requests_started:false`.
Formal template dry-run loaded30train IDs (E/V), zero H objects and no provider requests.
New G2 deterministic simulation reached V gates, selection/archive/commit and resumed without
repeated model generation. Native Airline baseline/A-A tests executed two independent local
completion-boundary episodes and reused both on resume. No live Airline rollout/evolution,
no live A/A calibration and no paid connectivity probe was performed.
