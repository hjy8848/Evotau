# Airline Failure Analyst → assigned Skill Mutator

Algorithm: `analyst_skill_v_validation_v3`, derived from `6f43f04`.
The existing Airline run remains on its original checkout/source; this change was developed
in an isolated worktree. No live API calls or new native episodes were started for this change.

## Mechanism

Old: frozen E overviews/evidence → three independent bias-based Direct Mutators → Validator,
Screen, Full-E, V-primary opponents/gates, Archive.

New: the same frozen E evidence → Failure Analyst → mechanism diversity review → immutable
hypothesis assignment → one Mutator per assigned hypothesis → the same evaluation pipeline.
The three old biases are not renamed or applied to the new mode. Fewer hypotheses (including zero)
are valid. There are at most three ordinary proposals; the existing optional crossover remains a
separate parent-merge proposal (maximum one). Crossover uses the old schema instruction, not the
single-hypothesis restriction; it is skipped when no hypothesis was assigned.

Analyst returns `hypotheses` and `insufficient_evidence_reason`. Each hypothesis has exactly:
`mechanism_id`, `target_task_ids`, `protected_success_task_ids`, `observed_deviation`,
`root_cause_hypothesis`, `evidence_refs`, `expected_behavior_change`, `alternative_explanations`,
`regression_risk`, `repairability` (`skill`, `not_skill`, `uncertain`). No Skill payload is permitted.
Targets/control labels and every five-field citation are checked against retained E task×seed
observations, not trusted because a model claimed them. Scores are never rewritten.

For multiple hypotheses, one extra Evolver call compares every unordered mechanism pair:
`same_mechanism=true/false/null` plus a reason. Identical normalized operational corrections are
also deterministically deduplicated. Semantic equivalence remains a model judgment, not a proven
fact. An uncertain pair is recorded and not allocated twice. Eligible hypotheses are ordered by
repairability, retained reference count, and stable input order; `not_skill` is recorded but not
mutated. This is a conservative allocation heuristic, not a validated diagnosis-quality metric.

Every non-NO_OP mutation must keep its assigned mechanism ID, exact root-cause hypothesis, full
target set, allowed success controls, and assigned evidence references. It can recheck the claim
and return NO_OP. It cannot silently invent a different cause. Skill schema is unchanged.
Public Airline tool names and policy rules are reusable within a shared operational condition;
fixture records, hidden goals and gold answers are still prohibited. Semantic validation remains
an LLM judgment plus existing structural checks, not a deterministic guarantee of semantic safety.

Each generation freezes `service_failure_analysis`, optional `service_mechanism_diversity`, and
`service_hypothesis_assignment` before existing `service_proposals-gXXXX-direct-N` stages.
Generation artifacts expose hypothesis → candidate → mutation → screen/gate lineage. The existing
call journal records raw inputs/outputs and schema errors. Parsed invalid Analyst/diversity output
is recorded as rejection with no blind fallback. Network, JSON parsing and native execution errors
still fail closed; successful calls/stages are reused on exact frozen resume.

PRISM source inspected directly:
[proposer.py](https://github.com/airbnb/agent-harness-optimizer/blob/main/agent_harness_optimizer/optimizers/prism/proposer.py),
[loop.py](https://github.com/airbnb/agent-harness-optimizer/blob/main/agent_harness_optimizer/optimizers/prism/loop.py).
We borrow separate analysis/mutation and action-sequence reasoning with passing controls.
We do not copy middleware, forced exact-N analysis, heuristic fallback or model-output coercion.

## Frozen boundaries and cost

New config: `configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml`.
E10/V20/H20 IDs, single-seed diagnosis, native models, official DeepSeek Evolver arguments,
G2, concurrency2, max_steps200, evidence settings, activator, Screen thresholds, V-primary
bootstrap/Bonferroni rules, replay and H evaluation are unchanged. No new request/token/context
cap was imposed; null settings preserve the operator's preference. A/A remains uncalibrated,
not falsely marked confirmed. No existing scores/checkpoints are implicitly imported.

Each generation adds one Analyst request and, for 2–3 hypotheses, one semantic diversity request.
There are 0–3 assigned Mutator requests instead of three forced independent proposals. Crossover
and evaluation costs remain conditional as before. Extra reasoning might improve diversity but
can also incorrectly merge distinct mechanisms or reject good hypotheses. No live evidence yet
shows improved Airline Skill quality, promotion probability or total cost.

## Offline comparison prepared from actual Gen0 evidence

`experiments/replay-inputs/airline-gen0-e10-service-context.json` contains the original frozen
Gen0 proposal input with its source path/hash, model/request arguments and E-context digest.
All ten task results remain visible; only the original representative raw evidence is expanded.
It contains no V/H trajectories and no hidden scenario/gold fields. No native scores are imported
into a new formal experiment by this replay.

From the isolated repository checkout, prepare both variants without requests:

```sh
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/replay-airline-skill-generation.py \
  --source experiments/replay-inputs/airline-gen0-e10-service-context.json \
  --output experiments/diagnostics/airline-analyst-prompt-comparison
```

Only after separate authorization, repeat the same command with `--execute` and the official
DeepSeek key temporarily available through `OPENAI_API_KEY`. It performs three old Mutator calls,
one Analyst, zero/one diversity call and zero-to-three assigned Mutators, with no native rollout,
Validator, E/V gate or H access. No automatic retry or model switch. The immutable plan freezes
script/prompt hashes, evidence and model arguments; journal and API usage are resumable in that
same directory. A changed plan needs a new directory, not editing the old plan.

Outputs: `replay-plan.json`, `prepared-inputs.json`, `evolution-v2/*.json`, provider inputs/outputs,
`api-usage-live.json`, and `replay-results.json`. Compare supported distinct mechanisms, concrete
interventions, schema/evidence validity and repeated rejected corrections. Better-looking prose
is not evidence of efficacy. Native screening of generated candidates requires a separately
authorized experiment/compatibility audit; this replay does not evaluate their effects.

Offline config validation (no `--execute`):

```sh
PYTHONPATH=src /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-airline-gateway.py --mode formal \
  --config configs/airline-failure-analyst-qwen37plus-official-dsflash-p2.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

## Validation

Executed with the existing environment and pinned native data:

```sh
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python -m pytest -q
/Users/spring/RSI/Evotau/.venv/bin/ruff check src tests \
  experiments/execution/replay-airline-skill-generation.py
git diff --check
```

331 tests passed, six dependency deprecation warnings. New tests cover diversity, zero/one/three
allocation, bad IDs/seeds/indices/hash/labels, binding, NO_OP, E-only inputs, rejected-candidate
memory isolation, unchanged Screen/Full-E/V, offline provider/journal G2 with budget accounting,
interrupted-stage resume, exact completed-call reuse, replay preparation/resume, old prompt bytes
and separate manifest identities. The legacy morphology test requires an existing local historical
artifact; exact bytes were copied from the original checkout into this worktree's ignored run
directory. No historical source/artifact was rewritten.

Repository-wide `ruff check .` was also run: historical archived scripts/runbooks have pre-existing
lint violations outside src/tests. They are preserved for reproducibility; active code, all tests
and the new replay script pass Ruff. Online model behavior and new Skill quality remain unverified.

Most useful review points:
- `evolution_candidates.py`: versioned Analyst/Mutator/Validator prompts and provider dispatch.
- `failure_analysis.py`: evidence contracts, assignment binding, semantic uncertainty/selection.
- `skill_evolution.py`: frozen Analyst/assignment stages; unchanged evaluation afterward.
