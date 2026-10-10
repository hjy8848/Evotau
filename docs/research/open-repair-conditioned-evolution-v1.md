# Repair-Conditioned Open-Ended Evolution v1

## Scope and protocol identities

Baseline audited: `6fc0df82d6716f328be89bb21895e5325faffc9f`; the remote target
branch matched that commit before edits. Local branch is `codex/airline-failure-analyst`.
No existing modifications were present. No model, Keychain or paid native simulation
was invoked. This task explicitly prohibits an unsolicited push.

Existing Customer Skill v1, original RC-Bandit scheduler/reward, native tasks,
policy, tools, evaluator, E/V/H boundaries and legacy promotion remain available.
The two audited mismatches were confirmed: credit did not feed Service, and the
scheduler organized proposals through fixed direction hints.

New optional policy fields:

- `open_repair_search`: `open_repair_conditioned_search_v1`, mode `open_rc` or
  `free_search`. Mutually exclusive with `repair_conditioned_bandit`.
- `discovery_handoff`: `repair_discovery_handoff_v1`. Available with either search
  strategy; never applied to a historical experiment implicitly.
- `targeted_repair`: `discovery_targeted_repair_v1`. Requires handoff and independent
  loaded V. An E-only run cannot deploy a Skill under this protocol.

Algorithm, source fingerprint, optional policies and manifest bind a new experiment.
Old scores/caches/checkpoints are not imported. A source compatibility audit is still
required for any future import. Synthetic artifacts are NOT sources of real repair credit.

## Separate incumbent fitness and discovery yield

The incumbent is still selected only by strictly lower current-Service native E
accuracy; ties retain the incumbent. Bandit reward does not replace fitness.

For every eligible candidate, the existing dual-Service feedback executes the same
lawful compiled Customer on the promoted repair endpoints and all declared feedback
seeds. Complete actual Customer trajectory review, exact identities, valid native
outcomes, original Service/tool evidence at both endpoints, replicated residual or
regression labels and reviewer-supported novelty are necessary for credit.

The new archive revalidates the actual reviewed Trial and provenance, rather than
trusting `reward=1`. Discovery indexes hold Customer/procedure identities, source
Trial path/hash, original endpoint/message references, validity reports, reviewer
provenance, cost, novelty scope and incumbent-selection status. Full trajectories
remain in their original files; indexes do not duplicate all E rollouts.

One normalized mechanism/task/type key has one immutable Discovery record. Duplicate
items in a single review index the first supported item once; the unmodified complete
review remains in the Trial. Conflicting writes, source changes, symlinks and foreign
manifest identities fail closed. Separate immutable use/evaluation events record
prompt handoff, frozen target status, target native outcomes, deployment and Gate
artifact references. Unselected Customer discoveries are retained equally.

## Discovery → Service

After this generation's Customer evaluation and selection, all credited discoveries
are published. The same generation's Service Analyst/Mutator receives explicit
`discovery_repair_targets`, including the observable mechanism hypothesis and S+
trajectory evidence. These are **E training repair targets**, never validation data.

The projection uses `build_service_mutation_evidence`: task ID, observed native
outcome, original dialogue/tool fragments and digest/index references only. It drops
scenario, description, Customer hidden objectives, gold/reference actions and private
Customer review explanations. Service history continues to expose E effects only;
new targeted configurations also mask V rejection feedback.

Frozen targets are chosen by Trial ID then Discovery ID before Service generation.
Default maximum is one per generation, binding every Service proposal in that
round to the same target. All discoveries remain visible; only the frozen subset
gets additional targeted evaluation. Existing Service proposal/crossover counts
are unchanged. When no discovery exists, the original Service path remains.

The next Customer uses only the previous completed promoted Repair Pair. Its source
cells and compressed evidence include the actual accepted discovery-target repair,
in addition to incumbent E results. There is no current-generation future repair
information in Customer generation.

## Targeted promotion

Default frozen rules:

```yaml
targeted_repair:
  protocol_version: discovery_targeted_repair_v1
  max_targets_per_generation: 1
  min_fixed_cells: 2
  max_target_regressions: 0
  require_all_seed_improvement: true
  require_zero_hard_violations: true
```

1. Existing structural, semantic, policy and Skill-budget validation still run.
2. Re-execute the frozen discovery challenge at its task and all declared seeds on
   the current deployed S+ and proposal S'. A stale Service frontier, wrong strategy,
   task or seed is an error. Unknown outcomes are INCONCLUSIVE.
3. Require replicated native fail→pass improvement, positive net fixes, no target
   regression, no new stuck condition and no hard violation. By default **every
   declared seed must improve**. Reviewer praise cannot satisfy this check.
4. E initial screening protects incumbent successes. Full E protection evaluates
   current, Native and enabled archived Customers under frozen repair seeds using
   the existing observed preservation check. A target fix cannot override regression.
5. Keep independent V task-block/finite-panel Gates, seeds, look correction and risk
   policy. For targeted proposals the V objective is preservation for all opponents;
   target E improvement owns superiority. This change applies ONLY to the new
   protocol, not legacy E-superiority or V-primary experiments.
6. Any required rejection rejects; insufficient/unknown evidence prevents deployment.
   Complete target and protection results, actual cost counters and decision reasons
   are retained. Additional target/protection episodes use the existing panel runner,
   frozen cache identity, journal and interruption handling.

This certifies observed training-challenge repair plus the configured V evidence.
It does not establish causal Skill benefit, independent V generalization or H gains.
V is repeatedly used by search; statistical calibration and multiplicity limits remain.

## Open generation, not dynamic arms

Open-RC never invokes `choose_arm`, computes UCB or emits a direction hint. It uses
changed SkillMemory, the previous promoted pair's E observations/compressed evidence
and prior lawful/invalid/duplicate/ineffective procedures and discoveries. Hypotheses
and substantive distinctions remain open natural language inside the existing
Customer Skill schema; trigger/procedure/stop conditions and legality are unchanged.

Its prompt removes the old four-direction example anchoring. Empty evidence refs
remain permitted for a genuinely new hypothesis; supplied refs must match original E
messages exactly. No pair is explicitly ordinary open exploration, yielding no fake
repair reward. Free Search receives ordinary candidate history, not pair, repair
mechanism or discovery feedback in its generation input. It nevertheless receives the
same downstream dual-Service measurement workload for cost-matched comparison.

Default maximum trials, two feedback seeds and evidence limits are explicit frozen
policy. No new model role, dynamic-arm generator or extra debate is introduced.
Unknown costs remain null; attempted calls, unavailable usage and failed trials are
recorded separately from completed native episodes. Open and Bandit states/ledgers
have distinct protocol identities; interruption replays completed stages without
extra requests or credit.

### Evidence contract issue found during implementation

Existing compression stores each retained message under `message`; the repair
reviewer checked `role` at the wrapper level. This prevented positive evidence
validation on the actual builder path. New handoff/Open identities explicitly unwrap
that exact message while keeping indices, original hashes and excerpt metadata.
Historical Bandit inputs/selection sequences are unchanged. The B comparison uses
unchanged Bandit allocation/reward with the *new* handoff identity, and must not be
represented as an exact replay of the old reflection input.

## Prepared configurations and offline entry

- A: `configs/airline-free-search-v1-e10-offline.yaml`
- B: `configs/airline-rc-bandit-v1-e10-offline.yaml`
- C: `configs/airline-open-rc-v1-e10-offline.yaml`
- C with promotion: `configs/airline-open-rc-targeted-v1-e10-v20-offline.yaml`

All use identical declared models/args, E10, five candidates, two feedback seeds,
G2, P2 and max_steps=200. A/B/C are E-only search templates; they cannot execute
V-primary deployment without separate V. The targeted template enables V20 and
leaves H disabled. Every template has a fresh output/checkpoint identity and
`real_provider_enabled: false`. No template is authorization for a paid run.

```bash
PYTHONPATH=src TAU2_DATA_DIR=/Users/spring/.cache/evotau/tau2-data-b7ea9074 \
  /Users/spring/RSI/Evotau/.venv/bin/python \
  experiments/execution/run-open-repair-offline.py \
  --config configs/airline-open-rc-v1-e10-offline.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

`--synthetic` additionally runs the real orchestration with explicit Fake Providers/
Runner: **E3/V1**, two declared seeds, G2. This fixture demonstrates an unselected
Customer discovery reaching Service, target improvement, E protection, V preservation,
archive and checkpoint. It is not an E10 online experiment or benchmark result.
`synthetic-manifest.json` freezes fixture/source identity; complete immutable stages,
trial references and generation output are saved independently. There is no live
execution flag or credential retrieval path in this launcher.

## Next minimal real comparison — proposal only

First require one real previously promoted, provenance-audited S-/S+ pair. Without
one, this comparison cannot answer repair-conditioned discovery efficiency. Use a
new E3 subset fixed before outcomes, two declared seeds, two candidates per A/B/C,
the same models/args, and no V/H or additional Service promotion. Evaluate legitimate,
reviewer-supported novel replicated Discovery Yield and cost per valid discovery;
record rejections, ordinary no-pair fallback and uncertainty separately.

Let N=3, K=2, R=2. For each method, dual feedback needs `2*N*R*K=24` episode conditions;
its S+ seed-1 cells can reuse *identical* candidate fitness conditions within that new
manifest. Add up to N incumbent conditions. Thus approximately 27 unique cold episode
conditions per method, 81 across three, **plus real repair acquisition cost**. Invalid
candidates skip dual runs; incompatible cross-method caches are not shared.

Cost is NOT simply 81 multiplied by an E3 rehearsal total. Calculate per-cell native
Customer/Service calls, optional Activator calls, native evaluator assertions, all
semantic/actual-trajectory validators, six generator and discovery-review requests,
repair acquisition and any authorized recovery attempts. Use archived usage by call
name and episode trace lengths at matching runtime settings to obtain intervals;
proxy tokens are not provider usage tokens. Token cost is the sum of actual input/
output usage for each request. At max_steps=200, a loose native structural bound is
400 model calls per episode plus task-specific evaluator calls; it is an analytical
upper estimate, **not an imposed request cap or expected consumption**. No reliable
new-method expected dollar cost has been measured yet. Authorization must approve
fresh frozen live configs and the full quoted cost interval before execution.

A later targeted-promotion run adds per candidate target E conditions, full-E current/
Native/archive protection and all independent V opponent panels. Those costs must be
included when reporting Service Targeted Repair Success, regression or V/H outcomes.
Do not silently label search-only cost as total-method cost.

## Remaining research risks

1. Independent provider executions can diverge despite equal task×seed. No activated
   Skill causal effect follows from paired outcome flips; A/A remains necessary.
2. Repair-relatedness/semantic novelty relies on the existing Evolver's evidence review.
   Normalized-key dedup does not prove global novelty or remove evaluator subjectivity.
3. Adaptive V reuse, small task panels and previously touched H cannot establish broad
   generalization. Target E gains and Synthetic acceptance must stay separately labeled.
