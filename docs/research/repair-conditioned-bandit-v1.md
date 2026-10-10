# Repair-Conditioned Bandit v1 (opt-in)

Engineering implementation, not live effectiveness evidence. Customer Skill v1,
τ-bench scoring, Customer strict lower complete-valid E accuracy selection, and
Service Screen/full-E/V promotion are unchanged. No API experiment is authorized
by this document or the offline configuration.

## Closed loop and timing

`E → Service promotion → freeze S−/S+ → NEXT generation arm → structured Customer
Skill v1 → static validity → E on current Service → same Customer E on both frozen
Service endpoints/declared seeds → actual Customer validity → evidence review →
Trial → version-local next arm`.

`skill_evolution.run_skill_evolution_v2` saves the pair only after existing Service
promotion. A generation cannot read its own future repair. The next Customer must
see the pair's deployed S+ identity. If the previous generation had no promotion,
the frontier resets to `no_repair_pair`: deterministic uniform arms, zero repair
reward, no additional dual-Service evaluation. We deliberately do not silently keep
an older frontier through unproductive generations. Old versions remain auditable.
Rejected Service candidates may be compared separately as `shadow_repair`; they
never become this frontier, get reward, or become a deployment.

## Frozen allocation

Optional `skill_evolution_v2.repair_conditioned_bandit`:

```yaml
protocol_version: repair_conditioned_bandit_v1
beta: 1.0
window_trials: 50
open_exploration_interval: 5
max_trials_per_generation: 2
feedback_seeds: [1, 2]
min_replications: 2
max_context_tokens: 30000
```

Absent means byte-identical legacy policy serialization and original proposal
batching. Present requires Customer Skill v1. Manifest/source/config identities
freeze the new scheduler separately; old checkpoints cannot silently acquire it.
A different parameter/pair/source requires independent provenance/configuration.
Candidate count cannot exceed the declared per-generation pull budget.

A pull requests **one** independent candidate. Arms are directions, not a new Skill
schema: disclosure timing, request decomposition, authorization boundary,
correction/recovery, and open exploration (new lawful mechanisms remain possible).
Choices and ties are deterministic in that order. Each pair gets separate statistics;
no version carries old reward as current evidence. A quota forces open exploration
at least once in every declared interval (at most five). Otherwise untried arms get
cold start, then the last `window_trials` for that version determine
`mean + beta*sqrt(log(1+total)/(1+arm_count))`.

## Feedback contract

`repair_feedback.paired_observations` requires unique complete E task×seed cells,
known native outcomes, exact Customer compiled identity, explicit distinct Service
identities, and complete validity at both endpoints. Raw rewards/termination/source
refs remain archived. Four observed transitions:

| Before | After | Class | Credit |
|---|---|---|---|
| fail | pass | repaired | Coverage only |
| fail | fail | residual | No credit from labels alone |
| pass | fail | regression | Suspected regression, not causal proof |
| pass | pass | stable | Protection control |

Positive credit requires **all** actual Customer cells valid, a promoted pair,
repair-related and novel reviewer-supported mechanism, source-bound original
Service/tool evidence at **both** endpoints for every predeclared replication seed,
at least two seeds, consistent discovery classification, and no previously credited
normalized `(task, discovery type, observable mechanism)` key across versions. The existing
Evolver role performs the structured evidence review; there is no new model/agent
role. Reviewer prompt also receives prior mechanism descriptions to assess semantic
novelty; exact normalized dedup is programmatic, broader novelty remains model-assessed.
Repaired/stable alone, invalid/uncertain/duplicate candidates, a single flip, unknown
outcomes, missing reviewer support, or shadow pairs cannot get positive credit.
`causal_service_failure_confirmed` is always false. Identical seed ≠ identical
conversation; even replication and reviewer approval are observational evidence.

All E results stay visible. `evolution_context.build_repair_pair_evidence` expands
matching representative **tasks** at both endpoints with **all their seeds**, plus
success/repair controls, original indexed/hash-bound messages, tool arguments and
necessary results. Service-safe projection excludes hidden Customer scenario/gold.
No V/H trajectory, score or rejection explanation enters arm prompts or repair reward.
Raw trajectories remain untouched. Oversized repair prompting errors explicitly before
proposal; an oversized complete repair-review context yields pending/zero credit with
its reason, not silently cut evidence. Normal Customer prompt/trajectory allowances
remain its existing separately frozen policy.

## Costs, failures and resume

`BanditTrialLedger` is independent of bounded `CustomerAttemptHistory`. Under
`rc-bandit-trials/`, immutable start/failure/final events bind manifest, E task IDs,
choice, candidate/procedure identity, raw paired records, evidence/report provenance,
score, generation and actual cost. Candidate evaluation SHA links the ordinary
Customer artifact. Recorded reviewer requests bind input/output digests and model.
The RequestBudget remains global: native episodes, Activator, Evolver and validation
calls all count. Trial deltas record API calls, tokens, unknown-usage count, denials,
review calls, completed/started native attempts, and explicit simulated counters.
Per-call usage snapshots are retained before/after. Unknown tokens are never estimated
as zero. Shared incumbents and Service-search expenses remain in global API usage;
use **whole-run** cost, not just Trial deltas, for comparisons.

Schema rejection, static rejection, trajectory invalid/uncertain and duplicate
candidates settle a zero-reward consumed pull. JSON/transport/budget/runtime exceptions
halt through the existing fail-closed protocol and publish unresolved failure + cost
snapshots; they are **not** transformed into native failures or silently retried.
Pending failed pulls cannot be skipped to cherry-pick successful arms: evolution stops
until explicit existing recovery resolves them. A later final settles the *same* pull;
failure snapshots are cumulative evidence, not additional charges to sum again.
The journal freezes choice, proposal, both E panels, per-cell review and final feedback.
Generation checkpoints restore pair and versioned state; replay does not publish a
second reward. Recorded provider caching and native cache remain the sole request
reuse mechanism. In-flight/unknown dispatched requests still require existing explicit
operator recovery; no blind resend, automatic model change, or incompatible import.
Costs require monotonic restored global accounting. Counter reset fails closed.

## Offline commands and fixtures

The new independent configuration uses Airline E10 IDs
`1,0,39,49,11,27,34,12,7,15`, fitness seed 1, V/H bodies sealed, G2 and five scripted
pulls per generation. Its model mapping is declared metadata, never called.
`run-rc-bandit-offline.py` has no execute/live option:

```bash
PYTHONPATH=src python experiments/execution/run-rc-bandit-offline.py \
  --config configs/airline-rc-bandit-v1-e10-dryrun.yaml --tau2-data-dir "$TAU2_DATA_DIR"
# Add --scripted for independently labelled synthetic transitions and allocation.
```

Default only validates pinned inputs and E split. `--scripted` saves full synthetic
records, source hashes, ledger and two allocation rounds; it is **not** an alternating
τ run, not a real reviewer, and not an estimate of learning benefit. The tests also
run the actual alternating G2 control flow with Fake Providers/Runner: promotion,
next-generation dual-Service E, full validity, feedback/archive/checkpoint, interruption
and exact replay. Legacy Customer-only diagnostic remains single Service and unchanged.

## Next E-only experiment (not started)

First demonstrate live Customer Skill validity with the existing Customer-only
configuration, under separate authorization. Then construct an audited real promoted
pair. Create new independent E-only configurations for uniform allocation, current-failure
UCB and RC-UCB; use identical tasks/models, repetition seeds, candidate legality chain,
and **total** native/API/token budget including evidence reviews and failures. If only
shadow repairs exist, label diagnostics as shadow and make no RC promotion claims.

For N E tasks, K candidate pulls, R feedback seeds, complete valid paired evaluation
needs up to `2*N*R*K` candidate native episodes, plus shared incumbents and the cost
of obtaining Service repair pairs. Current-S fitness episodes may be reused only on
exact native cache identities. Extra research calls per paired candidate are one
generator, one static review, up to `2*N*R` complete-cell validity reviews and one
repair-evidence review; cache reuse reduces actual calls, never the planned upper count.
Estimate tokens using archived *per-role* distributions and actual turn-length ranges,
including Activator; estimate cost with the selected provider's verified rates.
Freeze those estimates/budget and require live authorization before executing.

Remaining hypotheses: live legality/novelty, reviewer reliability, seed independence,
repeatability under nondeterministic providers, discovery efficiency vs strong
cost-matched baselines, and downstream V/H generalization. G2 with very few trials
mainly tests wiring/cold start, not reliable online learning. No claim that Bandit
curriculum is novel; the cited ActiveSaddler/SCOPE research must be independently
verified before a paper's novelty argument.
