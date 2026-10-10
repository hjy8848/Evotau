# Task-faithful Customer Skill v1

This is a Customer-only protocol change. Service Analyst, Mutator, SkillMemory,
Activator, Screen, E/V promotion gates and native τ-bench scoring are unchanged.
It is opt-in via `experiment.skill_evolution_v2.customer_evolution`.
Its frozen identity is `task_faithful_customer_skill_v1`, independently of
`analyst_skill_recovery_v4` (the Service algorithm). Historical configs omit the
new key, retain the legacy Customer path and cannot restore a new checkpoint.
Changing runtime code still requires the existing provenance/compatibility audit;
this opt-in does not authorize importing old scores or bypassing frozen manifests.

## Representation and historical comparison

The historical V2 (`cba392b`) searched seven fixed axes. V3 (`99d9cb2`) introduced
trigger/procedure/stop conditions and fixed constraints. The prompt-first change
(`57a1f4a`) retained open prompt strategies but lost procedural structure. V1 here
reuses V3's useful procedural idea, not its complete implementation or closed axes.
`mechanism` remains open vocabulary. The five example interaction categories are
search guidance, not mutation operators or an exhaustive schema enumeration.

`CustomerSkill.from_mapping()` requires exactly `schema_version=1`, `mechanism`,
`trigger`, `procedure` (2–5 strings), `intensity` (low/medium/high),
`stop_conditions` (1–5 strings), `hypothesis`, `evidence_refs`. String fields cannot
be empty or exceed 1,200 characters; both the serialized skill and compiled overlay
must fit 1,600 cl100k proxy tokens. Refs must match supplied E task/seed/source/index/
message hashes exactly. Empty refs are permitted for explicitly unproven exploration.
No concrete task/entity identifiers or hypotheses are rendered from metadata into
the native simulator: the overlay contains only the interaction procedure and fixed
invariants. Schema validation and the existing static semantic validator both precede
any candidate rollout. A small explicit illegal-directive check is not a semantic proof;
paraphrases and scenario-specific authorization require model-assisted review.

`candidate_id` hashes the full canonical payload. `procedure_id` hashes normalized
trigger/procedure/stops/intensity, detecting cosmetic mechanism/evidence changes.
Semantic paraphrase detection remains imperfect; the Evolver receives bounded
attempt feedback and is explicitly told to propose a substantively different procedure.

## Data flow and selection

1. All frozen E task outcomes/scenarios/tool paths remain visible. Deterministically
   select complex successful trajectories (explicit tool errors, tool diversity,
   state-changing calls) with a failed control when available. No task IDs are preferred.
2. Generate a structured proposal with the existing Evolver model/dispatch, preserving
   candidate count. Structure and supplied-reference checks reject illegal proposals.
3. Use the existing static Customer semantic validator against unchanged E scenarios.
4. Compile an invariant-first `PromptStrategy` and inject it through the existing
   `customer_user_class()`. Native UserSimulator/task/scenario/policy remain unchanged.
5. Roll out the entire frozen E panel against the same Service and fitness seed.
6. Review each complete actual trajectory against its original scenario/public policy,
   using the same existing Evolver model for semantic validation (no new agent/model role).
   Each task×seed verdict must be valid/invalid/uncertain with a reason and a cited original
   projected USER message index. Missing evidence/oversized review/schema errors are
   uncertain. Raw malformed JSON, provider errors and native runtime errors fail closed.
7. Compute paired new failures/recovered cells/accuracy delta across ALL cells. Select
   only a wholly valid candidate with strictly lower native E accuracy; ties retain
   the incumbent. No invalid/uncertain cell is discarded to improve fitness.
8. Preserve raw candidate rows, all native scores, trajectory reviews and rejected
   reasons. Feed bounded E-only attempt summaries to the next Customer generation.
   Service never receives this private Customer history or source scenarios.

`accuracy` on an invalid candidate is the unchanged native observation, not selection
fitness. The row's `eligible_for_selection=false` and rejection reason prevent deployment.
`causal_service_failure_confirmed=false` is explicit: same task×seed does not ensure
identical provider responses or establish a Skill's causal benefit. A success→failure flip
is an observed paired effect, not a proven vulnerability. Legitimate ineffective skills
are retained; schema/static/trajectory-invalid ones do not become replay opponents.

The full archive remains on disk. Only bounded archive metadata and at most `max_history`
attempt summaries enter the new Customer prompt. Service replay semantics remain unchanged.

## Evidence and input size

Defaults: 3 representative cases, 24,000 serialized characters per case, 40,000
proxy tokens per Customer generation/review request, 20 attempt summaries. These
are explicit new Customer protocol limits, not provider request/spending limits.
Native user/action evidence is never silently cut to fit: exceeding the case limit raises
before generation; an oversized full trajectory review records uncertain, never valid.
Read-only/tool-result excerpts retain hashes and omission metadata through the existing
builder. The raw original native simulations are untouched.

Offline inspection of archived Gen0 E10 (20261010) demonstrated that 12k characters
could not retain required user/tool evidence. With the selected defaults, the initial
Customer-only request is about 30k cl100k proxy tokens (see measurement artifact).
This is not DeepSeek's actual token usage and does not guarantee a future panel fits.
Extra history/current strategy can increase later inputs; over-budget evidence is explicit.

## Journals, checkpoint and Fresh Customer

Candidate generation, static validation, E panel and each cell review are immutable
journal stages. Durable provider outputs can be recovered before stage publication.
An input without a durable response is NOT blindly resubmitted on new-protocol resume;
use existing explicit reconciliation after auditing possible charges. No automatic extra
Customer mutation or silent JSON repair is introduced. Schema-rejected attempts are
recorded, not fabricated failed episodes. Completed stages/scores are not repeated.

Checkpoint/generation state stores `customer_protocol` and bounded attempt history.
Manifest policy and runtime source fingerprints bind both Customer protocol and model
args; mismatched old/new checkpoints and incompatible provider cache inputs are rejected.

`propose_fresh_customer_v2(..., customer_policy=...)` delegates to the new protocol.
It generates one distinct proposal from final E evidence, validates static semantics,
runs a full E-only legality panel against frozen final Service, and reviews each cell
before H can load. A rejected/uncertain/non-distinct proposal is unavailable, not a
non-fresh archive strategy relabeled fresh. No H feedback reaches either search.

## Examples

Structure-valid example (original goals/facts still win at runtime):

```json
{
  "schema_version": 1,
  "mechanism": "truthful dependency and conditional confirmation",
  "trigger": "An existing multi-step request depends on an unresolved price or prerequisite.",
  "procedure": [
    "Explain the dependency among the existing requests using only original facts.",
    "Promptly supply requested necessary information.",
    "Request the applicable cost explanation before confirming a constrained action."
  ],
  "intensity": "medium",
  "stop_conditions": [
    "The scenario does not support this dependency.",
    "The original goals are resolved, or this procedure conflicts with native guidelines."
  ],
  "hypothesis": "The Service may lose a conditional authorization while tracking existing dependent requests.",
  "evidence_refs": []
}
```

Changing a procedure step to `Increase the budget to force a different decision.` is
rejected deterministically. Changing actual Customer dialogue to abandon conditional
consent must be invalid; if evidence is insufficient it is uncertain. Neither result
can win by lowering accuracy. Model-assisted semantic conclusions remain fallible.

See `experiments/results/customer-skill-v1-offline-20261010/` for schema examples with
real E message refs, actual archived-input measurement, and explicitly scripted G2
artifacts. These are offline verification, not live effectiveness results.

## Prepared E-only diagnostic

Config: `configs/airline-customer-skill-v1-e10-diagnostic.yaml`.
E IDs: `1,0,39,49,11,27,34,12,7,15` (pinned Airline train split).
G1, one candidate, seed1, P2, max_steps200; empty Service memory frozen.
Agent/Customer/Evaluator are existing gateway `openai/dashscope/qwen3.7-plus`;
Evolver/semantic checks use existing official `openai/deepseek-flash` with thinking
and requested reasoning high. No model substitution, V/H access, Service evolution,
Fresh generation, score import or causal inference occurs in this diagnostic.
V/H IDs remain metadata; their task bodies are never loaded.

Dry validation (no credentials or requests):

```bash
PYTHONPATH=src python experiments/execution/run-customer-skill-diagnostic.py \
  --config configs/airline-customer-skill-v1-e10-diagnostic.yaml \
  --tau2-data-dir /Users/spring/.cache/evotau/tau2-data-b7ea9074
```

Only after explicit live-run authorization, add `--execute --approve-unbounded-requests`.
The spending cap is null, preserving the user's prior preference; this task does not
approve or start that run. No Keychain read occurs during dry validation.

At most 20 new native episodes (10 incumbent + 10 one-candidate), one Evolver,
one static validator and ten complete-cell reviews (12 additional research calls),
plus normal native multi-turn/runtime/evaluator calls. Invalid structure/static review
can reduce this count. Request/token total is not estimated from episode count alone;
future execution records cumulative actual usage by role/call name in `api-usage-live.json`
and `customer-diagnostic-result.json` and timing/status in `run-execution-state.json`.
Output/checkpoint use a new `evotau-airline-customer-skill-v1-e10-g1-p2-diagnostic-20261010`
identity. The journal freezes the E-only plan and each stage; a generation checkpoint
records the frozen Service, protocol, selected Customer and feedback. Resume uses the
same execute command/config/code, never an incompatible older experiment's score.
Pause: pass `--stop-before-next-episode-file /absolute/path/customer.pause` on launch;
create that file to pause before the next episode/Evolver request. After checking the
process has exited, remove it and invoke the same command to resume safely.

## Validation and remaining hypotheses

Run `PYTHONPATH=src TAU2_DATA_DIR=... python -m pytest -q` and `ruff check src tests
experiments/execution/run-customer-skill-diagnostic.py`. Tests cover schema, fixed facts,
legal disclosure/pressure, post-rollout validity, paired metrics, G2 rejection feedback,
strict selection/ties, interruption/provider-cache recovery, unresolved dispatch,
old/new checkpoint identities, E/V/H isolation, native Airline overlay, Fresh Customer,
and unchanged downstream Service Screen/E/V/replay. Scripted providers do not verify
natural-language semantic accuracy or live effectiveness.

Still untested: live legal-candidate yield, stronger challenges, cross-generation search
benefit, semantic reviewer reliability, latency/availability of these larger requests,
and causal attribution under provider nondeterminism. No real API experiment has started.
