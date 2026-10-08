# Direct Skill Evolution V2.1

Algorithm version: `direct_skill_evolution_v1`. This is a new research treatment, not a timeout fix or an in-place continuation of a frozen Diagnoser experiment. No live API request was made during implementation.

## Flow

```text
Customer proposal → Customer semantic validator → fixed E fitness / selection
→ deterministic E-wide evidence builder
→ Direct Skill Mutator (hypothesis + typed mutation in one inference)
→ structural / semantic / budget / dedup validation
→ paired Screen → Full-E superiority → V preservation gates
→ promotion / archive / lineage → generation commit
→ fresh Customer semantic decision → native / available fresh H endpoints
```

The independent `diagnose()` call, Diagnoser prompt/schema, `service_diagnosis` stage and `recommended_surface == skill` filter are removed. Observed MAX_STEPS or weak signals can inform a behavioral hypothesis; no repair-surface classifier blocks proposal generation. NO_OP, rejected candidates and no promotion remain valid outcomes.

## One canonical mutation contract

Entry: `V2Providers.propose_skill_mutation(context)` in `evolution_candidates.py`; orchestration: `run_skill_evolution_v2()` in `skill_evolution.py`.

Inputs contain all E task×seed outcomes/tool-path overviews, bounded original failure evidence and passing contrasts, current V2 memory, native policy, E-only mutation effects, prior fixes/regressions and a proposal bias. Customer hidden scenarios, task descriptions, gold targets, V trajectories and H content are excluded. Current outcome metadata is allowlisted; original trajectories remain immutable on disk.

The canonical typed mutation retains `operation`, `target_skill_id`, `skill`/`children`, `analysis`, `semantic_family`, `substantive_delta_from_prior`, and adds:

- `root_cause_hypothesis`, `expected_effect`, `regression_risk`: explicit natural-language hypotheses, not ground truth.
- `evidence_task_ids`: observed failed E targets; `protected_success_task_ids`: observed passing E controls.
- `evidence_refs`: exact `{task_id, seed, trajectory_ref, projected_message_index, message_sha256}` tuples from retained original messages. Every non-NO_OP target requires a reference. Omitted messages cannot be cited as supplied evidence.
- `target_cluster_id`: a candidate-owned hypothesis ID, kept for the existing effect/archive lineage contract. The lineage also records its root-cause hash and `source: direct_candidate`; it is no longer issued by a separate Diagnoser.

Old `expected_fixes` / `protected_cases_at_risk` model outputs are not silently converted. Malformed output stops with immutable response and schema-error evidence. Existing explicit, bound schema retry authorization remains required. Cache identity includes prompt, input, model and args; old Mutator calls cannot satisfy new requests.

## Compression and bounded search

Customer and Service share the deterministic evidence-building implementation with separate information permissions and case selection. Default direct evidence expands four cases, prioritized failures plus the passing control with the closest visible tool path. All E outcomes remain visible. `mutation_context.representative_cases` can explicitly expand coverage in a new frozen config; this is not permanently limited to four.

Default `case_chars: 12000` preserves decisive user quotations, state-changing parameters and necessary results; optional excerpts/field projections and omissions are labeled and digest-linked. Required evidence exceeding this allowance raises an error. Prompt plus full mutation context is checked against `max_proxy_tokens: 40000` with `cl100k_base`, including crossover. Oversize input fails before provider dispatch; no silent trimming occurs. The cap is an engineering guard, not a provider token count.

On the saved E20, rebuilding from original native simulations yielded **27,022 proxy tokens** for a generation-zero narrow-applicability request. Source hashes and references are in `research/audits/direct-skill-evolution-20261008/context-measurement.json`. Later generations include memory/history and can be larger. This measurement imported no episode scores/cache into a new experiment.

`service_evolution.candidates_per_generation: 3` preserves the previous 3×1 independent search budget. Biases cycle narrow applicability, minimal behavior, structural decomposition. At most one additional crossover is conditional on complementary local repairs; it must independently pass all checks. Maximum candidate count is therefore four in this config, not four obligatory model calls.

Legacy `candidates_per_cluster × max_clusters_per_generation` explicitly migrates to the fixed per-generation product, with migration metadata. Legacy `history.representative_cases` migrates to the direct evidence limit unless an explicit `mutation_context` is supplied, also recorded. Missing algorithm version means `diagnoser_v2`: readable historical configuration, forbidden for execution on current HEAD. Do not edit frozen configs to switch mechanisms.

## Fail-fast V Gate

`evaluation.v_gate_mode: fail_fast` stops remaining V opponents only after a required V gate returns **REJECTED**. Already completed gates remain saved; skipped opponents are frozen as **NOT_EVALUATED**, and `risk_profile_complete: false`. INCONCLUSIVE does not trigger this shortcut. Provider/schema/runtime errors remain exceptions, not scored rejection.

`full_audit` retains all opponent evaluations for a complete risk profile. Both modes freeze the same maximum look count in advance: direct candidates plus possible crossover, multiplied by required opponent gates and Full-E superiority. Early stopping does not reduce multiplicity correction, weaken thresholds, or permit promotion with missing gates. A rejected candidate never changes accepted memory. Finite V3/H5 outcomes do not establish population generalization.

Activator, task×seed screen protections, Full-E attribution, replay, archive/crossover, fresh Customer rejection protocol, H loading boundary and native evaluator are unchanged.

## Independent configuration and commands

Prepared config: `configs/v2-direct-skill-qwen37plus-inferai-v4pro-e20-v3-h5-g2-p1.yaml`.

- E20/V3/H5 retained, excluded 46/47, G2, fitness seed1, P1, max_steps32.
- Native Agent/Customer/Evaluator/Activator remain InferAI Qwen3.7-plus, thinking disabled. Evolver remains InferAI DeepSeek V4 Pro, thinking enabled/high; no model substitution is part of this refactor.
- Independent experiment/output/checkpoint paths; no historical cache import. Proposed finite request cap100000; new launch confirmation is required. Prepared availability and useful mutations remain unverified live.

Offline validation (does not read H task content or fetch credentials):

```bash
PYTHONPATH=src python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-direct-skill-qwen37plus-inferai-v4pro-e20-v3-h5-g2-p1.yaml \
  --tau2-data-dir /path/to/pinned/tau2-data --dry-run
```

Future launch, only after authorization:

```bash
PYTHONPATH=src python experiments/execution/run-v2-live-evolution.py \
  --config configs/v2-direct-skill-qwen37plus-inferai-v4pro-e20-v3-h5-g2-p1.yaml \
  --tau2-data-dir /path/to/pinned/tau2-data --approved-request-cap 100000 \
  --stop-before-next-episode-file /tmp/evotau-direct-skill.pause
```

Pause by creating that file. Remove it and rerun the identical command to resume the same frozen run. Do not change models/config/native conditions or manually alter artifacts to resume. Budget-exhaustion/ambiguous in-flight request recovery remains governed by the existing release recovery protocol. Cross-version episode reuse requires a separate explicit compatibility audit of native task/model/args/seed/Customer/Service/evaluator conditions and import provenance; none was performed as an experiment import here.

Old Diagnoser results, failures and immutable provider outputs remain available for ablations. Console renders their legacy diagnosis fields and old mutation metadata read-only; new runs display direct hypotheses/refs and contain no Diagnoser request.

## Verification limits

Offline tests cover legal proposals without a diagnosis, NO_OP despite failures, invalid labels/IDs/digests, hidden metadata isolation, rejected-memory separation, Screen/E/V/replay, finite budgets, conditional crossover, exact resume and incompatible cache exclusion. Deterministic E20/G2 orchestration tests exercise fresh acceptance/rejection, H finalization and interruption/resume in Gen1 and H; these use explicitly synthetic outcomes. Native integration tests execute pinned τ-bench tools/evaluator with scripted local completions. They make no live provider requests and do not establish research effectiveness.

Review `evolution_candidates.py` for the contract, `evolution_context.py` for evidence preservation, and `skill_evolution.py` for bounded direct proposals and fail-fast V scheduling.

## Authorized official Flash continuation (2026-10-08)

`configs/v2-direct-skill-qwen37plus-official-dsflash-e20-v3-h5-g2-p1.yaml`
uses official `openai/deepseek-flash` for every Evolver/semantic-validator request,
with thinking enabled, high reasoning requested, max_tokens65536. InferAI Qwen runtime
roles and activator remain unchanged. The official launcher replaces authorization
only for `api.deepseek.com` POST requests to the configured no-tools model; it never
puts the official key in the runtime environment or alters InferAI authentication.

`import-direct-native-baseline.py` audits the recorded parent Git source digest,
unchanged native runtime bytes against the reviewed refactor commit, pinned source
blobs, tasks/models/args/seed/initial strategies/evaluator and activation condition.
It imports only the complete initial native E20 records/simulations. No Customer
proposal, Validator decision, selection stage, Diagnoser/Mutator cache or checkpoint
is imported. All Customer/Evolver output regenerates. Binding-only transformations
and hashes are recorded; prior cumulative usage is carried forward. The initial
baseline is 17/20; this is cached benchmark evidence, not a new treatment effect.

```bash
PYTHONPATH=src python experiments/execution/run-v2-official-deepseek-evolver.py \
  --config configs/v2-direct-skill-qwen37plus-official-dsflash-e20-v3-h5-g2-p1.yaml \
  --tau2-data-dir /path/to/pinned/tau2-data --approved-request-cap 100000 \
  --stop-before-next-episode-file /tmp/evotau-direct-official.pause
```

This is an explicitly changed-model/new-mechanism treatment with audited native
baseline reuse, not a pure new independent replicate. Availability of both required
model IDs was refreshed via their credential-scoped `/models` endpoints before
launch; listing alone does not prove the new Direct Mutator will succeed.
