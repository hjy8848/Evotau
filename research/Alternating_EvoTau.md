# EvoTau: alternating self-evolution on τ-bench

## Research definition

EvoTau fixes τ-bench as its task and environment substrate. Its research addition is a Customer–Service alternating loop:

```text
(C_t, S_t) → (C_(t+1), S_t) → (C_(t+1), S_(t+1)) → …
```

With S frozen, Customer evolution searches for task-grounded interaction strategies that reveal the Service's current failure boundary. This covers natural cooperative difficulty and natural policy conflict without separate Customer systems or predefined challenge classes. With the selected Customer frozen, Service evolution studies actual τ-bench trajectories and proposes a natural-language strategy update.

The method is prompt-first. The strategy carrier is open natural language; the first version does not encode customer behavior axes, mutation operators, rule records, challenge budgets, taxonomy-based fitness, or a strict multi-panel repair gate. Failures, non-improvements, and strategy drift remain visible research outcomes. Constraints protect benchmark integrity and information isolation, rather than determining which interaction behaviors may be generated.

## Fixed substrate and split boundary

τ-bench continues to provide tasks and scenarios, Retail policy, initial backend state and state transitions, tools, orchestrator, native Customer/Service runtime, evaluator, reviewer, and benchmark data. EvoTau appends strategy text to the native Customer and Service prompts and invokes the pinned τ-bench runtime. It does not modify upstream task truth, policy, tool code, backend semantics, or evaluation.

E and V are loaded from the official train split. Evolvers and selection judges receive only E task intent/scenario, the current strategies, visible trajectories and tool outcomes, native task result, reviewer feedback, and relevant prior generation feedback. They do not receive reference actions, evaluation criteria, gold answers, H task objects, or H trajectories. H is loaded after evolution and fresh adaptive-Customer generation. Final evaluation compares S₀ and S_T under both native τ-bench Customer and an E-derived fresh adaptive Customer.

Service updates are judged first on the same evolved-Customer challenge. A small native-Customer V panel then prevents an update from converting a previously successful reference case into a failure. The clean reference is the current generation's S_t, not a permanently frozen S₀. There is no failure ontology or independent audit protocol between observed trajectory and evolution.

## Source execution map

```text
evotau-evolve / alternating_run.run_from_config
  ├─ AlternatingManifest + pinned E/V loader
  ├─ TauBenchEpisodeRunner
  │    └─ tau_adapter → τ-bench Orchestrator/run_simulation/evaluator/reviewer
  └─ alternating.run_alternating_evolution
       ├─ Customer Evolver → candidate prompt overlays
       ├─ E episode runs with S_t frozen → Customer judge → C_(t+1)
       ├─ Service Evolver sees selected Customer trajectory
       ├─ E replay under proposed Service → same-challenge judge
       ├─ small native-Customer V comparison against S_t
       └─ generation artifact + checkpoint
            ↓
       fresh Customer proposal from E only
            ↓
       load H → compare S₀/S_T × native/fresh Customer
```

The central callable is `evotau.alternating.run_alternating_evolution`. The command-line handoff is `evotau.alternating_run.run_from_config`. A generation artifact contains before/after Customer and Service snapshots, E episode references, selection reasons, clean-panel episode references, and provider-usage deltas. The episode runner retains the full native simulation, record, and per-call usage telemetry.

## What changed from the former design

Removed from the active method: V1/V2 fixed-axis Customer types and mutation operators; V3 behavior parser and keyword/regex rejection; challenge budgets; the fixed Retail MVP failure taxonomy as an evolution prerequisite; failure confirmation/strict attribution in Customer selection; structured policy-rule Service patches and patch token ceiling; the multi-stage Service transition and repair gate; Pilot condition machinery and its static/random/frozen variants; strict preregistration, power, and old RQ1–RQ3 analysis tied to those mechanisms.

Retained because they support execution or observation: the pinned τ-bench adapter and native evaluator/reviewer; source/task fingerprints and frozen role settings; request/token accounting; E/V/H panel loader; complete episodes, checkpoint and manifest; and the local Console's run launch, artifact and trajectory views. The current Console reads Phase 0 and alternating-run schemas. Older experiment directories remain untouched as files, but their retired Pilot, failure-taxonomy, and cross-play schemas are not carried forward as a second runtime or reader.

The one-episode Phase 0 command remains a connectivity check only. It does not run evolution and is not a second strategy method.

## Evidence and open risks

### Verified in software

- Unit tests verify that Customer and Service receive open prompt overlays and τ-bench base prompts remain intact.
- Fake-runner tests verify that the Service is fixed through Customer search; the selected Customer is fixed through Service search; the clean comparison uses the generation's S_t; multiple generations chain from the previous pair; and H content is absent from evolution/fresh-Customer contexts.
- Artifact tests verify generation snapshots and checkpoint persistence.
- A deterministic offline native-runtime smoke was run successfully. It completed one alternating generation and 8 native τ-bench episodes across Customer selection, the Service challenge, the E-only fresh challenge, and four H endpoint cells. The fixture judge rejected the Service proposal, so the conditional clean V comparison correctly did not run. The Retail orchestrator executed `find_user_id_by_email`; native scoring and review ran. Only HTTP model completions were stubbed locally. The smoke uses `max_steps=2` and does not demonstrate task completion. The fixture judges retained the Customer incumbent and rejected the Service proposal because the fixture trajectories were identical; these are wiring outcomes, not model-quality evidence.

### Inferred from the implementation

- A live run follows the same transition sequence with provider-backed role models because all EvoTau and τ-bench calls use the shared τ completion boundary and frozen role settings.
- H remains inaccessible to evolvers during normal CLI execution because the H task loader is called only after E-only fresh-Customer generation.

### Main risks

- The selection judges can be inconsistent or reward persuasive explanations over a better trajectory.
- Open text can drift from the original task or invent facts; the first version records this for analysis rather than blocking generation.
- A small V panel only catches obvious clean-task regressions.
- One task per panel in the example is a wiring smoke, not a generalization result.
- The local deterministic completion stub cannot establish live provider compatibility, model quality, or positive evolution.

## Operator map

| Question | Where to look |
|---|---|
| Where does Customer evolve? | `LLMAlternatingEvolvers.customer_candidates`, called from the Customer phase in `run_alternating_evolution`. |
| Where does Service evolve? | `LLMAlternatingEvolvers.service_candidate`, called after selecting C_(t+1). |
| What does τ-bench do? | Task/scenario, policy, backend, tools, conversation runtime, native evaluation and reviewer. |
| What did EvoTau add? | Prompt strategy overlays, trajectory-based LLM evolution/selection, the alternating loop, split isolation and artifacts. |
| Where does one generation start? | `run_alternating_evolution`, loop body at `for generation in range(...)`. |
| If evolution does not happen, where to inspect? | `generation-XXXX.json`: Customer candidates/selection, Service analysis/proposal/selection, E challenge episode records and V clean panel. Then inspect the referenced native simulation and usage telemetry. |
