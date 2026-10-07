# Skill Evolution V2

V2 extends the `6fa200421dead77db1bac5e9694dd67825428778` research baseline. It changes the external skill search and per-turn guidance selection. Native τ-bench tasks, facts, tools, policy, backend, UserSimulator, LLMAgent generation, evaluator and success semantics are unchanged. The pre-implementation audit is in [skill-evolution-v2-invariants.md](skill-evolution-v2-invariants.md).

## Data flow and ownership

```text
E native incumbent rollouts
  → task-faithful Customer proposal + semantic validation
  → native E candidate score (strictly lower wins; ties retain incumbent)
  → observed failing interactions + similar passing controls + bounded effect summary
  → Diagnose (skill / tool_boundary / runtime_protocol / stochastic_or_weak)
  → three independently frozen proposal biases per selected skill cluster
  → structural/semantic/budget/dedup checks
  → paired screen: targets + protected passes + prior fixes + seeded clean tasks
  → full E paired attribution for screen survivors
  → formal V paired gates under current + archived + native Customers
  → deterministic winner selection → accepted runtime memory + checkpoint
  → locally useful research archive, rejection lineage, failure matrix
```

Current Customer repair requires **superiority on paired E**. Archived and native Customers require **preservation**, so equal scores can pass. In V-enabled runs, a candidate must first improve current-Customer E, then preserve V against every replay opponent (including current Customer). E-only mechanism smoke uses these distinct objectives on E, with **NO GENERALIZATION CLAIM**. No weighted average can hide another opponent's regression. Screen protection covers each incumbent-passing task×seed cell, even a passing seed of a task that also contains a target failure.

Only accepted `ServiceSkillMemoryV2` enters runtime. Before each native Service generation, `SkillActivator` sees previous/current observable messages, tool results and the accepted catalog (ID, trigger, signature). It never receives task IDs, hidden scenarios, native reward, evaluator targets or future messages. Selected guidance is appended to the native policy for that turn only; the next turn rebuilds the prompt from the native baseline. An empty catalog records an empty decision without a request. K is at most two. The explicit `render_all_v1` ablation uses a no-provider selector to inject all accepted V2 skills and records what was actually rendered on each turn.

Customer proposal/validation can see E scenarios. Service diagnosis/mutation cannot see scenarios or task descriptions, which can contain hidden intent. Their task IDs are research metadata, excluded from runtime skill/activation inputs. Customer archive text is task-general validated interaction policy, not Customer Judge free text.

## Modules and schema

| Files | Responsibility |
|---|---|
| `service_skills.py`, `skill_evolution_config.py` | V2 signatures/memory, typed edits, count/character/token budgets, strict frozen settings |
| `skill_activation.py`, `tau_adapter.py`, `tau_episode_runner.py` | Direct native agent subclass, observable activation, per-turn sidecars, native completed-episode cache |
| `evolution_failures.py`, `evolution_gate.py` | Paired task×seed effects, failure matrix, cheap screen, task-block bootstrap and promotion |
| `evolution_candidates.py` | Diagnose/mutate/crossover/semantic-validation prompts and exact recorded-call recovery |
| `evolution_archive.py`, `evolution_history.py` | Semantic dedup, Pareto-capped archive, complementary ancestry, Customer replay, bounded history/stagnation |
| `evolution_artifacts.py`, `skill_evolution.py` | Immutable stage journal, activation summary, staged V2 orchestration and resumable search state |
| `alternating_manifest.py`, `alternating_run.py` | Manifest binding, V1/V2 dispatch, role usage, final E-only fresh challenge and H scorecard |
| `web/evolution_view.py`, `web/artifact_reader.py`, `web/run_progress.py`, templates | Artifact-only board, lineage, diagnosis, gates, archive, progress and activation traces |

Runtime V2 memory is `{schema_version: 2, skills: [...]}`. Each skill has `skill_id`, `trigger`, `guidance`, and `activation_signature` with positive/negative conditions and interaction phases. IDs are assigned by a persistent high watermark; rejected proposals do not free IDs for reuse. Supported intent is ADD, NARROW_TRIGGER, EXPAND_TRIGGER, REWRITE_GUIDANCE, SPLIT, DELETE, NO_OP. Applicability-only edits preserve guidance; guidance-only edits preserve trigger/signature. SPLIT assigns two new IDs. Extra draft fields cannot override generated IDs.

V2 checkpoint/result/generation schema is **3**. The checkpoint saves active memory/provenance, high watermark, accepted and rejected `MutationEffects`, full failure matrix, evolution archive, Customer archive, bounded history, activation configuration/statistics, statistical policy/looks, final E records and committed generations. MutationEffect includes paired fixes/breaks/persistent outcomes, helpfulness/harmfulness, stuck/step/tool/token/observed violation deltas, decision/reason and parent mutations. Candidate artifacts distinguish screen-only E attribution from full E attribution; V promotion evidence is separate. Archive Pareto dominance requires the same evaluation scope and exact paired-cell/baseline comparison key. Screen accuracy and full-E accuracy are never compared as if they shared a denominator.

`evolution-v2/<stage>.json` is an immutable envelope containing manifest SHA, input SHA, payload SHA and envelope SHA. Frozen diagnosis, proposals, semantic validation, panel records, screen, gate, selection and generation commit are reused only on exact binding. The provider-call record has a response SHA so a completed Evolver request can also be recovered after a crash before stage publication. Corruption or changed input fails closed. Per-turn activation artifacts and their aggregate trace have independent hashes bound to manifest and episode condition.

Native episode identity remains task + seed + Customer strategy + Service carrier, under a frozen manifest. Existing locks, bounded executor, request reservations and cache single-flight logic remain in use. Phases/generations stay sequential; no individual episode turn is parallelized. An incomplete episode cannot become fitness. Resume reuses successful episodes and frozen calls; it retries an incomplete attempt only when explicitly resumed. A runtime source/config change requires a fresh run, not reuse of an old score under a new treatment.

## Statistical and reliability interpretation

All pairs require identical unique task×seed cells. Missing/unknown scores are never failure. The bootstrap resamples **tasks as blocks**, retaining all seeds within a task; extra seeds are not extra independent tasks. Bonferroni corrects a frozen upper bound on candidate × replay-opponent looks within each generation. This is not a guarantee against every later adaptive analysis across generations or post-hoc experiment selection.

The frozen `statistical_gate.method` distinguishes two contracts:

- `task_block_bootstrap`: population-inference acceptance requires adequate independent tasks/seeds, a positive success lower bound for superiority (a nonnegative lower bound for preservation), the harmfulness upper bound, and stuck checks. A task-level Wilson bound prevents false zero-risk claims. Insufficient evidence remains `INCONCLUSIVE`.
- `finite_panel_paired`: a small fixed-panel check, requiring at least two paired seeds, zero observed pass→fail cells, zero newly stuck cells, and no hard violation increase. Superiority additionally requires strictly increased accuracy; preservation permits ties. Bootstrap/Wilson intervals are recorded as diagnostics, **not acceptance evidence**. `inference_scope=frozen_tasks_observed_seeds_only` and `population_risk_certified=false`. This contract cannot certify a population harmfulness bound or generalization.

`pass^k` is the within-task without-replacement estimate over available repeated seeds; unavailable trial counts are not invented.

The V-enabled example retains **E20 / V3 / H5** and explicitly selects `finite_panel_paired`. V3 can now accept observed preservation without pretending to substantiate the 5% population harmfulness bound. The bootstrap minimum remains eight; it has not been lowered. A formal population study must instead freeze `task_block_bootstrap` and an adequately sized V panel; even eight independent tasks may be insufficient for the risk bound. Neither a finite-panel promotion nor the E-only smoke substitutes for that study.

`max_steps` is independent reliability evidence (count/rate, mean/p95 steps). Token budgets use `tiktoken` cl100k_base for deterministic accounting, not claimed Qwen/DeepSeek billing tokenization. Rendered tokens are measured on actual selected blocks; API usage comes from provider responses, separately by role. Episode total tokens include runtime calls including activation; role-specific usage isolates Service and activator costs. An episode-level activation success rate is association, **not causal credit**. Per-skill overhead on a multi-skill turn is joint overhead and must not be summed as uniquely attributed cost.

Native success remains the correctness metric. Policy/protocol deltas count explicitly observed/instrumented violations; zero is not a proof of the absence of every policy violation. No new policy oracle or Reviewer has been added. Backend/provider errors fail closed and retain diagnostics, rather than being relabeled as behavioral failures. Some failing attempts only have individual activation sidecars; the completed aggregate trace/activation score statistics are unavailable for them.

## Backward compatibility and Console

V1 carriers, configurations and static render-all semantics remain supported. V1/schema1–2 artifacts remain readable. Missing old activation/effect/archive metrics display Unavailable or render-all / legacy; the Console never retroactively invents them. V2 does not import V1 scores/cache or automatically migrate an old run/checkpoint into a changed treatment.

`/runs/{id}/evolution` presents committed candidate boards, diagnoses/repair surfaces, expected and actual fixes/risks, task×seed outcome links, ancestry, statistical intervals/rejection reasons, archived research candidates, Customer replay and frozen live-stage evidence. The run overview includes Evolution Health; skill snapshots show signatures, provenance and observed activation statistics; assistant messages show their recorded per-turn decision. Compare marks runs NOT COMPARABLE when carrier, activation/activator, search, seeds, gate, replay or archive conditions differ.

The Console renders stored statistics; it does not run providers/bootstrap or select candidates. Checkpoint/journal/sidecar hashes and E/V evidence IDs are verified before rendering. H remains sealed until a complete final endpoint scorecard exists, including a completed E-only run with H disabled. Raw result exposure is suppressed or allowlisted while sealed. Redaction covers activation reason and other free text; real numeric token counts remain visible.

## Configurations and first experiments

New configurations:

- `configs/alternating-skill-memory-v2-qwen3-7-plus-retail-mechanism-smoke.yaml`: G2, C1, E20, P4, max_steps32, V/H disabled; paired multi-seed screens/gates and native/archived replay.
- `configs/alternating-skill-memory-v2-qwen3-7-plus-retail.yaml`: same new V2 treatment, V/H enabled, explicitly finite-panel V preservation; no population-risk certificate.

Both retain the pinned Retail train/test IDs and exclude 46/47. Agent/Customer/Evaluator/Activator use the existing InferAI Qwen3.7-plus configuration, thinking disabled; Evolver uses the existing InferAI DeepSeek V4 Pro thinking-enabled/high request. Provider availability and the actual reasoning behavior are **not live-verified by this implementation**. Use an independent output/checkpoint for every treatment. No old real directory is reused.

After installing `.[tau-bench,web,dev]`, the first real mechanism command is:

```bash
# TAU2_DATA_DIR must point to the pinned data; OPENAI_API_KEY is temporarily
# injected from the existing InferAI Keychain entry, never stored in a config.
evotau-evolve --config configs/alternating-skill-memory-v2-qwen3-7-plus-retail-mechanism-smoke.yaml
```

This documentation does not start that run. Respect the provider's actual dispatch-rate limits through the existing transport pacing launcher when deploying it; concurrency is not evidence of unlimited provider throughput. Do not substitute models or add silent retries.

First isolate the known repair morphology using a **frozen identical repair**, Customer, task/seed schedule and runtime/evaluator models: compare render-all versus activation. That avoids attributing a difference caused by a newly generated repair to the activator. Report whether the two local fixes survive and how many passing cases break; the requested target is ≥2 fixes and ≤1 break, not a demonstrated result. The search configuration starts empty, so a newly generated repair cannot be claimed to reproduce the historical 75%→65% transition without this controlled follow-up.

## Ablation matrix

All switches are frozen under `experiment.skill_evolution_v2`, except the corresponding runtime selector in `experiment.evolution`. Clone the config and change its ID/output/checkpoint; never edit a started run's manifest. Keep tasks, models/prompts, seeds, budgets and native semantics fixed wherever the ablation permits.

| Treatment | Change from Full V2 |
|---|---|
| Full V2 | All default mechanisms enabled |
| no_activation | Set both runtime selector fields to `render_all_v1`; no activator provider calls; budget covers all rendered skills |
| no_lineage | `service_evolution.lineage: false`; hide prior edit-effect feedback from proposer, retain audit records |
| single_candidate | `service_evolution.candidates_per_cluster: 1` |
| no_dedup | `service_evolution.semantic_dedup: false` |
| no_crossover | `service_evolution.crossover: false` |
| no_archive | `archive.enabled: false` (current-generation complementary candidates can still cross unless crossover also disabled) |
| single_seed | `evaluation.screen_seeds: [1]`, `gate_seeds: [1]`; formal statistical gate can correctly be inconclusive |
| no_statistical_gate | `statistical_gate.enabled: false`; strict positive paired delta and hard observed reliability checks retained |
| no_customer_replay | `opponent_replay.enabled: false`; current Customer only |
| no_history_summary | `history.summarize: false`; uncompressed effect metadata and current trajectories, no hidden scripts |

For the full search comparison, use matched independent seed replications; preserve every rejection and unknown attempt. Report native success/pass^k, helpfulness/harmfulness/benefit-risk, fixed/broken/persistent/recurring outcomes, stuck/steps/tools, observed protocol evidence, role usage/tokens/wall time, proposals/dedup/screen/gates/archive/crossover and V/H flags. More search looks change the correction and compute cost; this is part of the treatment, not hidden.

## Verification and saved offline evidence

```bash
TAU2_DATA_DIR=/path/to/pinned/data python -m pytest -q
python experiments/execution/run-v2-offline-smoke.py \
  --tau2-data-dir /path/to/pinned/data \
  --export /tmp/evotau-v2-native-evidence-new
# A complete exported project can be opened with:
evotau-web --project-root /tmp/evotau-v2-native-evidence-new
```

The implementation passed 179 offline tests and Ruff checks. Saved final engineering evidence is in `experiments/results/skill-evolution-v2-offline-native-20261007-final/`; an earlier diagnostic snapshot is preserved separately and explicitly documents a subsequently corrected Evolver accounting omission.

The exporter runs the pinned native Customer/Service/tools/backend/evaluator with a **scripted local completion boundary** and saves manifest, config, checkpoint, simulations, activation decisions, native rewards and stage/provider-call records. It exercises ADD, screen rejection, activation and exact resume with zero new local calls. It is E1/G1 with a deliberately tiny max_steps, not a useful task-performance experiment. The deterministic fake-runner suite separately exercises independent candidates, promotion, local losers, complementary crossover, replay and two-generation resume. Neither is evidence that a live LLM will repair the historical failures.

Tests cover catalog isolation/negative conditions/empty selection/K, prompt reset, structural edits/budgets, dedup, accepted/rejected memory separation, native fixed/broken attribution, unknowns/statistics/protected regression, archive/crossover, multiple interrupted stages and call recovery, fresh-challenge validation before H, identical S0/ST endpoint reuse, legacy Console compatibility, redaction and sealed H on Console surfaces.

## Design provenance and unverified risks

The following are the **design influences supplied in the task**, not claims of reproducing their implementations or published numerical results:

| Supplied influence | V2 design use |
|---|---|
| Beyond Prompts / PRISM | root-cause routing, passing controls, constrained edits and reliability risk |
| POLCA | stochastic seeds, retained candidates and bounded historical summaries |
| GEPA | per-instance effects, locally useful losers and complementary synthesis |
| SEPO | typed structural edits, newly fixed/broken lineage and prompt budgets |
| ESPO | Diagnose/Diversify/Stabilize and independent proposal biases |
| Reinforced Agent | helpfulness/harmfulness as repair diagnostics |
| SAGE statistical gate | regression-aware conservative uncertainty handling |
| EvolveMem | structured actions, reject regressions and explore stagnation |
| Contextual Experience Replay | retrieve relevant current evidence and summarize past effects |
| Darwin Gödel Machine | preserve stepping stones in an evolution-only archive |
| AgentSquare / AFlow | modular recombination, staged screening and branching candidates |
| Self-play / Reflective Experience Replay | archived adversaries and native clean replay |

Live activation applicability, semantic validators, diagnosis quality, mutation diversity and useful crossover remain empirical risks. The semantic duplicate check is a deterministic combined structure/family/text/delta heuristic, not a proof of semantic equivalence; a misleading delta claim can evade some near-duplicate detection. Multi-seed evaluation, replay and per-turn activation add compute and provider latency. Formal statistical power is limited by independent V tasks, and repeated seeds/provider determinism may not be independent draws. No live InferAI request or research-effectiveness claim is made by this implementation delivery.


## Gate calibration and controlled live measurement

`configs/skill-evolution-v2-activation-morphology-smoke.yaml` isolates the archived V1 repair's activation boundary. It freezes the previous Customer, trigger and guidance; Pro proposes only a narrower signature from observable E trajectories. The measurement runs empty memory, render-all, and activation against six diagnostic Retail train tasks (22, 80, 98, 4, 21, 35), each with seeds 1 and 2: **36 episodes**, P4, max_steps32, V/H disabled. Each treatment has its own manifest, cache, usage ledger and native artifacts. No new Customer search or multi-generation selection runs. This tests a known repair morphology; it does not prove the complete V2 search finds good skills.

```bash
PYTHONPATH=src python experiments/execution/run-v2-activation-morphology-smoke.py \
  --config configs/skill-evolution-v2-activation-morphology-smoke.yaml \
  --tau2-data-dir /path/to/pinned/data --dry-run
# Remove --dry-run only for the authorized live measurement.
```

The live launcher performs one no-tools JSON check per model, obtains the existing Chinese-group key from Keychain without persisting it, and shares the existing 30/60.1-second dispatch pacing across native and evolution calls. Total budget is unlimited. Provider errors remain recorded unknown attempts; there is no automatic model fallback or retry loop. Resume binds saved manifests and exact journal inputs and reuses completed native episodes/stages. Reporting distinguishes per-seed cell counts from the historic single-seed task counts (2 fixes/4 breaks); two seeds cannot be counted as two independent tasks.
