# EvoTau: accuracy-driven alternating evolution

## Method and fixed inputs

```text
(C_t, S_t) → (C_(t+1), S_t) → (C_(t+1), S_(t+1))
```

Customer proposes task-grounded, reusable interaction strategies. Holding Service fixed, the incumbent and candidates are evaluated on the same E panel and `evolution_fitness_seed`. A candidate replaces the incumbent only if native E accuracy is strictly lower. Ties keep the incumbent. Missing native scores stop fitness computation rather than being scored as failure.

Holding the selected Customer fixed, Service Evolver studies observed E conversations, tool results, native outcomes and the unchanged Retail policy. It proposes either a PromptStrategy rewrite or one local SkillMemory ADD/UPDATE/NO_OP. SkillMemory starts empty, injects all active skills, retains stable skill IDs and provenance, and applies at most one mutation per generation. Rejected proposals do not become runtime skills.

Service is accepted only for strictly higher E accuracy under the same selected Customer. With `run_validation: true`, aggregate native-Customer V accuracy must also not decrease relative to the current S_t. This does not prohibit every individual pass→fail transition. With V disabled, acceptance is explicitly E-only mechanism smoke.

The benchmark's task, original scenario, policy, tools, backend, native LLMAgent/UserSimulator and evaluator remain unchanged. Evolution updates external behavior guidance, not model weights. There are four model roles: agent, customer, evaluator and evolver. There is no Reviewer, Customer Judge, Service Judge, failure taxonomy, TaskContract or behavioral blacklist in alternating selection. Native evaluation may invoke an LLM for NL assertions.

## Information boundary and panels

Customer Evolver can read its E source scenarios. Service Evolver cannot read hidden `user_scenario` or Customer Judge reasons: its context contains observed conversation/tool messages, native outcomes, policy, current strategies, numeric accuracy history and accepted Service evolution history. MultiToolMessage results are expanded into visible tool messages. Raw provider data and gold evaluator targets are omitted from projected evidence.

E and V come from Retail train, H from test, and panels/exclusions are disjoint and frozen. Optional H evaluation loads H content only after evolution and E-only fresh-Customer generation. The endpoint comparison crosses S₀/S_T with native/fresh Customers, reusing identical conditions. `run_heldout: false` prevents H task loading entirely.

## Execution map

```text
evotau-evolve / alternating_run.run_from_config
  ├─ frozen manifest + split/pin verification + invocation provenance
  ├─ TauBenchEpisodeRunner
  │    └─ native Orchestrator → sequential turns → native evaluator (reviewer disabled)
  └─ alternating.run_alternating_evolution
       ├─ incumbent E → Customer Evolver → candidate E → strict minimum accuracy
       ├─ Service Evolver → proposal E → strict accuracy improvement
       ├─ optional native-Customer V accuracy gate
       └─ generation artifact + checkpoint
            ↓ optional
       E-derived fresh Customer → load H → endpoint comparison
```

A bounded ThreadPoolExecutor runs tasks within a panel (and Customer candidate panels) while preserving input result order. It never overlaps Customer/Service phases or generations and never parallelizes the turns inside an episode. Provider accounting, reservations, per-episode usage and diagnostics use the existing shared budget with thread-local contexts. Successful identical task/seed/C/S conditions are reused across panels; a per-condition lock coalesces concurrent duplicates. Panel references preserve which generations evaluated each condition.

## Failure and reproducibility

The original manifest remains immutable. Source and selected-config fingerprints, model settings, benchmark pins and task selection govern resume compatibility. Original Git commit/dirty provenance is retained; every invocation records current Git provenance separately. Merely committing results or adding an unrelated config does not prevent unchanged-source resume. Changed runtime source or experiment conditions require a fresh run and cannot reuse historical scores.

Completed episodes require valid native scoring and saved native trajectories. A failed/unscored episode is not cached or admitted to selection. Per-attempt artifacts retain the original exception, partial conversation and sanitized provider response metadata. Evolver input artifacts retain exact research prompts/context. Execution state finishes as complete, failed or paused; resume does not repeat completed conditions. Failures do not trigger hidden retries, model substitution, fabricated dialogue termination or task omission.

For inspection, start with `run-execution-state.json` and `api-usage-live.json`, then the referenced `episodes/*/incomplete-run.json` or `evolver-calls/*/failure.json`. Complete decisions are in `generation-XXXX.json`; candidate/stage artifacts explain progress before a generation is committed. Native simulations retain benchmark outcomes, while Evolver contexts expose only permitted evidence. Console display supports current schema v2 and incomplete conversations.

## What the evidence can establish

Offline tests establish wiring, isolation, selection, concurrency, accounting, checkpoint/cache behavior, and outgoing request serialization. They do not establish live-provider reasoning compliance, model ability or successful evolution. The historical [2026-10-06 audit](audits/2026-10-06-idea-implementation-audit.md) describes real runs and their limitations; [the fix report](audits/2026-10-06-audit-fixes.md) maps its engineering findings to regressions.

Use separate measurements: Customer pressure compares accuracy before/after challenge with Service fixed; Service repair compares old/proposed Service under the same selected Customer. Native and fresh H are generalization evidence only when enabled. Fixed E-only search can overfit, aggregate V can conceal paired regressions, and fixed-seed API sampling may remain nondeterministic at the provider. Cache reuse is therefore one evaluation of a condition, not repeated-trial evidence.

Open text can still drift or invent facts; prompt grounding and information isolation do not prove semantic correctness. NO_OP and rejection remain legitimate research outcomes. Repairing transport or accounting does not establish that SkillMemory will learn useful skills. After the thinking-parameter fix, a new manifest/output and live parameter check are necessary before interpreting new provider results.
