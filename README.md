# EvoTau

EvoTau studies alternating Customer challenge and Service repair on a fixed τ-bench substrate. Tasks, scenarios, policy, backend, tools, native Customer/Service runtime and evaluator stay fixed. Evolution changes external prompt strategies or Service SkillMemory; it does not update model weights.

## Research loop

```text
(C_t, S_t)
    freeze S_t
    evaluate incumbent and candidate Customers on fixed E tasks + fitness seed
    strictly lower native E accuracy wins; ties keep the incumbent
    freeze the selected Customer
    propose Service repair from observed conversation/tool results + policy + native outcomes
    strictly higher E accuracy is required
    if V is enabled: native-Customer V accuracy must also not decrease
    commit (C_(t+1), S_(t+1)) and checkpoint
```

Skill Evolution V2 is an independent `skill_memory_v2` carrier with per-turn observable activation, typed edits, direct evidence-backed Skill mutation, independent candidates, paired screening/statistical gates, rejection lineage, research archives/crossover and Customer replay. Its configs, schemas, compatibility rules, risks and ablation matrix are documented in [Skill Evolution V2](docs/skill-evolution-v2.md). The research loop below describes the legacy accuracy-driven carriers; formal V2 uses the stricter paired V promotion gate.

The [2026-10-08 pre-formal readiness audit](research/audits/2026-10-08-v2-readiness-audit.md) separates engineering tests, representative real InferAI requests and unverified research effects. Full E/V/H launch remains blocked; the E20/V16/H40 pilot config is disabled pending provider readiness and explicit budget/launch approval.

Customer strategies are reusable natural-language interaction skills grounded in each task's original objective and facts. Legacy configurations support the original `prompt_strategy` carrier and `skill_memory_v1`. V1 SkillMemory starts empty, permits one ADD/UPDATE/NO_OP per generation, and injects all active skills into the native Service prompt. No retrieval or selector is used.

The opt-in [Task-faithful Customer Skill v1](docs/research/task-faithful-customer-skill-v1.md) adds open-ended procedural Customer skills, success-prioritized E evidence, static and actual-trajectory legality checks, and bounded rejection feedback. Only wholly valid candidates with strictly lower native E accuracy can win. The independent Airline Customer-only diagnostic config is prepared for dry validation; no live effectiveness is claimed.

Customer Evolver receives its source `user_scenario` and E trajectories. Service Evolver receives observed conversations, tool results, policy and native outcomes; it receives neither the hidden scenario nor Customer Judge free text. Neither Evolver receives reference actions, evaluator targets, or H content. Reviewer, Customer Judge and Service Judge calls are **zero** in the active alternating runtime. The native evaluator can still make LLM calls for task NL assertions.

E/V use the official Retail train split; H uses test. `run_validation: false` selects E-only mechanism-smoke acceptance. `run_heldout: false` prevents H loading and endpoint evaluation. When H is enabled, an E-only fresh Customer is generated before H content is loaded. S₀ and S_T are compared under native and fresh adaptive Customers. An unchanged Service reuses identical endpoint episodes.

## Run

```bash
python -m pip install -e '.[tau-bench,web]'
cp configs/alternating-evolution.yaml configs/my-alternating-run.yaml
# Freeze models, args, unique id/output/checkpoint and explicitly enable the provider.
# Set TAU2_DATA_DIR to the pinned τ-bench data directory.
evotau-evolve --config configs/my-alternating-run.yaml
```

The generic template is provider-disabled; archived real configurations are independent experiment specifications. Freeze four roles: `agent`, `customer`, `evaluator`, `evolver`. `request_budget_cap: null` removes the EvoTau request cap while retaining accounting. The adapter uses no transport retries or provider response caching. Episode concurrency is bounded by `max_parallel_episodes` (1–8); turns, phases and generations remain sequential.

Runtime roles and Chat Completions Evolvers use the existing τ-bench/LiteLLM transport against the configured provider, including InferAI. An Evolver configured with `api_protocol: responses` uses the direct InferAI Responses client. Credentials are resolved separately by role/config; they are not experiment artifacts. `thinking_mode` is converted using the common runtime converter before dispatch, and configured `reasoning_effort` is preserved. Request metadata records the actual arguments; provider compliance still requires live verification.

`evotau-phase0` validates pins/splits offline; `evotau-phase0-run` performs a configured single-episode wiring check. Phase 0 does not run evolution.

## Artifacts and resume

A run saves its frozen manifest, selected-config fingerprint, run context, completed native simulations/records, generation stages/proposals, atomic checkpoint, API usage, and final result. Optional H evaluation saves the fresh-Customer proposal and endpoint comparisons. Evolver calls save their exact prompt/context plus parsed response or failure.

Episode cache identity is task + seed + Customer strategy + Service carrier. The run's manifest freezes source, models, arguments, benchmark pins and selected config. Identical conditions reuse completed scores across panels/generations and concurrent duplicates dispatch once. `episode-panel-references.json` records each panel's reference without inflating completed episode counts. Cache hits require no provider reservation. This is fixed-seed evaluation reuse, not independent resampling.

Resume preserves the original manifest and validates all actual execution inputs. Git commit/dirty changes from archiving results are recorded per invocation without invalidating unchanged source. Unrelated config files do not affect source identity. Changes to source, selected config, models or benchmark inputs still require a new run. In particular, scores from before the runtime thinking-parameter fix cannot be reused as corrected-condition scores.

Missing or invalid native scores remain incomplete and cannot enter accuracy or selection. Failures save a partial conversation, sanitized per-call response metadata (ID, finish reason, visible/reasoning/tool counts, tokens, latency and error), and explicit failed/paused execution state. Logs exclude credentials, prompts and reasoning text; Evolver input artifacts separately contain the research context. Resume reuses successes and reruns incomplete attempts only. No fake STOP, silent skip or automatic empty-response retry is introduced.

`total_wall_clock_seconds` sums recorded invocation durations; `elapsed_since_first_start_seconds` also includes time between invocations. Per-role API latency is summed request time, which can exceed wall-clock time with concurrency.

## Local Console and verification

```bash
evotau-web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The Console reads saved Phase 0 and alternating artifacts, including schema v2 results, failed partial conversations and reused panel references. It does not recalculate scores. H artifacts remain sealed until completion.

```bash
python -m pytest -q
ruff check src tests
```

Tests cover selection/freeze order, information boundaries, SkillMemory, concurrency, budget reservations, resume compatibility, cache reuse, failure diagnostics and Console display. The runtime-parameter regression test intercepts actual τ `generate` → LiteLLM → OpenAI SDK HTTP serialization locally; it does not make a live request or establish provider/model capability.

See [the current method note](research/Alternating_EvoTau.md), [the historical audit](research/audits/2026-10-06-idea-implementation-audit.md), and [the audit fixes](research/audits/2026-10-06-audit-fixes.md).
