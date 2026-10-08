# Direct Skill Evolution delivery — offline only

Baseline: `dd67c9b`; branch `codex/alternating-evolution-refactor`.
New research mechanism: `direct_skill_evolution_v1`.

Completed: independent Diagnoser removal; canonical evidence-backed direct mutations;
fixed three independent proposal biases plus at most one conditional crossover;
strict schema/case/reference validation; compressed evidence with explicit input limits;
fail-fast V rejection with NOT_EVALUATED gaps and unchanged maximum-look correction;
historical Console reading; new independent config with no cache imports.

Validation:

- Complete offline pytest suite: 281 passed, 5 upstream deprecation warnings.
- Ruff across `src` and `tests`: passed.
- `git diff --check`: passed.
- Real pinned data dry validation: E20/V3 loaded, H content not loaded; real_requests_started=false.
- Deterministic E20/G2 control-flow tests: native/fresh H endpoint handling,
  Gen1/H interruptions and exact resume; scores are scripted, not research measurements.
- Native τ-bench integration tests: actual native actors/tools/backend/evaluator,
  scripted local completion boundary; malformed JSON, timeouts, budget and native faults.
- Historical archived Diagnoser Console rendered successfully without provider dispatch.

`context-measurement.json` records an offline reconstruction from saved original E20
native simulations, source digests, all 20 result cells and four detailed raw contrasts.
Generation-zero direct Prompt+context is 27,022 cl100k_base proxy tokens.
This is a context measurement only: zero episode/cache/fitness imports into a new run.

Live provider verification in this refactor: **none**. New API requests: **zero**.
No real experiment was started and no H tasks/results were used for evolution.

Research effectiveness is unverified. Provider availability/latency, useful proposals,
semantic validation quality, later-generation context growth and empirical skill
activation remain risks. An exceeded context limit stops visibly; finite V3/H5
acceptance does not certify population generalization. Fail-fast leaves a deliberately
incomplete rejected-candidate risk profile; full_audit is available.

See `docs/direct-skill-evolution.md` for the contract, frozen configuration, migration,
future authorization-dependent launch, pause/resume and compatibility-audit requirements.
