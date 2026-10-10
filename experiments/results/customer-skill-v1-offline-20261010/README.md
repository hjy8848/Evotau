# Customer Skill v1 offline verification

No API requests or live episodes were started. No credentials were read.

- `verification.json`, `pytest.log`: final full-suite 417 pass, 6 dependency deprecation warnings; Ruff and diff checks pass. An initial existing native timeout-injection case failed once, then passed individually and in subsequent full suites. This non-reproduced test-stability issue is retained, not claimed fixed.
- `context-measurement.json`: actual archived Airline Gen0 E10 observations only, no V/H content. Three representative cases retain required user/action evidence. Initial diagnostic request is 30,135 cl100k proxy tokens, not actual provider tokens. Old scores were not imported into a new run.
- `legal-customer-skill.json`: structure-valid example with a supplied E message reference. Semantic validity and effect remain hypotheses requiring live review/rollouts.
- `illegal-customer-skill.json`: the same example with an explicit budget-changing step; rejected by deterministic validation.
- `g2-scripted-diagnostic.json`: two generations with artificial three-task outcomes, frozen Service and no extra callbacks on resume. This is control-flow evidence, not native benchmark/effectiveness data.
- `dry-validation.log`: prepared Airline E10 config passes dry validation without provider dispatch. No new experiment directory was created.

Protocol, limits, failure handling, source locations and future commands: `docs/research/task-faithful-customer-skill-v1.md`.
