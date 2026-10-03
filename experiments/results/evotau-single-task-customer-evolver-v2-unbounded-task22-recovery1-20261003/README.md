# EvoTau one-task Customer Evolver V2 diagnostic

This is a single-task diagnostic, not a multi-task effectiveness result. It used reviewed Retail task `22` as the only executed E task, seed `1`, one Customer V2 generation, and two provider-generated candidates. V93 and H were not executed. The recorded configuration has `request_budget_cap: null`, omits `max_tokens` for every role, and disables model thinking.

## Outcome

The incumbent and candidate 1 both received native τ-bench reward `1.0`. Candidate 2 received `0.0`. However, EvoTau Core's frozen Customer fitness is the count of distinct tasks with an independently verified, reproducible Service policy failure; it does not maximize native reward. No episode yielded a failure in the frozen Service-failure taxonomy, so all three fitness scores were `0`. Core retained the incumbent with the decision `no strict discovery improvement; incumbent retained`. No Customer strategy was promoted, and no fresh confirmation rollout was needed.

Candidate 2 changed `request_order` and `request_decomposition`. The τ-bench native reviewer flagged the Customer's `US` country value as unsupported by the scenario instructions; the independent EvoTau auditor nevertheless marked the Customer valid and strategy-adherent. The DB trace shows the resulting value mismatch (`US` versus expected `USA`), explaining the zero native reward. This is an audit disagreement; the current result preserves both judgments rather than silently overriding either one. Candidate 1 changed only `disclosure` and achieved reward `1.0`; τ-bench's reviewer noted procedural multi-tool-call issues, but those are outside the frozen three-signature Service failure taxonomy and therefore did not count as EvoTau fitness.

## Provider usage

The complete recovery run accounts for 80 provider attempts: 79 succeeded and one failed with a CloudFront 504 during the original independent audit. The saved native trajectory was reused; only its audit was retried before running the two candidate episodes. Usage totals were 366,342 prompt tokens and 22,813 completion tokens. LiteLLM has no price mapping for `deepseek-v4-flash`, so billed cost is unknown. The API key was injected from macOS Keychain and is not stored in this archive.

## Artifacts

- `single-task-evolution-result.json`: full Core generation decision and episode summaries.
- `analysis-summary.json`: compact candidate comparison and attribution notes.
- `manifest.json`, `run-config.yaml`, and `run-context.json`: pinned environment, no-cap provider settings, panel, and provenance.
- `episodes/`: native trajectories, independent audits, τ-bench DB-state traces, EpisodeRecords, and usage telemetry. `episodes/recovered-baseline/` links the successfully completed baseline trajectory to its retried independent audit.
- `recovery-source-incomplete.json`: preserved details of the original audit timeout.
- `runbook/`: exact one-off orchestration scripts used for this run and the audit recovery. Re-running them requires a new output path and Keychain-injected `OPENAI_API_KEY`.
- `SHA256SUMS`: checksums for every artifact other than itself.

The first attempt, including its original 504 and the completed native trajectory, remains preserved at `experiments/results/evotau-single-task-customer-evolver-v2-unbounded-task22-20261003`.
