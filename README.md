# EvoTau

EvoTau is a local research layer for policy-preserving customer–service co-evolution on a pinned tau-bench Retail text runtime. τ-bench remains responsible for environment state, tools, simulation, and evaluation; EvoTau adds bounded strategy search, failure attribution, repair gates, replay archive, and manifest-bound checkpoints.

## Local setup

Run these commands from the EvoTau directory on macOS/Linux:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    python -m pytest -q -p no:cacheprovider
    python -m evotau.phase0 --config configs/mvp.yaml
    python -m evotau.mechanism_preflight --config configs/phase3-mechanism.yaml

The pinned upstream dependency is optional because the offline protocol tests do not need tau-bench or model credentials. To install the audited upstream revision for integration work:

    python -m pip install -e ".[tau-bench]"

The upstream pin is sierra-research/tau2-bench commit b7ea9074c1cba482b30687fecdb5c8425fd6f619, package version 1.0.1.

## Phase 0

The Phase 0 manifest record is saved at experiments/manifests/evotau-phase0-retail-smoke.json. It selects Retail train task 73 for evolution and task 93 for validation, with heldout empty and tasks 46 and 47 excluded. It sets evaluation_type to all, max_steps to 64, one episode, concurrency one, zero provider retries, and a maximum of 70 provider attempts. The manifest records the EvoTau Git commit, whether the worktree was clean, an EvoTau source snapshot hash, and the pinned τ-bench source fingerprints. The exact task and split file fingerprints are in configs/mvp.yaml.

By default, real_provider_enabled is false and model IDs are unset. The preflight command validates manifest invariants without calling a model. If exact upstream task and split files are available locally, pass both to the preflight command using --tasks-json and --split-json; their Git blob fingerprints are checked before eligibility validation.

The optional runtime entry points are in `src/evotau/tau_adapter.py`. `run_phase0_episode` and `evotau-phase0-run` refuse to run unless the manifest explicitly enables the real provider and freezes all role model IDs. The CLI verifies every manifest source fingerprint, task eligibility, and output-path uniqueness before dispatch. It uses τ-bench `run_simulation` with `EvaluationType.ALL`, then invokes the native full conversation reviewer; evaluator and reviewer calls share the same hard budget. Provider response caching and retries must be disabled. The runner saves the manifest, native simulation trajectory, review, actual completion-attempt and cache-hit counters, reported prompt/completion tokens, and hashes of the rendered agent/customer prompts. If token usage is absent, the result records that count as unavailable. If review fails after simulation, the trajectory and a budgeted incomplete-run record are retained. Offline tests and preflight make no real provider requests; installation is needed only for the optional upstream integration test.

To execute the single live Phase 0 episode, first make a separate config copy with all three model IDs frozen, `real_provider_enabled: true`, and a new `output_path`. Then invoke `python -m evotau.phase0_run --config path/to/frozen-live-config.yaml --tau2-data-dir /path/to/pinned-tau2/data`. The checked-in MVP config intentionally remains provider-disabled.

The pinned native Retail environment and `Orchestrator` construction, including both prompt adapters, have been verified without provider calls. To repeat that integration check after installing the optional extra, set `EVOTAU_TAU2_DATA_DIR` to the `data` directory from the pinned checkout and run:

    EVOTAU_TAU2_DATA_DIR=/path/to/tau2-bench/data python -m pytest -q -k pinned_tau_runtime

This check verifies all 19 pinned τ-bench source fingerprints, loads the real Retail task, constructs the native environment and `Orchestrator`, and checks both prompt adapters without provider calls. It does not run `run_simulation`; the one-episode Phase 0 exit remains pending because the manifest still disables provider use and has no frozen model IDs.

## Phases 1–3 mechanism layer

The provider-agnostic research protocol is implemented in `src/evotau`: immutable episode/failure records, explicit independent failure promotion, deterministic one-axis Customer mutation, paired strict selection, an append-only SQLite archive of verified failures and strategy snapshots, structured policy-referenced Service repairs, target/history/clean/validation gates, atomic manifest-bound per-episode checkpoints, and a two-generation Customer-first controller. The controller accepts an `EpisodeRunner`; it does not create model clients or supply task truth. A caller must provide independent audit references before a failure contributes to fitness, and a Service strategy can change only when a returned `GateReport` accepts that exact candidate. Phase 3 also freezes three MVP failure classes and their exact Retail policy references—missing identity verification, missing explicit confirmation before a write, and incomplete write scope—into its hashed manifest; out-of-scope signatures or policy references cannot enter fitness or the replay archive.

Independent episode audits separately record factual Customer validity, whether an EvoTau behavior had an opportunity to apply, and adherence when applicable. `None` can run the native τ-bench Customer without a strategy overlay; Service clean-gate pairs require that mode, while adversarial target/history/validation pairs must have an applicable, adherent Customer strategy. Cross-play reports valid, invalid, infrastructure, uncertain, not-applicable, and strategy-adherence counts separately; attribution rates use applicable, adherent episodes as their denominator.

Customer mutation records now include exact changed fields, the expected behavioral effect, and the failure IDs that support a failure-conditioned proposal. `FailureArchive.active_representatives()` caps replay representatives at 32, orders them deterministically by recent recurrence, severity, replay coverage, and recency, and retains the full append-only history. `mark_replayed()` and `active_replay_coverage()` report representative coverage; each generation checkpoint stores the resulting coverage snapshot with its decision. `evotau.calibration.calibrate_attribution()` reports whether independent double review of at least 30 distinct candidate positives meets the plan's 90% conservative precision and 10% critical-fact dispute thresholds. It returns `insufficient_evidence` or `pause_automation` when the corresponding condition is not met; it does not generate human reviews.

After collecting two independent human judgments per proposed failure, save them in the strict version 1 JSON format (`schema_version` and `cases`; each case contains distinct anonymized reviewer IDs, reviewer refs/verdicts, and `critical_fact_dispute`) and run `evotau-calibrate-attribution --input path/to/reviews.json`. The report includes a SHA-256 of the exact input artifact.

Offline mechanism fixtures cover no-change generations, invalid/unverified attribution, repair rejection, replay deduplication, episode and decision recovery, manifest-bound strategy IDs, and fresh-seed Customer confirmation. Each atomic generation checkpoint now retains the candidate evaluations, confirmation panel, selection decision, verified failures, Service gate outcome, and idempotent archive writes. A prepared-but-uncommitted generation is resumed from that saved decision, so it does not ask the ServiceTransition to regenerate a decision after a crash. `build_crossplay_matrix` summarizes balanced frozen Customer × Service runs, keeps invalid/uncertain/infrastructure episodes visible, and counts only independently verified failures. These checks establish software behavior only; they do not establish a research result. `configs/phase3-mechanism.yaml` freezes the E/V tasks, upstream source fingerprints, strategy hashes, K=2, two generations, 23-episode/1,800-attempt ceilings, and provider-disabled default. The preflight validates task eligibility against the exact pinned task and split blobs when those files are supplied.

For a future multi-task pilot, `evotau.eligibility.validate_generalization_selection()` checks train-only E/V, official-test H, ex-ante task eligibility, and disjoint extracted customer/order/entity identifiers. It rejects an unsafe requested partition rather than silently filtering tasks. Its ID-based screen does not prove policy/tool compatibility or task satisfiability; those require a recorded semantic task review before a run.

`evotau.native_runner.run_native_phase3` now connects the controller to the pinned τ-bench `run_simulation` path without forking its runtime. It requires a complete Phase 0 result and adjacent manifest/simulation whose upstream pin, E task, source fingerprints, role models, seed, and budget record match the Phase 3 manifest. It records the Phase 0 result fingerprint into an immutable `run-context.json` and the checkpoint hash, and carries Phase 0's provider attempts into the shared 1,800-attempt ceiling. Phase 0's one episode also counts toward the 23-episode limit. Live use needs caller-supplied independent episode audit and gated Service-transition callbacks; the adapter cannot infer those judgments from τ-bench's native reviewer alone. This is currently a Python API, not a standalone Phase 3 CLI. The Phase 0 native construction/prompt integration check passes; no live `run_simulation`, full native Phase 3 loop, 10–25 episode smoke, Pilot, or Formal has run.

## Project boundaries

The research plan is maintained at research/EvoTau_Research_and_Engineering_Plan_v1.md. EvoTau does not modify task truth, tool semantics, or evaluator behavior.
