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

The optional runtime entry points are in src/evotau/tau_adapter.py. `run_phase0_episode` and `evotau-phase0-run` refuse to run unless the manifest explicitly enables the real provider and freezes all role model IDs. The CLI verifies every manifest source fingerprint, task eligibility, and output-path uniqueness before dispatch. It uses tau-bench `run_simulation` with `EvaluationType.ALL`, then invokes the native full conversation reviewer; evaluator and reviewer calls share the same hard budget. Provider response caching and retries must be disabled. The runner saves the manifest, native simulation trajectory, review, actual completion-attempt and cache-hit counters, reported prompt/completion tokens, and hashes of the rendered agent/customer prompts. If token usage is absent, the result records that count as unavailable. If review fails after simulation, the trajectory and a budgeted incomplete-run record are retained. Offline tests and preflight make no real provider or network requests; the budget test uses an in-memory stub.

To execute the single live Phase 0 episode, first make a separate config copy with all three model IDs frozen, `real_provider_enabled: true`, and a new `output_path`. Then invoke `python -m evotau.phase0_run --config path/to/frozen-live-config.yaml --tau2-data-dir /path/to/pinned-tau2/data`. The checked-in MVP config intentionally remains provider-disabled.

The pinned native Retail environment and `Orchestrator` construction, including both prompt adapters, have been verified without provider calls. To repeat that integration check after installing the optional extra, set `EVOTAU_TAU2_DATA_DIR` to the `data` directory from the pinned checkout and run:

    EVOTAU_TAU2_DATA_DIR=/path/to/tau2-bench/data python -m pytest -q -k pinned_tau_runtime

This check confirms runtime construction and prompt composition only. It does not run `run_simulation`; the one-episode Phase 0 exit remains pending because the manifest still disables provider use and has no frozen model IDs.

## Phases 1–3 mechanism layer

The provider-agnostic research protocol is implemented in `src/evotau`: immutable episode/failure records, explicit independent failure promotion, deterministic one-axis Customer mutation, paired strict selection, an append-only SQLite archive of verified failures and strategy snapshots, structured policy-referenced Service repairs, target/history/clean/validation gates, atomic manifest-bound per-episode checkpoints, and a two-generation Customer-first controller. The controller accepts an `EpisodeRunner`; it does not create model clients or supply task truth. A caller must provide independent audit references before a failure contributes to fitness, and a Service strategy can change only when a returned `GateReport` accepts that exact candidate.

Offline mechanism fixtures cover no-change generations, invalid/unverified attribution, repair rejection, replay deduplication, and resume. These tests establish software behavior only; they do not establish a research result. `configs/phase3-mechanism.yaml` freezes the E/V tasks, upstream source fingerprints, strategy hashes, K=2, two generations, 23-episode/1,800-attempt ceilings, and provider-disabled default. The preflight validates task eligibility against the exact pinned task and split blobs when those files are supplied. Phase 0 native construction and prompt injection pass the optional pinned-runtime integration check; the full native episode and any real provider episode remain unrun. Pilot and formal phases remain future work as defined in the research plan.

## Project boundaries

The research plan is maintained at research/EvoTau_Research_and_Engineering_Plan_v1.md. EvoTau does not modify task truth, tool semantics, or evaluator behavior.
