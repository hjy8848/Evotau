# EvoTau

EvoTau is a local research layer for policy-preserving customer–service co-evolution on a pinned tau-bench Retail text runtime. Phase 0 keeps tau-bench responsible for environment state, tools, simulation, and evaluation; EvoTau adds bounded prompt adapters, an immutable manifest, task eligibility checks, and a hard provider-request budget.

## Local setup

Run these commands from the EvoTau directory in PowerShell:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    python -m pytest -q -p no:cacheprovider
    python -m evotau.phase0 --config configs/mvp.yaml

The pinned upstream dependency is optional because the offline Phase 0 checks do not need tau-bench or model credentials. To install the audited upstream revision for integration work:

    python -m pip install -e ".[tau-bench]"

The upstream pin is sierra-research/tau2-bench commit b7ea9074c1cba482b30687fecdb5c8425fd6f619, package version 1.0.1.

## Phase 0

The Phase 0 manifest record is saved at experiments/manifests/evotau-phase0-retail-smoke.json. It selects Retail train task 73 for evolution and task 93 for validation, with heldout empty and tasks 46 and 47 excluded. It sets evaluation_type to all, max_steps to 64, one episode, concurrency one, zero provider retries, and a maximum of 70 provider attempts. The exact task and split file fingerprints are in configs/mvp.yaml.

By default, real_provider_enabled is false and model IDs are unset. The preflight command validates manifest invariants without calling a model. If exact upstream task and split files are available locally, pass both to the preflight command using --tasks-json and --split-json; their Git blob fingerprints are checked before eligibility validation.

The optional runtime entry points are in src/evotau/tau_adapter.py. run_phase0_episode refuses to run unless the manifest explicitly enables the real provider and freezes all role model IDs. It uses tau-bench run_simulation with EvaluationType.ALL, then invokes the native full conversation reviewer; evaluator and reviewer calls share the same hard budget. No provider call is made by the offline tests or preflight.

## Project boundaries

The research plan is maintained at research/EvoTau_Research_and_Engineering_Plan_v1.md. Phase 0 does not implement an evolver, failure attribution, service repair, or multi-generation experiments. It does not modify task truth, tool semantics, or evaluator behavior.
