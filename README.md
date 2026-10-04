# EvoTau

EvoTau is a τ-bench-based alternating self-evolution framework. An adaptive Customer searches for interaction strategies that expose weaknesses in the current Service; the Service then adapts from the resulting τ-bench trajectories. The benchmark tasks, scenarios, policy, backend, tools, native runtime, evaluator, and reviewer remain fixed.

## Research loop

```text
(C_t, S_t)
    freeze S_t
    run incumbent and candidate Customers on E through τ-bench
    compare their real trajectories and native outcomes → C_(t+1)
    freeze C_(t+1)
    evolve a natural-language Service strategy from its trajectories
    replay proposal on the same challenge; compare on a small native-Customer V panel
    → S_(t+1)
```

Customer is a task-grounded adaptive challenge generator. Its behavior may be cooperative-but-difficult or adversarial when the task itself creates a conflict. There is no predefined challenge taxonomy. Both Customer and Service strategies are free-form natural-language prompt overlays; no mutation operators, challenge counter, policy-rule schema, or generation-time behavioral blacklist defines the search space.

τ-bench owns the benchmark and execution semantics. EvoTau adds strategy prompting, the alternating loop, simple trajectory-based selection, split-aware loading, generation checkpoints, and endpoint comparison. Evolvers receive E task intent/scenario, visible policy where relevant, complete conversation/tool outcomes, native task result, native reviewer feedback, and prior generation feedback. They never receive reference actions, evaluation criteria, or H task content.

## Run

Install the project and its τ-bench integration:

```bash
python -m pip install -e '.[tau-bench,web]'
```

Set `TAU2_DATA_DIR` to the pinned τ-bench data directory. The checked-in config is intentionally provider-disabled:

```bash
cp configs/alternating-evolution.yaml configs/my-alternating-run.yaml
# Edit the copy: set a unique id/output/checkpoint path, freeze role models,
# and set real_provider_enabled: true.
evotau-evolve --config configs/my-alternating-run.yaml
```

The checked-in config itself is a template and the CLI refuses to run it while the provider is disabled. In the copy, freeze the five role models (`agent`, `customer`, `reviewer`, `evaluator`, `evolver`) and sampling settings, then set `real_provider_enabled: true`. `request_budget_cap: null` means no EvoTau request cap; every call is still counted. τ-bench provider retries and response caching are disabled by the adapter. The CLI does not edit a config or silently enable a provider.

The sample panel uses E task 73, V task 93, and held-out H task 5. E and V are loaded from τ-bench train; H is loaded only after evolution and fresh adaptive-Customer generation finish. The example runs two generations with two Customer proposals per generation. Adjust panels and generation count in a reviewed copy of the YAML.

For a single native τ-bench wiring check, `evotau-phase0` performs an offline split/pin preflight and `evotau-phase0-run` runs its one explicitly configured episode. Phase 0 is only a runtime check; it is not a second evolution method.

## What is saved

Each run directory contains a frozen manifest and run context, one JSON record and native τ-bench simulation per episode, per-episode usage telemetry, a generation artifact, an atomic resume checkpoint, a fresh adaptive-Customer proposal, a final endpoint comparison, and `alternating-result.json`. The endpoint comparison crosses S₀/S_T with native τ-bench Customer and a fresh Customer generated from E-only evidence, then evaluates those conditions on H.

An accepted Service update requires two simple observations: the proposal is judged better on the same evolved-Customer challenge, and it does not turn a previously successful case in a small native-Customer V panel into a failure. Each generation compares against its current S_t. A candidate can be rejected, unchanged, or unsuccessful; all episodes remain saved and no positive evolution is required for a completed software run.

## Local Console

```bash
evotau-web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The Console previews the frozen config, launches the same CLI runner, and displays saved generations and conversation/tool traces. It does not perform evolution or recalculate τ-bench scores.

## Validation status

The unit tests exercise freeze order, trajectory context, selection, multi-generation continuity, prompt overlays, task-split isolation, checkpointing, endpoint comparisons, held-out sealing, and episode display. A deterministic offline smoke ran one alternating generation and 8 episodes through the installed pinned τ-bench runtime, including a real Retail tool execution and native scoring/review. It uses `max_steps: 2`; only provider HTTP completions were stubbed, so it verifies wiring rather than task completion, model capability, or positive evolution. No live-provider result is implied by that smoke.

See [the current research and engineering note](research/Alternating_EvoTau.md) for the execution map, information boundary, what was verified, and open risks.
