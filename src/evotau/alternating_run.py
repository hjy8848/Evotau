"""CLI entry point for a native τ-bench alternating evolution run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from .alternating import (
    EpisodeJobTelemetry,
    LLMAlternatingEvolvers,
    _write_json_atomic,
    propose_fresh_customer_challenge,
    run_alternating_evolution,
    run_final_endpoint_evaluation,
)
from .alternating_manifest import AlternatingManifest
from .budget import RequestBudget
from .episode_execution import StopBeforeEpisodeDispatch
from .phase0 import load_config
from .phase0_run import _write_json_once
from .strategies import PromptStrategy
from .tau_adapter import verify_tau2_installation
from .tau_episode_runner import TauBenchEpisodeRunner
from .tau_provenance import (
    role_model_args_for_runtime,
    sha256_json,
    verify_git_blob_sha1,
    write_manifest_once,
)


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path | None = None,
    stop_before_next_episode_file: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_started = perf_counter()
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    manifest = AlternatingManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError("provider calls are disabled in this alternating-run config")
    data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR to the pinned τ-bench data")

    project_root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    output_directory = project_root / manifest.output_path
    checkpoint_path = project_root / manifest.checkpoint_path
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest.to_document():
            raise ValueError("existing run directory belongs to a different frozen manifest")
    elif any(output_directory.iterdir()):
        raise FileExistsError("refusing to bind an existing run directory without its manifest")
    else:
        write_manifest_once(manifest_path, manifest)
    execution_state_path, execution_state = _begin_execution_attempt(
        output_directory, manifest.sha256,
    )
    budget = RequestBudget(manifest.request_budget_cap)
    budget.enable_live_usage(output_directory / "api-usage-live.json")
    evolution_tasks = load_alternating_tasks(
        manifest,
        data_dir,
        include_validation=manifest.run_validation,
        include_heldout=False,
    )
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_dir,
        request_budget=budget,
        output_directory=output_directory,
        stop_before_next_episode_file=stop_before_next_episode_file,
        task_objects=evolution_tasks,
    )
    expected_evolution_tasks = set(manifest.evolution_task_ids)
    if manifest.run_validation:
        expected_evolution_tasks.update(manifest.validation_task_ids)
    if set(runner.tasks) != expected_evolution_tasks:
        raise ValueError("evolution runner loaded a task outside the enabled E/V panels")

    models = dict(manifest.role_models)
    role_args = role_model_args_for_runtime(manifest.role_model_args)
    providers = LLMAlternatingEvolvers(
        model=models["evolver"],
        model_args=role_args["evolver"],
    )
    run_context = {
        "schema_version": 2,
        "manifest_sha256": manifest.sha256,
        "config_sha256": sha256_json(config),
        "provider_models": {
            "customer_evolver": models["evolver"],
            "service_evolver": models["evolver"],
            "task_evaluator": models["evaluator"],
        },
        "task_panels": {
            "E": list(manifest.evolution_task_ids),
            "V": list(manifest.validation_task_ids),
            "H": list(manifest.heldout_task_ids),
        },
        "heldout_policy": (
            "H task content is loaded only after evolution and fresh challenge generation."
            if manifest.run_heldout
            else "H task content is not loaded in this mechanism-smoke run."
        ),
        "reviewer_enabled": False,
        "max_parallel_episodes": manifest.max_parallel_episodes,
        "evolution_fitness_seed": (
            manifest.seed if manifest.evolution_fitness_seed is None
            else manifest.evolution_fitness_seed
        ),
        "run_validation": manifest.run_validation,
        "run_heldout": manifest.run_heldout,
    }
    _write_or_verify_context(output_directory / "run-context.json", run_context)

    initial_customer = PromptStrategy(manifest.initial_customer_strategy)
    initial_service = PromptStrategy(manifest.initial_service_strategy)
    episode_job_telemetry = EpisodeJobTelemetry(manifest.max_parallel_episodes)
    evolution_started = perf_counter()
    evolved = run_alternating_evolution(
        tasks=runner.tasks,
        evolution_task_ids=manifest.evolution_task_ids,
        validation_task_ids=manifest.validation_task_ids,
        seed=manifest.seed,
        generations=manifest.generations,
        customer_candidate_count=manifest.customer_candidates,
        clean_panel_size=manifest.clean_panel_size,
        max_parallel_episodes=manifest.max_parallel_episodes,
        evolution_fitness_seed=manifest.evolution_fitness_seed,
        run_validation=manifest.run_validation,
        episode_job_telemetry=episode_job_telemetry,
        initial_customer=initial_customer,
        initial_service=initial_service,
        runner=runner,
        customer_evolver=providers.customer_candidates,
        service_evolver=providers.service_candidate,
        domain_policy=runner.service_policy_text,
        request_budget=budget,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        manifest_sha256=manifest.sha256,
    )

    # The fresh challenge is generated from E-only evidence before any H task
    # object or H trajectory is loaded into this process.
    fresh_customer = None
    if manifest.run_heldout:
        fresh_customer = propose_fresh_customer_challenge(
            evolved,
            providers.customer_candidates,
            tasks=runner.tasks,
            runner=runner,
            domain_policy=runner.service_policy_text,
            request_budget=budget,
            proposal_path=output_directory / "fresh-customer-proposal.json",
            manifest_sha256=manifest.sha256,
        )
    evolution_wall_clock_seconds = perf_counter() - evolution_started
    validation_wall_clock_seconds = sum(
        float(item.get("timing", {}).get("validation_wall_clock_seconds", 0.0))
        for item in evolved.generations
    )
    heldout_evaluation = None
    heldout_wall_clock_seconds = 0.0
    if manifest.run_heldout:
        heldout_started = perf_counter()
        heldout_tasks = load_alternating_tasks(
            manifest,
            data_dir,
            include_validation=manifest.run_validation,
            include_heldout=True,
        )
        heldout_runner = TauBenchEpisodeRunner(
            manifest=manifest,
            config=config,
            data_dir=data_dir,
            request_budget=budget,
            output_directory=output_directory,
            include_heldout=True,
            stop_before_next_episode_file=stop_before_next_episode_file,
            task_objects=heldout_tasks,
        )
        heldout_tasks = {task_id: heldout_tasks[task_id] for task_id in manifest.heldout_task_ids}
        if fresh_customer is None:
            raise RuntimeError("held-out evaluation requires a fresh adaptive Customer")
        heldout_evaluation = run_final_endpoint_evaluation(
            heldout_tasks=heldout_tasks,
            heldout_task_ids=manifest.heldout_task_ids,
            seed=manifest.seed + manifest.generations + 50_000,
            initial_service=initial_service,
            final_service=evolved.service,
            fresh_customer=fresh_customer,
            runner=heldout_runner,
            max_parallel_episodes=manifest.max_parallel_episodes,
            telemetry=episode_job_telemetry,
            output_path=output_directory / "heldout-endpoint-evaluation.json",
        )
        heldout_wall_clock_seconds = perf_counter() - heldout_started
    api_usage_by_call_name = budget.api_usage_by_call_name()
    execution_state = _finish_execution_attempt(execution_state_path)
    completed_episodes = len(tuple((output_directory / "episodes").glob("*/episode-record.json")))
    failed_episode_attempts = len(
        tuple((output_directory / "episodes").glob("*/incomplete-run.json")),
    )
    final_result = {
        "schema_version": 2,
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "run_context_sha256": sha256_json(run_context),
        "initial_customer": initial_customer.to_dict(),
        "initial_service": initial_service.to_dict(),
        "final_customer": evolved.customer.to_dict(),
        "final_service": evolved.service.to_dict(),
        "generations": list(evolved.generations),
        "fresh_adaptive_customer": (
            None if fresh_customer is None else fresh_customer.to_dict()
        ),
        "heldout_endpoint_evaluation": heldout_evaluation,
        "evolution_fitness_seed": (
            manifest.seed if manifest.evolution_fitness_seed is None
            else manifest.evolution_fitness_seed
        ),
        "validation_enabled": manifest.run_validation,
        "validation_evaluated": any(
            generation.get("service_phase", {}).get("validation_evaluated", False)
            for generation in evolved.generations
        ),
        "heldout_evaluated": manifest.run_heldout,
        "episode_job_telemetry": episode_job_telemetry.snapshot(),
        "provider_usage": budget.snapshot().to_dict(),
        "api_usage_by_call_name": api_usage_by_call_name,
        "api_usage_by_role": _api_usage_by_role(api_usage_by_call_name),
        "resume_count": execution_state["resume_count"],
        "completed_episodes": completed_episodes,
        "failed_episode_attempts": failed_episode_attempts,
        "timing": {
            "evolution_wall_clock_seconds": round(evolution_wall_clock_seconds, 6),
            "validation_wall_clock_seconds": round(validation_wall_clock_seconds, 6),
            "heldout_wall_clock_seconds": round(heldout_wall_clock_seconds, 6),
            "total_wall_clock_seconds": execution_state["total_wall_clock_seconds"],
            "current_process_wall_clock_seconds": round(perf_counter() - run_started, 6),
            **episode_job_telemetry.snapshot(),
        },
    }
    _write_json_atomic(output_directory / "alternating-result.json", final_result)
    return output_directory, final_result


def _begin_execution_attempt(
    output_directory: Path, manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = output_directory / "run-execution-state.json"
    now = datetime.now(UTC)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("manifest_sha256") != manifest_sha256:
            raise ValueError("run execution state belongs to a different frozen manifest")
        invocation_count = int(state.get("invocation_count", 0)) + 1
        first_started_at = state.get("first_started_at")
    else:
        invocation_count = 1
        first_started_at = now.isoformat()
    state = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "first_started_at": first_started_at,
        "last_started_at": now.isoformat(),
        "invocation_count": invocation_count,
        "resume_count": max(0, invocation_count - 1),
        "status": "running",
    }
    _write_json_atomic(path, state)
    return path, state


def _finish_execution_attempt(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(state["first_started_at"])
    completed = datetime.now(UTC)
    state.update({
        "last_completed_at": completed.isoformat(),
        "status": "complete",
        "total_wall_clock_seconds": round((completed - started).total_seconds(), 6),
    })
    _write_json_atomic(path, state)
    return state


def _api_usage_by_role(
    usage_by_call_name: Mapping[str, Mapping[str, int | float]],
) -> dict[str, dict[str, Any]]:
    call_names_by_role = {
        "customer": ("user_simulator_response",),
        "service": ("agent_response",),
        "evaluator": ("nl_assertions_eval",),
        "customer_evolver": ("evotau_customer_evolver",),
        "service_evolver": ("evotau_service_evolver",),
        "reviewer": (
            "llm_judge_review", "llm_judge_streaming_review",
            "classify_authentication", "llm_judge_hallucination_check",
        ),
        "customer_judge": ("evotau_customer_selection",),
        "service_judge": ("evotau_service_selection",),
    }
    roles = {}
    assigned = set()
    for role, call_names in call_names_by_role.items():
        selected = {
            name: dict(usage_by_call_name[name])
            for name in call_names if name in usage_by_call_name
        }
        assigned.update(selected)
        roles[role] = _aggregate_call_usage(selected)
    other = {
        name: dict(usage)
        for name, usage in usage_by_call_name.items() if name not in assigned
    }
    roles["other"] = _aggregate_call_usage(other)
    for role, call_names in call_names_by_role.items():
        roles[role]["by_call_name"] = {
            name: dict(usage_by_call_name[name])
            for name in call_names if name in usage_by_call_name
        }
    roles["other"]["by_call_name"] = other
    return roles


def _aggregate_call_usage(
    calls: Mapping[str, Mapping[str, int | float]],
) -> dict[str, int | float]:
    integer_fields = (
        "calls", "successes", "failures", "prompt_tokens", "completion_tokens",
        "usage_responses", "usage_unavailable",
    )
    totals: dict[str, int | float] = {
        field: sum(int(item.get(field, 0)) for item in calls.values())
        for field in integer_fields
    }
    elapsed = sum(float(item.get("total_elapsed_seconds", 0.0)) for item in calls.values())
    totals["total_elapsed_seconds"] = round(elapsed, 6)
    totals["average_elapsed_seconds"] = round(
        elapsed / max(int(totals["calls"]), 1), 6,
    )
    return totals


def load_alternating_tasks(
    manifest: AlternatingManifest,
    data_dir: str | Path,
    *,
    include_validation: bool = True,
    include_heldout: bool,
) -> dict[str, Any]:
    """Load only the requested panel; before final evaluation, H contributes IDs only."""

    if type(include_validation) is not bool or type(include_heldout) is not bool:
        raise ValueError("task panel inclusion flags must be booleans")
    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"τ-bench data directory does not exist: {data_root}")
    os.environ["TAU2_DATA_DIR"] = str(data_root)
    verify_tau2_installation()
    import tau2
    from tau2.runner.helpers import get_tasks

    package_root = Path(tau2.__file__).resolve().parent
    for repository_path, digest in manifest.source_blob_sha1:
        source = (
            data_root / repository_path.removeprefix("data/")
            if repository_path.startswith("data/")
            else package_root / repository_path.removeprefix("src/tau2/")
        )
        verify_git_blob_sha1(source, digest)

    split_path = data_root / "tau2/domains/retail/split_tasks.json"
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    train = {str(item) for item in split_data.get(manifest.split_name, ())}
    heldout = {str(item) for item in split_data.get(manifest.heldout_split_name, ())}
    evolution = set(manifest.evolution_task_ids)
    validation = set(manifest.validation_task_ids)
    test = set(manifest.heldout_task_ids)
    groups = (evolution, validation, test)
    if any(groups[left] & groups[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("E, V, and H must be disjoint task ID panels")
    if not (evolution | validation) <= train or not test <= heldout:
        raise ValueError("E/V must be τ-bench train IDs and H must be test IDs")
    if set(manifest.excluded_task_ids) & (evolution | validation | test):
        raise ValueError("excluded task IDs cannot appear in E, V, or H")

    train_ids = manifest.evolution_task_ids + (
        manifest.validation_task_ids if include_validation else ()
    )
    tasks = list(get_tasks(
        manifest.domain, task_split_name=manifest.split_name, task_ids=list(train_ids),
    ))
    if include_heldout:
        tasks.extend(get_tasks(
            manifest.domain,
            task_split_name=manifest.heldout_split_name,
            task_ids=list(manifest.heldout_task_ids),
        ))
    by_id = {str(task.id): task for task in tasks}
    expected = set(train_ids) | (test if include_heldout else set())
    if set(by_id) != expected:
        raise ValueError("τ-bench did not return exactly the requested task panel")
    return by_id


def _write_or_verify_context(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != value:
            raise ValueError("existing alternating run context differs from the frozen config")
    else:
        _write_json_once(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/alternating-evolution.yaml"))
    parser.add_argument("--tau2-data-dir", type=Path)
    parser.add_argument("--stop-before-next-episode-file", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_directory, result = run_from_config(
            args.config,
            tau2_data_dir=args.tau2_data_dir,
            stop_before_next_episode_file=args.stop_before_next_episode_file,
        )
    except StopBeforeEpisodeDispatch:
        print("EvoTau paused safely before the next τ-bench episode.", file=sys.stderr)
        return 75
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"EvoTau alternating evolution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "generations": len(result["generations"]),
        "final_customer_id": sha256_json(result["final_customer"])[:16],
        "final_service_id": sha256_json(result["final_service"])[:16],
        "result": str(output_directory / "alternating-result.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
