"""CLI entry point for a native τ-bench alternating evolution run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from .alternating import (
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
)


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path | None = None,
    stop_before_next_episode_file: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
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
    budget = RequestBudget(manifest.request_budget_cap)
    evolution_tasks = load_alternating_tasks(manifest, data_dir, include_heldout=False)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_dir,
        request_budget=budget,
        output_directory=output_directory,
        stop_before_next_episode_file=stop_before_next_episode_file,
        task_objects=evolution_tasks,
    )
    if set(runner.tasks) != set(manifest.evolution_task_ids + manifest.validation_task_ids):
        raise ValueError("evolution runner loaded a task outside E or V")

    models = dict(manifest.role_models)
    role_args = role_model_args_for_runtime(manifest.role_model_args)
    providers = LLMAlternatingEvolvers(
        model=models["evolver"],
        model_args=role_args["evolver"],
        judge_model=models["evaluator"],
        judge_model_args=role_args["evaluator"],
    )
    run_context = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "config_sha256": sha256_json(config),
        "provider_models": {
            "customer_evolver": models["evolver"],
            "service_evolver": models["evolver"],
            "selection_judge": models["evaluator"],
        },
        "task_panels": {
            "E": list(manifest.evolution_task_ids),
            "V": list(manifest.validation_task_ids),
            "H": list(manifest.heldout_task_ids),
        },
        "heldout_policy": "H task content is loaded only after evolution and fresh challenge generation.",
    }
    _write_or_verify_context(output_directory / "run-context.json", run_context)

    initial_customer = PromptStrategy(manifest.initial_customer_strategy)
    initial_service = PromptStrategy(manifest.initial_service_strategy)
    evolved = run_alternating_evolution(
        tasks=runner.tasks,
        evolution_task_ids=manifest.evolution_task_ids,
        validation_task_ids=manifest.validation_task_ids,
        seed=manifest.seed,
        generations=manifest.generations,
        customer_candidate_count=manifest.customer_candidates,
        clean_panel_size=manifest.clean_panel_size,
        initial_customer=initial_customer,
        initial_service=initial_service,
        runner=runner,
        customer_evolver=providers.customer_candidates,
        service_evolver=providers.service_candidate,
        customer_judge=providers.choose_customer,
        service_judge=providers.service_is_better,
        domain_policy=runner.service_policy_text,
        request_budget=budget,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        manifest_sha256=manifest.sha256,
    )

    # The fresh challenge is generated from E-only evidence before any H task
    # object or H trajectory is loaded into this process.
    fresh_customer = propose_fresh_customer_challenge(
        evolved,
        providers.customer_candidates,
        tasks=runner.tasks,
        evolution_task_ids=manifest.evolution_task_ids,
        runner=runner,
        domain_policy=runner.service_policy_text,
        seed=manifest.seed + manifest.generations + 10_000,
        request_budget=budget,
    )
    heldout_tasks = load_alternating_tasks(manifest, data_dir, include_heldout=True)
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
    heldout_evaluation = run_final_endpoint_evaluation(
        heldout_tasks=heldout_tasks,
        heldout_task_ids=manifest.heldout_task_ids,
        seed=manifest.seed + manifest.generations + 50_000,
        initial_service=initial_service,
        final_service=evolved.service,
        fresh_customer=fresh_customer,
        runner=heldout_runner,
        output_path=output_directory / "heldout-endpoint-evaluation.json",
    )
    final_result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "run_context_sha256": sha256_json(run_context),
        "initial_customer": initial_customer.to_dict(),
        "initial_service": initial_service.to_dict(),
        "final_customer": evolved.customer.to_dict(),
        "final_service": evolved.service.to_dict(),
        "generations": list(evolved.generations),
        "fresh_adaptive_customer": fresh_customer.to_dict(),
        "heldout_endpoint_evaluation": heldout_evaluation,
        "provider_usage": budget.snapshot().to_dict(),
    }
    _write_json_atomic(output_directory / "alternating-result.json", final_result)
    return output_directory, final_result


def load_alternating_tasks(
    manifest: AlternatingManifest,
    data_dir: str | Path,
    *,
    include_heldout: bool,
) -> dict[str, Any]:
    """Load only the requested panel; before final evaluation, H contributes IDs only."""

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

    train_ids = manifest.evolution_task_ids + manifest.validation_task_ids
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
