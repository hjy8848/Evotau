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
from .provider_diagnostics import safe_error
from .service_skills import ServiceSkillMemory, ServiceSkillMemoryV2
from .strategies import PromptStrategy
from .tau_adapter import verify_tau2_installation
from .tau_episode_runner import TauBenchEpisodeRunner
from .tau_provenance import (
    capture_code_provenance,
    role_model_args_for_runtime,
    sha256_json,
    verify_git_blob_sha1,
    write_manifest_once,
)


def run_from_config(config_path, **kwargs):
    """Serialize all writers before provenance binding and accounting restoration."""
    from .release_recovery import frozen_run_lock

    file = Path(config_path).expanduser().resolve()
    manifest = AlternatingManifest.from_mapping(load_config(file))
    project = file.parent.parent if file.parent.name == "configs" else file.parent
    with frozen_run_lock(project / manifest.checkpoint_path):
        return _run_from_config(file, **kwargs)


def _run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path | None = None,
    stop_before_next_episode_file: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_started = perf_counter()
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    manifest = AlternatingManifest.from_mapping(config)
    if (manifest.service_carrier == "skill_memory_v2"
            and json.loads(manifest.skill_evolution_v2_json)["algorithm_version"] not in ("direct_skill_evolution_v1", "direct_skill_v_validation_v2")):
        raise ValueError("Legacy Diagnoser config is read-only; create a Direct Skill experiment")
    if manifest.skill_evolution_v2_json:
        frozen = json.loads(manifest.skill_evolution_v2_json)
        if (frozen["evaluation"].get("promotion_protocol") == "v_primary"
                and not frozen["evaluation"]["calibration_confirmed"]
                and not frozen["evaluation"].get("allow_uncalibrated_launch", False)):
            raise ValueError("V-primary formal launch requires A/A calibration and an explicitly frozen confirmed policy")
    if not manifest.real_provider_enabled:
        raise RuntimeError("provider calls are disabled in this alternating-run config")
    if (
        manifest.service_carrier == "skill_memory_v2"
        and (manifest.run_validation or manifest.run_heldout)
        and manifest.request_budget_cap is None
    ):
        raise ValueError("V2 validation/heldout runs require an explicit finite request budget")
    data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError(
            "provide --tau2-data-dir or set TAU2_DATA_DIR to the pinned τ-bench data"
        )

    project_root = (
        config_file.parent.parent
        if config_file.parent.name == "configs"
        else config_file.parent
    )
    output_directory = project_root / manifest.output_path
    checkpoint_path = project_root / manifest.checkpoint_path
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        manifest = manifest.bind_saved_provenance(json.loads(manifest_path.read_text(encoding="utf-8")))
    elif any(output_directory.iterdir()):
        raise FileExistsError("refusing to bind an existing run directory without its manifest")
    else:
        write_manifest_once(manifest_path, manifest)
    execution_state_path, execution_state = _begin_execution_attempt(
        output_directory,
        manifest.sha256,
    )
    stage = "initialization"
    try:
        stage = "task_loading"
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
            raise ValueError(
                "evolution runner loaded a task outside the enabled E/V panels"
            )

        models = dict(manifest.role_models)
        role_args = role_model_args_for_runtime(manifest.role_model_args)
        evolver_args = manifest.role_model_args_dict["evolver"]
        if evolver_args.get("api_protocol") != "responses":
            evolver_args = role_args["evolver"]
        providers = LLMAlternatingEvolvers(
            model=models["evolver"],
            model_args=evolver_args,
            request_budget=budget,
            output_directory=output_directory,
        )
        providers.stop_before_next_episode_file = stop_before_next_episode_file
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
            "evolution_carriers": {
                "customer_carrier": manifest.customer_carrier,
                "service_carrier": manifest.service_carrier,
                "service_skill_runtime": manifest.service_skill_runtime,
                "service_mutation_ops": list(manifest.service_mutation_ops),
                "max_service_mutations_per_generation": manifest.max_service_mutations_per_generation,
            },
        }
        _write_or_verify_context(output_directory / "run-context.json", run_context)

        initial_customer = PromptStrategy(manifest.initial_customer_strategy)
        initial_service = (
            ServiceSkillMemoryV2() if manifest.service_carrier == "skill_memory_v2" else
            ServiceSkillMemory()
            if manifest.service_carrier == "skill_memory_v1"
            else PromptStrategy(manifest.initial_service_strategy)
        )
        episode_job_telemetry = EpisodeJobTelemetry(manifest.max_parallel_episodes)
        stage = "evolution"
        evolution_started = perf_counter()
        evolution_runner = run_alternating_evolution
        if manifest.service_carrier == 'skill_memory_v2':
            from .evolution_candidates import V2Providers
            from .skill_evolution import run_skill_evolution_v2
            # Keep legacy public orchestration arguments out of the V2 entry point.
            def evolution_runner(**kwargs):
                for key in ('clean_panel_size', 'customer_evolver', 'service_evolver', 'service_carrier',
                            'service_evolver_model', 'service_evolver_reasoning_effort', 'service_evolver_provider'):
                    kwargs.pop(key, None)
                return run_skill_evolution_v2(**kwargs, providers=V2Providers(providers),
                                             policy=json.loads(manifest.skill_evolution_v2_json))

        evolved = evolution_runner(
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
            service_carrier=manifest.service_carrier,
            runner=runner,
            customer_evolver=providers.customer_candidates,
            service_evolver=(
                providers.service_skill_mutation
                if manifest.service_carrier == "skill_memory_v1"
                else providers.service_candidate
            ),
            domain_policy=runner.service_policy_text,
            request_budget=budget,
            output_directory=output_directory,
            checkpoint_path=checkpoint_path,
            manifest_sha256=manifest.sha256,
            service_evolver_model=models["evolver"],
            service_evolver_reasoning_effort=(
                None if "reasoning_effort" not in evolver_args
                else str(evolver_args["reasoning_effort"])
            ),
            service_evolver_provider=(
                "InferAI Responses API" if evolver_args.get("api_protocol") == "responses"
                else str(dict(manifest.provider_provenance).get(
                    "provider", "τ-bench/LiteLLM configured provider",
                ))
            ),
        )

        # The fresh challenge is generated from E-only evidence before any H task
        # object or H trajectory is loaded into this process.
        fresh_customer = None
        if manifest.run_heldout:
            stage = "fresh_customer"
            fresh_runner = propose_fresh_customer_challenge
            if manifest.service_carrier == "skill_memory_v2":
                from .skill_evolution import propose_fresh_customer_v2

                def fresh_runner(result, _evolver, **kwargs):
                    kwargs.pop("request_budget", None)
                    kwargs.pop("proposal_path", None)
                    return propose_fresh_customer_v2(
                        result,
                        V2Providers(providers),
                        output_directory=output_directory,
                        **kwargs,
                    )

            fresh_customer = fresh_runner(
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
            stage = "heldout"
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
            heldout_tasks = {
                task_id: heldout_tasks[task_id] for task_id in manifest.heldout_task_ids
            }
            if fresh_customer is None and manifest.service_carrier != "skill_memory_v2":
                raise RuntimeError(
                    "held-out evaluation requires a fresh adaptive Customer"
                )
            endpoint_runner = run_final_endpoint_evaluation
            if manifest.service_carrier == "skill_memory_v2":
                from .skill_evolution import run_v2_endpoint_evaluation

                def endpoint_runner(**kwargs):
                    return run_v2_endpoint_evaluation(
                        gate_seeds=json.loads(manifest.skill_evolution_v2_json)[
                            "evaluation"
                        ]["gate_seeds"],
                        **kwargs,
                    )

            heldout_evaluation = endpoint_runner(
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
        stage = "finalization"
        api_usage_by_call_name = budget.api_usage_by_call_name()
        execution_state = _finish_execution_attempt(execution_state_path)
        completed_episodes = len(
            tuple((output_directory / "episodes").glob("*/episode-record.json"))
        )
        failed_episode_attempts = len(
            tuple((output_directory / "episodes").glob("*/incomplete-run.json")),
        )
        final_result = {
            "schema_version": 3 if manifest.service_carrier == "skill_memory_v2" else 2,
            "status": "complete",
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.sha256,
            "run_context_sha256": sha256_json(run_context),
            "initial_customer": initial_customer.to_dict(),
            "initial_service": initial_service.to_dict(),
            "final_customer": evolved.customer.to_dict(),
            "final_service": evolved.service.to_dict(),
            "final_service_carrier": manifest.service_carrier,
            "final_service_provenance": [
                item.to_dict() for item in getattr(evolved, "service_provenance", ())
            ],
            "generations": list(evolved.generations),
            "fresh_adaptive_customer": (
                None if fresh_customer is None else fresh_customer.to_dict()
            ),
            "fresh_customer_outcome": (
                json.loads((output_directory / "fresh-customer-proposal.json").read_text())
                if manifest.service_carrier == "skill_memory_v2" and manifest.run_heldout
                else None
            ),
            "heldout_endpoint_evaluation": heldout_evaluation,
            "evolution_fitness_seed": (
                manifest.seed
                if manifest.evolution_fitness_seed is None
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
                "validation_wall_clock_seconds": round(
                    validation_wall_clock_seconds, 6
                ),
                "heldout_wall_clock_seconds": round(heldout_wall_clock_seconds, 6),
                "total_wall_clock_seconds": execution_state["total_wall_clock_seconds"],
                "current_process_wall_clock_seconds": round(
                    perf_counter() - run_started, 6
                ),
                **episode_job_telemetry.snapshot(),
            },
        }
        _write_json_atomic(output_directory / "alternating-result.json", final_result)
        return output_directory, final_result
    except BaseException as exc:
        status = (
            "paused"
            if isinstance(exc, (StopBeforeEpisodeDispatch, KeyboardInterrupt))
            else "failed"
        )
        _finish_execution_attempt(
            execution_state_path,
            status=status,
            failure={
                "stage": stage,
                "failure_type": type(exc).__name__,
                "failure_message": safe_error(exc),
                "task_id": getattr(exc, "task_id", None),
                "panel_name": getattr(exc, "panel_name", None),
                "call_name": getattr(exc, "call_name", None),
                "diagnostics_ref": getattr(exc, "diagnostics_ref", None),
                "last_generation_stage": _last_generation_stage(output_directory),
            },
        )
        raise


def _begin_execution_attempt(
    output_directory: Path, manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = output_directory / "run-execution-state.json"
    now = datetime.now(UTC)
    attempts = []
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("manifest_sha256") != manifest_sha256:
            raise ValueError("run execution state belongs to a different frozen manifest")
        attempts = state.get("attempts", [])
        invocation_count = int(state.get("invocation_count", 0)) + 1
        first_started_at = state.get("first_started_at")
    else:
        invocation_count = 1
        first_started_at = now.isoformat()
    attempts.append({
        "invocation": invocation_count, "started_at": now.isoformat(), "status": "running",
        "evotau": capture_code_provenance(runtime_only=True).to_dict(),
    })
    state = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "first_started_at": first_started_at,
        "last_started_at": now.isoformat(),
        "invocation_count": invocation_count,
        "resume_count": max(0, invocation_count - 1),
        "attempts": attempts,
        "status": "running",
    }
    _write_json_atomic(path, state)
    return path, state


def _finish_execution_attempt(
    path: Path, *, status: str = "complete", failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    completed = datetime.now(UTC)
    attempts = state["attempts"]
    attempt = attempts[-1]
    started = datetime.fromisoformat(attempt["started_at"])
    attempt.update({
        "ended_at": completed.isoformat(), "status": status,
        "wall_clock_seconds": round((completed - started).total_seconds(), 6),
        "failure": None if failure is None else dict(failure),
    })
    state.update({
        "last_completed_at": completed.isoformat(), "status": status,
        "failure": None if failure is None else dict(failure),
        "total_wall_clock_seconds": round(sum(
            float(item.get("wall_clock_seconds", 0.0)) for item in attempts
        ), 6),
        "elapsed_since_first_start_seconds": round((
            completed - datetime.fromisoformat(state["first_started_at"])
        ).total_seconds(), 6),
    })
    _write_json_atomic(path, state)
    return state


def _last_generation_stage(output: Path) -> Mapping[str, Any] | None:
    paths = sorted(output.glob("generation-*-stage.json"))
    if not paths:
        return None
    try:
        payload = json.loads(paths[-1].read_text(encoding="utf-8"))
        return {key: payload.get(key) for key in ("generation", "stage")}
    except (OSError, ValueError):
        return None


def _api_usage_by_role(
    usage_by_call_name: Mapping[str, Mapping[str, int | float]],
) -> dict[str, dict[str, Any]]:
    call_names_by_role = {
        "customer": ("user_simulator_response",),
        "service": ("agent_response",),
        "evaluator": ("nl_assertions_eval",),
        "customer_evolver": ("evotau_customer_evolver",),
        "service_evolver": ("evotau_service_evolver", "evotau_service_skill_evolver", "evotau_service_skill_mutator"),
        "skill_activator": ("evotau_skill_activator",),
        "skill_crossover": ("evotau_skill_crossover",),
        "semantic_validator": ("evotau_customer_semantic_validator", "evotau_skill_semantic_validator"),
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
    from tau2.data_model.tasks import Task

    package_root = Path(tau2.__file__).resolve().parent
    for repository_path, digest in manifest.source_blob_sha1:
        source = (
            data_root / repository_path.removeprefix("data/")
            if repository_path.startswith("data/")
            else package_root / repository_path.removeprefix("src/tau2/")
        )
        verify_git_blob_sha1(source, digest)

    split_path = data_root / f"tau2/domains/{manifest.domain}/split_tasks.json"
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
    requested_ids = train_ids + (manifest.heldout_task_ids if include_heldout else ())
    task_records = _load_selected_retail_task_records(
        data_root / f"tau2/domains/{manifest.domain}/tasks.json",
        set(requested_ids),
    )
    tasks = [Task.model_validate(record) for record in task_records]
    by_id = {str(task.id): task for task in tasks}
    expected = set(requested_ids)
    if set(by_id) != expected:
        raise ValueError("τ-bench did not return exactly the requested task panel")
    return by_id


def _load_selected_retail_task_records(
    path: str | Path,
    selected_task_ids: set[str],
) -> list[dict[str, Any]]:
    """Stream pinned domain tasks and decode only selected records.

    τ-bench's Retail loader validates every task before filtering by ID. The
    pinned data format places ``id`` first in each object, so unselected records
    can be skipped lexically without decoding their scenario or description.
    """

    def skip_whitespace(handle, captured: list[str] | None = None) -> None:
        while True:
            position = handle.tell()
            char = handle.read(1)
            if not char or not char.isspace():
                handle.seek(position)
                return
            if captured is not None:
                captured.append(char)

    def read_json_string(handle) -> tuple[str, str]:
        if handle.read(1) != '"':
            raise ValueError("pinned Retail task records must use a JSON string key and ID")
        raw = ['"']
        escaped = False
        while True:
            char = handle.read(1)
            if not char:
                raise ValueError("unexpected end of pinned Retail task JSON string")
            raw.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                token = "".join(raw)
                value = json.loads(token)
                if not isinstance(value, str):
                    raise ValueError("Retail task object keys and IDs must be strings")
                return token, value

    def consume_object_tail(handle, *, collect: bool) -> str:
        depth = 1
        in_string = False
        escaped = False
        tail: list[str] = []
        while depth:
            char = handle.read(1)
            if not char:
                raise ValueError("unexpected end of pinned Retail task object")
            if collect:
                tail.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
        return "".join(tail)

    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        skip_whitespace(handle)
        if handle.read(1) != "[":
            raise ValueError("pinned Retail tasks must be a top-level JSON array")
        first = True
        while True:
            skip_whitespace(handle)
            char = handle.read(1)
            if char == "]":
                break
            if first:
                if char != "{":
                    raise ValueError("pinned Retail tasks must contain JSON objects")
            else:
                if char != ",":
                    raise ValueError("pinned Retail task array has an invalid separator")
                skip_whitespace(handle)
                if handle.read(1) != "{":
                    raise ValueError("pinned Retail tasks must contain JSON objects")

            prefix = ["{"]
            skip_whitespace(handle, prefix)
            key_raw, key = read_json_string(handle)
            prefix.append(key_raw)
            if key != "id":
                raise ValueError(
                    "pinned Retail task records must put the id field first for sealed loading"
                )
            skip_whitespace(handle, prefix)
            colon = handle.read(1)
            if colon != ":":
                raise ValueError("pinned Retail task id field is missing a colon")
            prefix.append(colon)
            skip_whitespace(handle, prefix)
            id_raw, task_id = read_json_string(handle)
            prefix.append(id_raw)
            collect = task_id in selected_task_ids
            tail = consume_object_tail(handle, collect=collect)
            if collect:
                record = json.loads("".join(prefix) + tail)
                if task_id in found:
                    raise ValueError(f"duplicate task ID in pinned Retail data: {task_id}")
                selected.append(record)
                found.add(task_id)
            first = False

        skip_whitespace(handle)
        if handle.read(1):
            raise ValueError("pinned Retail task JSON contains trailing data")

    missing = selected_task_ids - found
    if missing:
        raise ValueError(f"requested τ-bench task IDs were not found: {sorted(missing)}")
    return selected


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
