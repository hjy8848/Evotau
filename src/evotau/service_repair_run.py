"""One-generation Service PromptStrategy continuation from archived E evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from .alternating import (
    EpisodeJobTelemetry,
    LLMAlternatingEvolvers,
    _accuracy,
    _episode_ref,
    _run_panel,
    _service_context_episodes,
    _write_json_atomic,
)
from .alternating_manifest import AlternatingManifest
from .budget import RequestBudget
from .phase0 import load_config
from .records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from .strategies import PromptStrategy
from .tau_episode_runner import TauBenchEpisodeRunner
from .tau_provenance import (
    role_model_args_for_runtime,
    sha256_json,
    write_manifest_once,
)

ARCHIVED_E_TASK_IDS = (
    "66", "92", "29", "67", "106", "22", "69", "98", "93", "88",
    "4", "21", "8", "54", "107", "48", "52", "80", "35", "16",
)
ARCHIVED_EXPERIMENT_ID = (
    "evotau-alternating-gpt61sol-e20-c1-mechanism-smoke-20261006-rpm-safe-timeout180"
)
SERVICE_PANEL_NAME = "service-repair-baseline-c0-x-s1"
_SERVICE_SYSTEM_PROMPT = (
    "You evolve the Service from native τ-bench task outcomes. Study the fixed task policy, "
    "the current Service strategy, Customer strategy, E-panel accuracy, real interaction "
    "trajectories, tool results, and task-success results. Propose a reusable natural-language "
    "Service strategy that improves task accuracy without changing tasks or policy. "
    "Do not modify tasks, policy, tools, backend, or evaluator. "
    "Do not use reference answers or hidden data. Return only JSON with string fields "
    "`analysis` and `strategy`."
)


@dataclass(frozen=True, slots=True)
class ArchivedServiceEvidence:
    archive_root: Path
    manifest_sha256: str
    task_ids: tuple[str, ...]
    seed: int
    customer: PromptStrategy
    service: PromptStrategy
    accuracy: float
    episodes: tuple[EpisodeRecord, ...]


class ArchivedTrajectoryReader:
    """Read trajectories only from the selected archived episode references."""

    def __init__(self, archive_root: str | Path) -> None:
        self.archive_root = Path(archive_root).expanduser().resolve()

    def load_trajectory(self, episode: EpisodeRecord) -> Mapping[str, Any] | None:
        if episode.trajectory_ref is None:
            return None
        root = self.archive_root.resolve()
        path = (root / episode.trajectory_ref).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("archived trajectory reference escapes the evidence directory") from exc
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("selected archived trajectory is missing or not a regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or str(payload.get("id")) != episode.episode_id
            or str(payload.get("task_id")) != episode.task_id
            or int(payload.get("seed", -1)) != episode.seed
        ):
            raise ValueError("archived native simulation does not match its EpisodeRecord")
        return payload


def load_archived_c0_s0_evidence(archive_directory: str | Path) -> ArchivedServiceEvidence:
    """Verify and load only the frozen generation-0 incumbent C0×S0 panel."""

    root = Path(archive_directory).expanduser().resolve()
    for name in ("manifest.json", "result-index.json", "generation-0000-stage.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"required archived evidence file is missing: {name}")
    manifest = _read_object(root / "manifest.json")
    manifest_sha = manifest.get("manifest_sha256")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(manifest_sha, str) or sha256_json(manifest_payload) != manifest_sha:
        raise ValueError("archived manifest SHA-256 is invalid")
    if manifest.get("experiment_id") != ARCHIVED_EXPERIMENT_ID:
        raise ValueError("archive is not the requested frozen GPT mechanism-smoke run")
    if manifest.get("upstream", {}).get("commit") != "b7ea9074c1cba482b30687fecdb5c8425fd6f619":
        raise ValueError("archived τ-bench pin differs from the expected upstream release")
    panels = manifest.get("task_panels", {})
    task_ids = tuple(str(item) for item in panels.get("E", ()))
    if task_ids != ARCHIVED_E_TASK_IDS:
        raise ValueError("archived E task panel does not match the frozen 20-task continuation panel")
    if (manifest.get("seed") != 1 or manifest.get("evolution_fitness_seed") != 1
            or manifest.get("max_steps") != 32 or manifest.get("max_parallel_episodes") != 1
            or manifest.get("run_validation") is not False
            or manifest.get("run_heldout") is not False):
        raise ValueError("archived seed, runtime, or disabled V/H settings are inconsistent")
    models = manifest.get("role_models", {})
    if models.get("evolver") != "gpt-6.1-sol":
        raise ValueError("archived Evolver model differs from the frozen experiment")
    for role in ("agent", "customer", "evaluator"):
        if models.get(role) != "openai/deepseek-v4-flash":
            raise ValueError(f"archived {role} runtime model differs from the frozen experiment")

    stage = _read_object(root / "generation-0000-stage.json")
    if (stage.get("manifest_sha256") != manifest_sha or stage.get("generation") != 0
            or stage.get("stage") != "customer_selected"
            or stage.get("selected_customer") != "incumbent"):
        raise ValueError("archived stage does not freeze the selected generation-0 incumbent")
    customer = PromptStrategy(str(stage.get("customer_before", {}).get("text", "")))
    service = PromptStrategy(str(stage.get("service_before", {}).get("text", "")))
    if customer.text != "" or service.text != "":
        raise ValueError("continuation requires the original frozen C0='' and S0='' strategies")
    expected_customer_id = customer_strategy_id(customer)
    expected_service_id = service_strategy_id(service)
    selected_accuracy = stage.get("selected_accuracy")
    if selected_accuracy != 0.8:
        raise ValueError("archived C0×S0 accuracy does not match 16/20")

    refs = stage.get("selected_episodes")
    if not isinstance(refs, list) or len(refs) != len(task_ids):
        raise ValueError("archived selected incumbent panel must contain exactly 20 episodes")
    if tuple(str(ref.get("task_id")) for ref in refs) != task_ids:
        raise ValueError("archived selected episode order or task IDs differ from E20")

    records_by_id: dict[str, tuple[Path, EpisodeRecord]] = {}
    for path in (root / "episodes").glob("*/episode-record.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("archived episode record is not a regular file")
        record = EpisodeRecord.from_dict(_read_object(path))
        if record.episode_id in records_by_id:
            raise ValueError("archived episode IDs are not unique")
        records_by_id[record.episode_id] = (path.parent, record)

    selected: list[EpisodeRecord] = []
    for ref in refs:
        episode_id = str(ref.get("episode_id", ""))
        match = records_by_id.get(episode_id)
        if match is None:
            raise ValueError(f"selected incumbent EpisodeRecord is missing: {episode_id}")
        episode_directory, record = match
        if (
            record.task_id != str(ref.get("task_id"))
            or record.seed != 1
            or record.status != EpisodeStatus.COMPLETE
            or record.customer_strategy_id != expected_customer_id
            or record.service_strategy_id != expected_service_id
            or record.trajectory_ref != ref.get("trajectory_ref")
        ):
            raise ValueError(f"archived incumbent episode does not match its frozen ref: {episode_id}")
        telemetry_path = episode_directory / "run-telemetry.json"
        telemetry = _read_object(telemetry_path)
        key = telemetry.get("episode_key")
        key_sha = telemetry.get("episode_key_sha256")
        if not isinstance(key, dict) or sha256_json(key) != key_sha:
            raise ValueError(f"archived incumbent episode key is invalid: {episode_id}")
        if (
            key.get("panel_name") != "generation-0-customer-incumbent"
            or key.get("task_id") != record.task_id
            or key.get("seed") != 1
            or key.get("customer_strategy_id") != expected_customer_id
            or key.get("service_strategy_id") != expected_service_id
        ):
            raise ValueError(f"archived episode was not generated by the C0×S0 incumbent panel: {episode_id}")
        ArchivedTrajectoryReader(root).load_trajectory(record)
        selected.append(record)

    accuracy = _accuracy(selected)
    if accuracy != selected_accuracy or sum(item.task_success is True for item in selected) != 16:
        raise ValueError("archived incumbent trajectories do not verify the recorded 16/20 accuracy")
    index = _read_object(root / "result-index.json")
    if (
        index.get("manifest_sha256") != manifest_sha
        or index.get("status") != "failed_partial"
        or index.get("episodes", {}).get("generation-0-customer-incumbent") is None
        or index.get("accuracy", {}).get("selected_customer") != "incumbent"
    ):
        raise ValueError("archived result index does not corroborate selected C0×S0 evidence")
    return ArchivedServiceEvidence(
        archive_root=root,
        manifest_sha256=manifest_sha,
        task_ids=task_ids,
        seed=1,
        customer=customer,
        service=service,
        accuracy=accuracy,
        episodes=tuple(selected),
    )


def run_service_repair_from_evidence(
    *,
    evidence: ArchivedServiceEvidence,
    tasks: Mapping[str, Any],
    runner: Any,
    domain_policy: str,
    service_evolver: Callable[[Mapping[str, Any]], Mapping[str, str]],
    output_directory: str | Path,
    manifest_sha256: str,
    model: str,
    reasoning_effort: str,
    api_base: str,
    request_budget: RequestBudget,
    max_parallel_episodes: int = 1,
) -> dict[str, Any]:
    """Make one Service proposal and at most one frozen-C0 E20 evaluation."""

    loaded_task_ids = {str(item) for item in tasks}
    if loaded_task_ids != set(evidence.task_ids):
        raise ValueError("loaded tasks must contain exactly the archived E20 panel")
    if max_parallel_episodes != 1:
        raise ValueError("Service-repair continuation must use the verified RPM-safe concurrency of 1")
    if not isinstance(domain_policy, str) or not domain_policy.strip():
        raise ValueError("the pinned Retail policy text is required")
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    service_context = {
        "generation": 0,
        "task_interactions": _service_context_episodes(
            evidence.episodes, ArchivedTrajectoryReader(evidence.archive_root), tasks,
        ),
        "selected_customer_accuracy": evidence.accuracy,
        "customer_strategy": evidence.customer.text,
        "current_service_strategy": evidence.service.text,
        "service_policy": domain_policy,
        "accuracy_history": [],
    }
    context_sha = sha256_json(service_context)
    context_artifact = {
        "source_manifest_sha256": evidence.manifest_sha256,
        "input_context_sha256": context_sha,
        "context": service_context,
    }
    _write_or_verify_json(output_root / "service-evolver-input.json", context_artifact)

    started = perf_counter()
    proposal_path = output_root / "service-proposal.json"
    call_path = output_root / "service-evolver-call.json"
    if proposal_path.exists():
        proposal = _read_object(proposal_path)
        if (
            proposal.get("manifest_sha256") != manifest_sha256
            or proposal.get("source_manifest_sha256") != evidence.manifest_sha256
            or proposal.get("input_context_sha256") != context_sha
            or proposal.get("model") != model
            or proposal.get("reasoning_effort") != reasoning_effort
        ):
            raise ValueError("saved Service proposal belongs to different frozen evidence")
        analysis = proposal["analysis"]
        strategy_text = proposal["strategy"]
        proposal_latency = proposal.get("latency_seconds")
    else:
        if call_path.exists():
            saved_call = _read_object(call_path)
            raise RuntimeError(
                "Service Evolver already has a recorded provider attempt "
                f"(status={saved_call.get('status')}); refusing to dispatch a duplicate request"
            )
        request = {
            "status": "dispatching",
            "attempt_count": 1,
            "manifest_sha256": manifest_sha256,
            "source_manifest_sha256": evidence.manifest_sha256,
            "input_context_sha256": context_sha,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "api_protocol": "inferai-responses",
            "endpoint": f"{api_base.rstrip('/')}/responses",
            "request_args": {
                "model": model,
                "reasoning": {"effort": reasoning_effort},
                "tools": "omitted; no tools were supplied",
                "temperature": "not supplied",
            },
            "usage": "pending",
        }
        _write_json_atomic(call_path, request)
        before = request_budget.snapshot()
        try:
            proposal = service_evolver(service_context)
            if (not isinstance(proposal, Mapping)
                    or not isinstance(proposal.get("analysis"), str)
                    or not isinstance(proposal.get("strategy"), str)):
                raise TypeError("Service Evolver must return string analysis and strategy fields")
        except Exception as exc:
            after = request_budget.snapshot()
            usage_row = request_budget.api_usage_by_call_name().get("evotau_service_evolver")
            failure = {
                **request,
                "status": "failed",
                "elapsed_seconds": round(perf_counter() - started, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "attempt_count": after.attempts - before.attempts,
                "usage": _usage_status(usage_row),
                "api_usage": usage_row,
            }
            _write_json_atomic(call_path, failure)
            _write_json_atomic(output_root / "run-failure.json", {
                "schema_version": 1,
                "stage": "service-evolver-proposal",
                "status": "failed_partial",
                "manifest_sha256": manifest_sha256,
                "source_manifest_sha256": evidence.manifest_sha256,
                "input_context_sha256": context_sha,
                "provider_attempts": failure["attempt_count"],
                "model": model,
                "reasoning_effort": reasoning_effort,
                "api_protocol": "inferai-responses",
                "latency_seconds": failure["elapsed_seconds"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "usage": failure["usage"],
                "gen1_started": False,
                "validation_run": False,
                "heldout_run": False,
            })
            _write_json_atomic(output_root / "service-repair-result.json", {
                "schema_version": 1,
                "experiment_id": "evotau-gpt61sol-service-promptstrategy-baseline-e20-20261006",
                "status": "failed_partial",
                "failure_stage": "service-evolver-proposal",
                "manifest_sha256": manifest_sha256,
                "source_manifest_sha256": evidence.manifest_sha256,
                "input_context_sha256": context_sha,
                "source_evidence": "archived generation-0 selected C0×S0 panel",
                "customer_evolution_rerun": False,
                "customer_candidate_panel_rerun": False,
                "customer_strategy": evidence.customer.to_dict(),
                "frozen_service_before": evidence.service.to_dict(),
                "service_analysis": None,
                "proposed_service": None,
                "evolution_task_ids": list(evidence.task_ids),
                "evolution_fitness_seed": evidence.seed,
                "s0_accuracy": {
                    "successes": 16,
                    "episodes": 20,
                    "accuracy": evidence.accuracy,
                    "episode_refs": [_episode_ref(item) for item in evidence.episodes],
                },
                "s1_accuracy": None,
                "net_accuracy_delta": None,
                "paired_outcomes": None,
                "transition_counts": None,
                "s1_episode_refs": [],
                "provider": {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "api_protocol": "inferai-responses",
                    "api_base": api_base,
                    "attempt_count": failure["attempt_count"],
                    "usage": failure["usage"],
                    "latency_seconds": failure["elapsed_seconds"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                "max_steps": 32,
                "max_parallel_episodes": max_parallel_episodes,
                "gen1_started": False,
                "validation_run": False,
                "heldout_run": False,
                "research_scope": (
                    "Diagnostic free-form PromptStrategy Service-repair baseline; not a completed "
                    "alternating-evolution experiment."
                ),
                "wall_clock_seconds": round(perf_counter() - started, 6),
                "api_usage_by_call_name": request_budget.api_usage_by_call_name(),
                "provider_usage": request_budget.snapshot().to_dict(),
            })
            raise
        after = request_budget.snapshot()
        usage_row = request_budget.api_usage_by_call_name().get("evotau_service_evolver")
        strategy = PromptStrategy(proposal["strategy"])
        proposal_document = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "source_manifest_sha256": evidence.manifest_sha256,
            "input_context_sha256": context_sha,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "api_protocol": "inferai-responses",
            "attempt_count": after.attempts - before.attempts,
            "usage": _usage_status(usage_row),
            "latency_seconds": round(perf_counter() - started, 6),
            "analysis": proposal["analysis"],
            "strategy": strategy.text,
            "strategy_id": service_strategy_id(strategy),
        }
        if proposal_document["attempt_count"] != 1:
            raise RuntimeError("expected exactly one Service Evolver provider attempt")
        _write_json_once(proposal_path, proposal_document)
        _write_json_atomic(call_path, {
            **request,
            "status": "succeeded",
            "elapsed_seconds": proposal_document["latency_seconds"],
            "attempt_count": proposal_document["attempt_count"],
            "usage": proposal_document["usage"],
            "api_usage": usage_row,
        })
        analysis = proposal_document["analysis"]
        strategy_text = proposal_document["strategy"]
        proposal_latency = proposal_document["latency_seconds"]

    proposed_service = PromptStrategy(strategy_text)
    common = {
        "schema_version": 1,
        "experiment_id": "evotau-gpt61sol-service-promptstrategy-baseline-e20-20261006",
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": evidence.manifest_sha256,
        "source_evidence": "archived generation-0 selected C0×S0 panel",
        "customer_evolution_rerun": False,
        "customer_candidate_panel_rerun": False,
        "customer_strategy": evidence.customer.to_dict(),
        "frozen_service_before": evidence.service.to_dict(),
        "service_analysis": analysis,
        "proposed_service": {
            "strategy": proposed_service.text,
            "strategy_id": service_strategy_id(proposed_service),
        },
        "evolution_task_ids": list(evidence.task_ids),
        "evolution_fitness_seed": evidence.seed,
        "max_steps": 32,
        "max_parallel_episodes": max_parallel_episodes,
        "validation_run": False,
        "heldout_run": False,
        "gen1_started": False,
        "research_scope": (
            "Diagnostic free-form PromptStrategy Service-repair baseline; not a completed "
            "alternating-evolution experiment."
        ),
        "provider": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "api_protocol": "inferai-responses",
            "api_base": api_base,
            "attempt_count": 1,
            "usage": _usage_status(request_budget.api_usage_by_call_name().get(
                "evotau_service_evolver",
            )),
            "latency_seconds": proposal_latency,
        },
    }

    if proposed_service.text == evidence.service.text:
        result = {
            **common,
            "status": "no_op",
            "s0_accuracy": {"successes": 16, "episodes": 20, "accuracy": evidence.accuracy},
            "s1_accuracy": None,
            "net_accuracy_delta": None,
            "paired_outcomes": None,
            "transition_counts": None,
            "s1_episode_refs": [],
            "stopped_reason": "S1 exactly matches S0; duplicate E20 rollout was skipped.",
            "wall_clock_seconds": round(perf_counter() - started, 6),
        }
        _write_json_atomic(output_root / "service-repair-result.json", result)
        return result

    telemetry = EpisodeJobTelemetry(max_parallel_episodes)
    candidate_runs = _run_panel(
        runner,
        task_ids=evidence.task_ids,
        tasks=tasks,
        seed=evidence.seed,
        customer=evidence.customer,
        service=proposed_service,
        panel_name=SERVICE_PANEL_NAME,
        max_parallel_episodes=max_parallel_episodes,
        telemetry=telemetry,
    )
    if len(candidate_runs) != len(evidence.task_ids):
        raise RuntimeError("S1 E20 panel did not return exactly one episode per task")
    transition_data = paired_outcome_transitions(evidence.episodes, candidate_runs)
    result = {
        **common,
        "status": "complete",
        "s0_accuracy": {
            "successes": sum(item.task_success is True for item in evidence.episodes),
            "episodes": len(evidence.episodes),
            "accuracy": evidence.accuracy,
            "episode_refs": [_episode_ref(item) for item in evidence.episodes],
        },
        "s1_accuracy": {
            "successes": sum(item.task_success is True for item in candidate_runs),
            "episodes": len(candidate_runs),
            "accuracy": _accuracy(candidate_runs),
        },
        "net_accuracy_delta": round(_accuracy(candidate_runs) - evidence.accuracy, 6),
        **transition_data,
        "s1_episode_refs": [_episode_ref(item) for item in candidate_runs],
        "episode_job_telemetry": telemetry.snapshot(),
        "stopped_reason": "Exactly one C0×S1 E20 panel completed; no later generation or panel was started.",
        "wall_clock_seconds": round(perf_counter() - started, 6),
    }
    _write_json_atomic(output_root / "service-repair-result.json", result)
    return result


def paired_outcome_transitions(
    s0_episodes: Sequence[EpisodeRecord], s1_episodes: Sequence[EpisodeRecord],
) -> dict[str, Any]:
    s0_by_task = {item.task_id: item for item in s0_episodes}
    s1_by_task = {item.task_id: item for item in s1_episodes}
    if len(s0_by_task) != len(s0_episodes) or len(s1_by_task) != len(s1_episodes):
        raise ValueError("paired episode panels must contain unique task IDs")
    if set(s0_by_task) != set(s1_by_task) or not s0_by_task:
        raise ValueError("paired episode panels must contain the same non-empty task IDs")
    rows = []
    counts = {"fail → pass": 0, "pass → fail": 0, "pass → pass": 0, "fail → fail": 0}
    for task_id, old in s0_by_task.items():
        new = s1_by_task[task_id]
        if old.task_success is None or new.task_success is None:
            raise ValueError("paired accuracy diagnostics require a native success value per task")
        old_success = old.task_success is True
        new_success = new.task_success is True
        transition = (
            "pass → pass" if old_success and new_success
            else "pass → fail" if old_success
            else "fail → pass" if new_success
            else "fail → fail"
        )
        counts[transition] += 1
        rows.append({
            "task_id": task_id,
            "S0_success": old_success,
            "S1_success": new_success,
            "transition": transition,
            "S0_episode_ref": _episode_ref(old),
            "S1_episode_ref": _episode_ref(new),
        })
    return {
        "paired_outcomes": rows,
        "transition_counts": {
            **counts,
            "repaired_task_count": counts["fail → pass"],
            "regressed_task_count": counts["pass → fail"],
            "unchanged_pass_count": counts["pass → pass"],
            "unchanged_fail_count": counts["fail → fail"],
        },
    }


def load_service_repair_tasks(
    manifest: AlternatingManifest, data_dir: str | Path,
) -> dict[str, Any]:
    """Load E only: this continuation never reads task content from V or H."""

    from .alternating_run import load_alternating_tasks

    tasks = load_alternating_tasks(
        manifest,
        data_dir,
        include_validation=False,
        include_heldout=False,
    )
    if set(tasks) != set(manifest.evolution_task_ids):
        raise ValueError("service-repair continuation must load exactly E, with no V/H content")
    return tasks


def run_from_config(
    config_path: str | Path,
    *,
    archived_evidence_directory: str | Path,
    tau2_data_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    config_file = Path(config_path).expanduser().resolve()
    config = load_config(config_file)
    manifest = AlternatingManifest.from_mapping(config)
    _validate_continuation_config(manifest)
    data_dir = tau2_data_dir or os.environ.get("TAU2_DATA_DIR")
    if data_dir is None:
        raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR to the pinned τ-bench data")

    evidence = load_archived_c0_s0_evidence(archived_evidence_directory)
    if tuple(manifest.evolution_task_ids) != evidence.task_ids:
        raise ValueError("continuation E panel differs from the archived evidence E panel")
    if (manifest.seed != evidence.seed or manifest.evolution_fitness_seed != evidence.seed
            or manifest.max_steps != 32):
        raise ValueError("continuation seed/max_steps differs from the archived C0×S0 run")

    project_root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    output_directory = project_root / manifest.output_path
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        if _read_object(manifest_path) != manifest.to_document():
            raise ValueError("existing service-repair output belongs to a different frozen manifest")
    elif any(output_directory.iterdir()):
        raise FileExistsError("refusing to use a non-empty continuation directory without its manifest")
    else:
        write_manifest_once(manifest_path, manifest)

    budget = RequestBudget(manifest.request_budget_cap)
    budget.enable_live_usage(output_directory / "api-usage-live.json")
    tasks = load_service_repair_tasks(manifest, data_dir)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_dir,
        request_budget=budget,
        output_directory=output_directory,
        task_objects=tasks,
    )
    evolver = LLMAlternatingEvolvers(
        model=dict(manifest.role_models)["evolver"],
        model_args=manifest.role_model_args_dict["evolver"],
        request_budget=budget,
    )
    run_context = {
        "schema_version": 1,
        "run_type": "service-repair-continuation",
        "manifest_sha256": manifest.sha256,
        "source_manifest_sha256": evidence.manifest_sha256,
        "reused_archived_customer_incumbent_episodes": 20,
        "customer_evolver_called": False,
        "customer_candidate_panel_rerun": False,
        "new_service_candidate_panel": "C0×S1 E20 only, if S1 differs from S0",
        "loaded_task_panels": {"E": list(evidence.task_ids), "V": [], "H": []},
        "evolver_model": dict(manifest.role_models)["evolver"],
        "evolver_reasoning_effort": manifest.role_model_args_dict["evolver"]["reasoning_effort"],
        "api_protocol": "InferAI Responses API",
        "max_parallel_episodes": manifest.max_parallel_episodes,
    }
    _write_or_verify_json(output_directory / "run-context.json", run_context)
    result = run_service_repair_from_evidence(
        evidence=evidence,
        tasks=tasks,
        runner=runner,
        domain_policy=runner.service_policy_text,
        service_evolver=evolver.service_candidate,
        output_directory=output_directory,
        manifest_sha256=manifest.sha256,
        model=dict(manifest.role_models)["evolver"],
        reasoning_effort=manifest.role_model_args_dict["evolver"]["reasoning_effort"],
        api_base=manifest.role_model_args_dict["evolver"]["api_base"],
        request_budget=budget,
        max_parallel_episodes=manifest.max_parallel_episodes,
    )
    result["api_usage_by_call_name"] = budget.api_usage_by_call_name()
    result["provider_usage"] = budget.snapshot().to_dict()
    _write_json_atomic(output_directory / "service-repair-result.json", result)
    return output_directory, result


def _validate_continuation_config(manifest: AlternatingManifest) -> None:
    if manifest.experiment_id != "evotau-gpt61sol-service-promptstrategy-baseline-e20-20261006":
        raise ValueError("unexpected Service-repair continuation experiment ID")
    models = dict(manifest.role_models)
    args = manifest.role_model_args_dict
    if models.get("evolver") != "gpt-6.1-sol":
        raise ValueError("Service-repair baseline requires the specified gpt-6.1-sol model")
    if args["evolver"].get("api_protocol") != "responses":
        raise ValueError("Service-repair baseline requires the InferAI Responses API transport")
    if args["evolver"].get("reasoning_effort") != "medium":
        raise ValueError("Service Evolver fallback must use reasoning_effort=medium")
    for role in ("agent", "customer", "evaluator"):
        if models.get(role) != "openai/deepseek-v4-flash":
            raise ValueError(f"{role} runtime model must remain DeepSeek V4 Flash")
    expected_runtime_args = {
        "temperature": 0.0,
        "api_base": "https://inferaiapi.com/v1",
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    runtime_args = role_model_args_for_runtime(manifest.role_model_args)
    if any(runtime_args[role] != expected_runtime_args for role in ("agent", "customer", "evaluator")):
        raise ValueError("DeepSeek runtime model arguments must remain unchanged")
    if (manifest.max_parallel_episodes != 1 or manifest.max_steps != 32
            or manifest.request_budget_cap is not None or manifest.run_validation
            or manifest.run_heldout):
        raise ValueError("continuation must use RPM-safe E-only settings with no request cap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/service-repair-baseline-gpt61sol-e20-20261006.yaml"),
    )
    parser.add_argument(
        "--archived-evidence", type=Path,
        default=Path("experiments/results/evotau-alternating-gpt61sol-e20-c1-mechanism-smoke-20261006-partial"),
    )
    parser.add_argument("--tau2-data-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_directory, result = run_from_config(
            args.config,
            archived_evidence_directory=args.archived_evidence,
            tau2_data_dir=args.tau2_data_dir,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"EvoTau Service-repair continuation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "source_accuracy": result["s0_accuracy"]["accuracy"],
        "candidate_accuracy": (
            None if result.get("s1_accuracy") is None else result["s1_accuracy"]["accuracy"]
        ),
        "repaired": result.get("transition_counts", {}).get("repaired_task_count"),
        "regressed": result.get("transition_counts", {}).get("regressed_task_count"),
        "result": str(output_directory / "service-repair-result.json"),
    }, ensure_ascii=False, indent=2))
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _usage_status(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"status": "unavailable", "prompt_tokens": None, "completion_tokens": None}
    if int(row.get("usage_responses", 0)) < 1:
        return {"status": "unavailable", "prompt_tokens": None, "completion_tokens": None}
    return {
        "status": "available",
        "prompt_tokens": int(row.get("prompt_tokens", 0)),
        "completion_tokens": int(row.get("completion_tokens", 0)),
    }


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_object(path) != value:
            raise ValueError(f"existing immutable artifact differs from current continuation: {path.name}")
        return
    _write_json_atomic(path, value)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen Service proposal: {path}")
    _write_json_atomic(path, value)


if __name__ == "__main__":
    raise SystemExit(main())
