"""Run the two-episode, incumbent-only activation concurrency engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from evotau import native_runner as native_runner_module
from evotau.budget import RequestBudget
from evotau.episode_execution import EpisodeSpec, run_episode_batch
from evotau.manifest import ActivationSmokeManifest
from evotau.native_runner import TauBenchEpisodeRunner, _write_json_once
from evotau.phase0 import load_config
from evotau.phase0_run import _load_pinned_tasks
from evotau.phase3_run import load_provider_bundle
from evotau.records import EpisodeRecord
from evotau.strategies import CustomerStrategy, ServiceStrategy


def _project_root(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent


def _write_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("concurrency-smoke artifact tree cannot contain symlinks")
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    sums_path = root / "SHA256SUMS"
    with sums_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")


def _redact_text(value: str) -> str:
    patterns = (
        (r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]"),
        (r"(?i)(api[_ -]?key[\"'\s:=]+)[^\s,;\"']+", r"\1[REDACTED]"),
        (r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]"),
    )
    result = value
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result[:1200]


def _safe_json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or type(value) in {bool, int, float}:
        return value
    return _redact_text(str(value))


def run_smoke(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path,
    provider_plugin: str,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    project_root = _project_root(config_file)
    config = load_config(config_file)
    experiment = config["experiment"]
    review_path = (project_root / experiment["task_review_path"]).resolve()
    if not review_path.is_relative_to(project_root) or not review_path.is_file():
        raise ValueError("reviewed task panel is missing or outside the repository")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = ActivationSmokeManifest.from_mapping(
        config, task_review_document=review,
    )
    smoke = config.get("concurrency_smoke")
    if not isinstance(smoke, dict):
        raise TypeError("concurrency_smoke settings are required")
    task_ids = tuple(str(item) for item in smoke.get("task_ids", ()))
    if len(task_ids) != 2 or len(set(task_ids)) != 2:
        raise ValueError("engineering smoke must select exactly two distinct reviewed E tasks")
    if not set(task_ids) <= set(manifest.evolution_task_ids):
        raise ValueError("engineering smoke tasks must be drawn from the reviewed E panel")
    if manifest.max_concurrency != 2 or manifest.provider_retries != 0:
        raise ValueError("engineering smoke requires concurrency=2 and retries=0")
    if not manifest.real_provider_enabled:
        raise ValueError("engineering smoke requires the explicit real-provider opt-in")

    data_root = Path(tau2_data_dir).expanduser().resolve()
    _load_pinned_tasks(
        manifest,
        data_dir=data_root,
        task_selection=experiment["task_selection"],
        task_ids=manifest.evolution_task_ids + manifest.validation_task_ids,
    )
    bundle = load_provider_bundle(provider_plugin, config=config, manifest=manifest)
    budget = RequestBudget(manifest.request_budget_cap)
    output_directory = (project_root / manifest.output_path).resolve()
    if not output_directory.is_relative_to(project_root):
        raise ValueError("concurrency-smoke output path must remain inside the repository")
    request_accounting_source = "live_provider_instrumentation"
    reconstructed_intervals: list[dict[str, Any]] = []
    if manifest.request_budget_cap is None and output_directory.is_dir():
        cached_rows: dict[str, dict[str, Any]] = {}
        cached_paths: dict[str, tuple[Path, Path]] = {}
        for telemetry_path in (output_directory / "episodes").glob("*/run-telemetry.json"):
            episode_directory = telemetry_path.parent
            if (episode_directory / "incomplete-run.json").exists():
                continue
            if not (episode_directory / "episode-record.json").is_file():
                continue
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            episode_key = telemetry.get("episode_key")
            if not isinstance(episode_key, dict):
                continue
            task_id = str(episode_key.get("task_id", ""))
            if task_id in task_ids and task_id not in cached_rows:
                cached_rows[task_id] = telemetry
                cached_paths[task_id] = (episode_directory, episode_directory / "episode-record.json")
        if set(cached_rows) == set(task_ids):
            request_accounting_source = "recovered_by_native_episode_cache"
            starts = {
                task_id: getattr(paths[0].stat(), "st_birthtime", paths[0].stat().st_ctime)
                for task_id, paths in cached_paths.items()
            }
            finishes = {
                task_id: getattr(paths[1].stat(), "st_birthtime", paths[1].stat().st_ctime)
                for task_id, paths in cached_paths.items()
            }
            origin = min(starts.values())
            reconstructed_intervals = [
                {
                    "task_id": task_id,
                    "started_monotonic_ns": round((starts[task_id] - origin) * 1_000_000_000),
                    "finished_monotonic_ns": round((finishes[task_id] - origin) * 1_000_000_000),
                }
                for task_id in task_ids
            ]

    audit_calls: list[dict[str, str]] = []
    audit_lock = threading.Lock()
    diagnostic_interrupts: list[dict[str, Any]] = []
    diagnostic_lock = threading.Lock()

    def audit_provider(simulation: Any, task: Any, customer: Any, service: Any, panel: str) -> Any:
        simulation_task_id = str(getattr(simulation, "task_id", ""))
        task_id = str(getattr(task, "id", ""))
        with audit_lock:
            audit_calls.append({
                "task_id": task_id,
                "simulation_task_id": simulation_task_id,
                "panel_name": panel,
            })
        return bundle.callbacks["audit_provider"](
            simulation, task, customer, service, panel,
        )

    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_root,
        request_budget=budget,
        audit_provider=audit_provider,
        output_directory=output_directory,
    )
    context = {
        "schema_version": 1,
        "purpose": "two-task incumbent-only concurrency engineering smoke",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "task_review_sha256": manifest.task_semantic_review_sha256,
        "tau_bench_commit": manifest.upstream_commit,
        "provider_provenance": bundle.provenance,
        "executed_tasks": list(task_ids),
        "customer_strategy": CustomerStrategy().to_dict(),
        "service_strategy": ServiceStrategy().to_dict(),
        "seed": manifest.seed,
        "max_concurrency": 2,
        "provider_retries": 0,
        "lifecycle": "no Customer evolution, Service proposal, repair, or gate",
        "request_budget_cap": manifest.request_budget_cap,
        "request_budget_mode": (
            "unbounded_diagnostic" if manifest.request_budget_cap is None else "capped"
        ),
    }
    context_path = output_directory / "concurrency-smoke-context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != context:
            raise ValueError("existing concurrency smoke context differs from this frozen request")
    else:
        _write_json_once(context_path, context)

    specs = tuple(
        EpisodeSpec(
            task_id=task_id,
            seed=manifest.seed,
            customer=CustomerStrategy(),
            service=ServiceStrategy(),
            panel_name="concurrency_engineering_smoke",
            generation=0,
        )
        for task_id in task_ids
    )
    intervals: list[dict[str, Any]] = []
    interval_lock = threading.Lock()

    def measured_runner(**kwargs: Any) -> EpisodeRecord:
        started = time.monotonic_ns()
        try:
            return runner(**kwargs)
        finally:
            finished = time.monotonic_ns()
            with interval_lock:
                intervals.append({
                    "task_id": str(kwargs["task_id"]),
                    "started_monotonic_ns": started,
                    "finished_monotonic_ns": finished,
                })

    initial_budget = budget.snapshot()
    if initial_budget.attempts or initial_budget.failures or initial_budget.in_flight:
        raise RuntimeError("provider callbacks unexpectedly made a request during setup")

    records: tuple[EpisodeRecord, ...] = ()
    batch_error: str | None = None
    original_run_with_budget = native_runner_module.run_with_budget

    def diagnostic_run_with_budget(orchestrator: Any, shared_budget: RequestBudget, **kwargs: Any) -> Any:
        try:
            return original_run_with_budget(orchestrator, shared_budget, **kwargs)
        except Exception as exc:
            chain = []
            current: BaseException | None = exc
            while current is not None and len(chain) < 4:
                chain.append({
                    "type": type(current).__name__,
                    "message": _redact_text(str(current)),
                })
                current = current.__cause__ or current.__context__
            task_id = str(getattr(getattr(orchestrator, "task", None), "id", "unknown"))
            trajectory = getattr(orchestrator, "trajectory", ()) or ()
            partial_messages = [_safe_json_value(message) for message in trajectory]
            diagnostic = {
                "schema_version": 1,
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "exception_chain": chain,
                "step_count": getattr(orchestrator, "step_count", None),
                "done": getattr(orchestrator, "done", None),
                "termination_reason": _safe_json_value(
                    getattr(orchestrator, "termination_reason", None)
                ),
                "partial_message_count": len(partial_messages),
                "partial_trajectory": partial_messages,
            }
            diagnostic_path = (
                output_directory / "diagnostics" / f"run-simulation-interrupt-task-{task_id}.json"
            )
            _write_json_once(diagnostic_path, diagnostic)
            with diagnostic_lock:
                diagnostic_interrupts.append({
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                    "diagnostic_artifact": diagnostic_path.relative_to(output_directory).as_posix(),
                    "partial_message_count": len(partial_messages),
                })
            raise

    native_runner_module.run_with_budget = diagnostic_run_with_budget
    try:
        records = run_episode_batch(
            measured_runner,
            specs,
            max_concurrency=2,
            request_budget=budget,
        )
    except Exception as exc:  # noqa: BLE001 - artifacts must not expose credential-bearing errors
        batch_error = type(exc).__name__
    finally:
        native_runner_module.run_with_budget = original_run_with_budget

    budget_before_cache = budget.snapshot()
    cache_rows: list[dict[str, Any]] = []
    if not batch_error and len(records) == len(specs):
        for spec, record in zip(specs, records, strict=True):
            kwargs = spec.runner_kwargs()
            is_cached = runner.has_completed_episode(**kwargs)
            cached_record = runner(**kwargs)
            cache_rows.append({
                "task_id": spec.task_id,
                "cache_reported_complete": is_cached,
                "same_record": cached_record.to_dict() == record.to_dict(),
            })
    budget_after_cache = budget.snapshot()

    interval_timing_source = "process_monotonic_clock"
    if reconstructed_intervals:
        intervals = reconstructed_intervals
        interval_timing_source = "episode_directory_and_record_creation_times"
    intervals.sort(key=lambda row: row["task_id"])
    has_overlap = (
        len(intervals) == 2
        and max(row["started_monotonic_ns"] for row in intervals)
        < min(row["finished_monotonic_ns"] for row in intervals)
    )
    episode_directories = sorted((output_directory / "episodes").glob("*"))
    saved_records: list[dict[str, Any]] = []
    trace_statuses: dict[str, str] = {}
    reviewer_outputs_bound = True
    artifact_integrity = True
    for directory in episode_directories:
        record_path = directory / "episode-record.json"
        simulation_path = directory / "native-simulation.json"
        telemetry_path = directory / "run-telemetry.json"
        trace_path = directory / "db-state-trace.json"
        required = (record_path, simulation_path, telemetry_path, trace_path)
        if not all(path.is_file() for path in required) or (directory / "incomplete-run.json").exists():
            artifact_integrity = False
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        task_id = str(record.get("task_id", ""))
        raw_review = record.get("raw_review") or {}
        reviewer_outputs_bound = reviewer_outputs_bound and (
            task_id == str(simulation.get("task_id", ""))
            and record.get("episode_id") == simulation.get("id")
            and isinstance(raw_review.get("native_review"), dict)
            and isinstance(raw_review.get("auth_classification"), dict)
            and record.get("trajectory_ref") == simulation_path.relative_to(output_directory).as_posix()
            and telemetry.get("episode_key", {}).get("task_id") == task_id
        )
        trace_statuses[task_id] = str(trace.get("status", "unavailable"))
        saved_records.append({
            "task_id": task_id,
            "episode_id": record.get("episode_id"),
            "status": record.get("status"),
            "native_reward": record.get("native_reward"),
            "tool_calls": record.get("tool_calls"),
            "audit_ref": record.get("audit_ref"),
            "independent_audit": telemetry.get("independent_audit"),
            "simulation_task_id": simulation.get("task_id"),
            "simulation_sha256": hashlib.sha256(simulation_path.read_bytes()).hexdigest(),
            "review_sha256": hashlib.sha256(
                json.dumps(raw_review, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "trace_sha256": trace.get("trace_sha256"),
        })

    if not audit_calls:
        for row in saved_records:
            audit = row.get("independent_audit")
            if row.get("audit_ref") and isinstance(audit, dict):
                audit_calls.append({
                    "task_id": row["task_id"],
                    "simulation_task_id": row["simulation_task_id"],
                    "panel_name": "recovered_from_episode_telemetry",
                })
    audit_bound = len(saved_records) == len(task_ids) and all(
        row.get("task_id") == row.get("simulation_task_id")
        and row.get("audit_ref") == (row.get("independent_audit") or {}).get("verifier_ref")
        and isinstance(row.get("independent_audit"), dict)
        for row in saved_records
    )
    final_budget = budget.snapshot()
    checks = {
        "two_reviewed_tasks": len(saved_records) == 2
        and [row.task_id for row in records] == list(task_ids),
        "actual_episode_overlap": has_overlap,
        "one_shared_budget_accounting": final_budget.cap == manifest.request_budget_cap
        and (final_budget.cap is None or final_budget.attempts <= final_budget.cap)
        and final_budget.in_flight == 0
        and final_budget.reserved == 0,
        "zero_provider_failures": final_budget.failures == 0 and final_budget.denied == 0,
        "independent_artifacts": artifact_integrity
        and len(episode_directories) == 2
        and len({row["episode_id"] for row in saved_records}) == 2,
        "episode_cache": len(cache_rows) == 2
        and all(row["cache_reported_complete"] and row["same_record"] for row in cache_rows)
        and budget_after_cache.attempts == budget_before_cache.attempts,
        "db_state_trace": len(trace_statuses) == 2
        and all(status == "complete" for status in trace_statuses.values()),
        "reviewer_outputs_bound_to_trajectory": reviewer_outputs_bound,
        "independent_audits_bound_to_task": audit_bound,
        "deterministic_aggregate_order": [row.task_id for row in records] == list(task_ids),
        "checkpoint_integrity": not (
            (output_directory / "checkpoint.json").exists()
            or (project_root / manifest.checkpoint_path).exists()
        ),
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) and batch_error is None else "failed",
        "experiment_id": manifest.experiment_id,
        "phase": "activation_concurrency_engineering_smoke",
        "manifest_sha256": manifest.sha256,
        "run_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "task_review_sha256": manifest.task_semantic_review_sha256,
        "task_ids": list(task_ids),
        "request_accounting_source": request_accounting_source,
        "customer_strategy": CustomerStrategy().to_dict(),
        "service_strategy": ServiceStrategy().to_dict(),
        "seed": manifest.seed,
        "max_concurrency": 2,
        "provider_retries": 0,
        "episode_intervals": intervals,
        "interval_timing_source": interval_timing_source,
        "overlap_observed": has_overlap,
        "cache_checks": cache_rows,
        "audit_calls": audit_calls,
        "diagnostic_interrupts": diagnostic_interrupts,
        "episodes": saved_records,
        "db_trace_statuses": trace_statuses,
        "checks": checks,
        "provider_budget": final_budget.to_dict(),
        "provider_failures_429_or_5xx": 0 if final_budget.failures == 0 else "inspect provider error logs",
        "checkpoint_status": "not used by this episode-only engineering smoke",
        "lifecycle_scope": "no evolution, Service repair, gate, or validation/H scheduling",
        "batch_error_type": batch_error,
    }
    result_path = output_directory / "concurrency-smoke-result.json"
    _write_json_once(result_path, result)
    _write_sums(output_directory)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, default=os.environ.get("TAU2_DATA_DIR"))
    parser.add_argument("--provider-plugin", required=True)
    args = parser.parse_args(argv)
    try:
        if args.tau2_data_dir is None:
            raise ValueError("provide --tau2-data-dir or set TAU2_DATA_DIR")
        result = run_smoke(
            args.config,
            tau2_data_dir=args.tau2_data_dir,
            provider_plugin=args.provider_plugin,
        )
    except Exception as exc:  # noqa: BLE001 - avoid printing provider credentials
        print(f"Concurrency smoke failed ({type(exc).__name__}); inspect immutable artifacts", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
