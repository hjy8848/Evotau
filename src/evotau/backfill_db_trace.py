"""Backfill observational DB State Trace files for immutable native episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .db_state_trace import (
    DBStateTraceError,
    _verify_pinned_sources,
    replay_db_state_trace,
    unavailable_trace,
    write_trace_once,
)


def _manifest_task_ids(manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("evolution_task_id", "validation_task_id", "task_id"):
        value = manifest.get(key)
        if isinstance(value, (str, int)):
            result.add(str(value))
    for key in (
        "evolution_task_ids",
        "validation_task_ids",
        "heldout_task_ids",
        "task_ids",
    ):
        values = manifest.get(key)
        if isinstance(values, list):
            result.update(str(value) for value in values)
    return result


def backfill_run(
    run_dir: str | Path,
    *,
    tau2_data_dir: str | Path,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Add missing trace artifacts while leaving every original artifact untouched."""

    unresolved_run_path = Path(run_dir).expanduser()
    if unresolved_run_path.is_symlink():
        raise ValueError("run directory unavailable")
    run_path = unresolved_run_path.resolve()
    data_root = Path(tau2_data_dir).expanduser().resolve()
    manifest_path = run_path / "manifest.json"
    if (
        run_path.is_symlink()
        or not run_path.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise ValueError("run manifest unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_pinned_sources(manifest, data_root)
    registered_ids = _manifest_task_ids(manifest)
    requested_ids = registered_ids if task_id is None else {str(task_id)}
    if not requested_ids or not requested_ids <= registered_ids:
        raise ValueError("requested task is not registered in the frozen run manifest")

    tasks_path = data_root / "tau2/domains/retail/tasks.json"
    split_path = data_root / "tau2/domains/retail/split_tasks.json"
    tasks_raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    split_raw = json.loads(split_path.read_text(encoding="utf-8"))
    task_rows = {str(row.get("id")): row for row in tasks_raw if isinstance(row, dict)}
    official_ids = {str(item) for values in split_raw.values() for item in values}
    if not requested_ids <= task_rows.keys() or not requested_ids <= official_ids:
        raise ValueError("manifest task is absent from the pinned official task data")

    # τ-bench data path constants are read at import time. The trace constructor
    # loads its RetailDB from this explicit data root, but setting the standard
    # variable also keeps any upstream utility consistent for this process.
    os.environ["TAU2_DATA_DIR"] = str(data_root)
    from tau2.data_model.tasks import Task

    task_models = {key: Task.model_validate(task_rows[key]) for key in requested_ids}
    episodes_dir = run_path / "episodes"
    if episodes_dir.is_symlink() or not episodes_dir.is_dir():
        raise ValueError("native episode directory unavailable")

    started = time.perf_counter()
    generated = unavailable = skipped = event_count = field_count = total_bytes = 0
    generation_seconds = 0.0
    for episode_dir in sorted(episodes_dir.iterdir(), key=lambda path: path.name):
        if episode_dir.is_symlink() or not episode_dir.is_dir():
            raise ValueError("episode store contains an unsafe entry")
        trajectory_path = episode_dir / "native-simulation.json"
        record_path = episode_dir / "episode-record.json"
        trace_path = episode_dir / "db-state-trace.json"
        if not trajectory_path.exists() or not record_path.exists():
            continue
        if (
            trajectory_path.is_symlink()
            or record_path.is_symlink()
            or not trajectory_path.is_file()
            or not record_path.is_file()
        ):
            raise ValueError("episode source artifact is unsafe")
        if trace_path.exists():
            skipped += 1
            continue
        trajectory_bytes = trajectory_path.read_bytes()
        trajectory = json.loads(trajectory_bytes)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        current_task = str(trajectory.get("task_id"))
        if current_task not in requested_ids:
            continue
        if (
            record.get("status") != "complete"
            or str(record.get("task_id")) != current_task
            or str(record.get("episode_id")) != str(trajectory.get("id"))
        ):
            continue
        t0 = time.perf_counter()
        try:
            trace = replay_db_state_trace(
                manifest=manifest,
                task=task_models[current_task],
                trajectory=trajectory,
                trajectory_bytes=trajectory_bytes,
                data_dir=data_root,
            )
        except DBStateTraceError as exc:
            trace = unavailable_trace(
                task_id=current_task,
                episode_id=str(trajectory.get("id", episode_dir.name)),
                manifest=manifest,
                trajectory_bytes=trajectory_bytes,
                reason_code=exc.reason_code,
            )
        except Exception:  # noqa: BLE001 - trace-only failure must remain non-fatal
            trace = unavailable_trace(
                task_id=current_task,
                episode_id=str(trajectory.get("id", episode_dir.name)),
                manifest=manifest,
                trajectory_bytes=trajectory_bytes,
                reason_code="trace_not_generated",
            )
        try:
            write_trace_once(trace_path, trace)
        except (OSError, TypeError, ValueError):
            # A failed sidecar write must never invalidate the source episode.
            unavailable += 1
            continue
        generation_seconds += time.perf_counter() - t0
        # Generation timing stays outside the deterministic trace artifact.
        total_bytes += trace_path.stat().st_size
        if trace.get("status") == "complete":
            generated += 1
            summary = trace.get("summary", {})
            event_count += int(summary.get("mutation_event_count", 0))
            field_count += int(summary.get("field_change_count", 0))
        else:
            unavailable += 1
    return {
        "run_dir": str(run_path),
        "task_ids": sorted(requested_ids),
        "generated": generated,
        "unavailable": unavailable,
        "skipped_existing": skipped,
        "mutation_event_count": event_count,
        "field_change_count": field_count,
        "trace_bytes": total_bytes,
        "trace_generation_seconds": round(generation_seconds, 6),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill deterministic DB State Trace sidecars."
    )
    parser.add_argument("run_dir", type=Path, help="completed native run directory")
    parser.add_argument(
        "--tau2-data-dir",
        type=Path,
        default=os.environ.get("EVOTAU_TAU2_DATA_DIR")
        or os.environ.get("TAU2_DATA_DIR"),
        help="pinned τ-bench data root (or EVOTAU_TAU2_DATA_DIR / TAU2_DATA_DIR)",
    )
    parser.add_argument("--task-id", help="backfill only this task ID")
    args = parser.parse_args(argv)
    if args.tau2_data_dir is None:
        parser.error("--tau2-data-dir or TAU2_DATA_DIR is required")
    try:
        result = backfill_run(
            args.run_dir, tau2_data_dir=args.tau2_data_dir, task_id=args.task_id
        )
    except (DBStateTraceError, ValueError, OSError, json.JSONDecodeError):
        print(
            "DB trace backfill stopped: frozen run or pinned data could not be verified.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
