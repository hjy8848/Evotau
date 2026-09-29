"""Deterministic, observational reconstruction of pinned τ-bench DB mutations.

The trace is a display artifact only. It never supplies task reward or any
research decision, and its replay path constructs no provider-backed objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import (
    TAU2_PACKAGE_VERSION,
    TAU_BENCH_COMMIT,
    TAU_BENCH_REPOSITORY,
    canonical_json,
    sha256_json,
    verify_git_blob_sha1,
)

SCHEMA_VERSION = 1
_RETAIL_TABLES = {"orders": "order", "products": "product", "users": "user"}
_SECRET_KEY = re.compile(
    r"api[_-]?key|token|secret|password|authorization|credential",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"(?i)\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}")


class DBStateTraceError(RuntimeError):
    """A fail-closed trace error with a safe, stable artifact reason code."""

    def __init__(self, reason_code: str):
        if reason_code not in {
            "pinned_environment_unavailable",
            "trajectory_replay_failed",
            "unsupported_domain",
            "state_serialization_failed",
            "tool_replay_mismatch",
            "trace_not_generated",
        }:
            reason_code = "trace_not_generated"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    document = dict(manifest)
    recorded = document.pop("manifest_sha256", None)
    actual = sha256_json(document)
    if not isinstance(recorded, str) or recorded != actual:
        raise DBStateTraceError("pinned_environment_unavailable")
    upstream = document.get("upstream")
    if not isinstance(upstream, Mapping) or (
        upstream.get("repository"),
        upstream.get("commit"),
        upstream.get("package_version"),
    ) != (TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION):
        raise DBStateTraceError("pinned_environment_unavailable")
    if document.get("domain") != "retail":
        raise DBStateTraceError("unsupported_domain")
    return recorded


def _verify_pinned_sources(manifest: Mapping[str, Any], data_dir: Path) -> str:
    """Verify every manifest source blob and the installed τ-bench Git pin."""

    fingerprint = _manifest_fingerprint(manifest)
    try:
        installed = distribution("tau2")
        if installed.version != TAU2_PACKAGE_VERSION:
            raise ValueError
        direct_url = installed.read_text("direct_url.json")
        source = json.loads(direct_url or "{}")
        commit = source["vcs_info"]["commit_id"]
        if commit != TAU_BENCH_COMMIT:
            raise ValueError
        import tau2

        package_root = Path(tau2.__file__).resolve().parent
        blobs = manifest.get("source_blob_sha1")
        if not isinstance(blobs, Mapping) or not blobs:
            raise ValueError
        for repository_path, digest in sorted(blobs.items()):
            if not isinstance(repository_path, str) or not isinstance(digest, str):
                raise TypeError
            relative = PurePosixPath(repository_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            if repository_path.startswith("data/"):
                path = data_dir / repository_path.removeprefix("data/")
            elif repository_path.startswith("src/tau2/"):
                path = package_root / repository_path.removeprefix("src/tau2/")
            else:
                raise ValueError
            verify_git_blob_sha1(path, digest)
        for required in (
            "data/tau2/domains/retail/db.json",
            "data/tau2/domains/retail/tasks.json",
            "data/tau2/domains/retail/split_tasks.json",
            "src/tau2/domains/retail/tools.py",
            "src/tau2/environment/environment.py",
        ):
            if required not in blobs:
                raise ValueError
    except (
        PackageNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        ImportError,
        json.JSONDecodeError,
    ) as exc:
        raise DBStateTraceError("pinned_environment_unavailable") from exc
    return fingerprint


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        # Round-tripping both enforces JSON values and removes model/runtime objects.
        return json.loads(
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise DBStateTraceError("state_serialization_failed") from exc


def _state_snapshot(environment: Any) -> dict[str, Any]:
    try:
        db = environment.tools.db
        state = _json_value(db)
        if not isinstance(state, dict) or not _RETAIL_TABLES.keys() <= state.keys():
            raise ValueError
        if any(not isinstance(state[table], dict) for table in _RETAIL_TABLES):
            raise ValueError
        return {table: state[table] for table in sorted(_RETAIL_TABLES)}
    except DBStateTraceError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise DBStateTraceError("state_serialization_failed") from exc


def _flatten_entity(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping) and value:
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_entity(value[key], path))
    else:
        result[prefix or "__entity__"] = value
    return result


def _state_hash(environment: Any) -> str:
    try:
        value = environment.get_db_hash()
        if not isinstance(value, str) or not value:
            raise ValueError
        return value
    except (AttributeError, TypeError, ValueError) as exc:
        raise DBStateTraceError("state_serialization_failed") from exc


def _changes(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[dict[str, Any]]:
    changes = []
    for table in sorted(_RETAIL_TABLES):
        old_rows, new_rows = before[table], after[table]
        for entity_id in sorted(set(old_rows) | set(new_rows), key=str):
            old_fields = (
                _flatten_entity(old_rows[entity_id]) if entity_id in old_rows else {}
            )
            new_fields = (
                _flatten_entity(new_rows[entity_id]) if entity_id in new_rows else {}
            )
            for field_path in sorted(set(old_fields) | set(new_fields)):
                old = old_fields.get(field_path)
                new = new_fields.get(field_path)
                if old != new:
                    changes.append(
                        {
                            "entity_type": _RETAIL_TABLES[table],
                            "entity_id": str(entity_id),
                            "field_path": field_path,
                            "before": old,
                            "after": new,
                        }
                    )
    return changes


def _redact_arguments(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(k): _redact_arguments(v, str(k))
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_redact_arguments(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def _tool_result_map(messages: list[Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in messages:
        candidates = (
            [message]
            if getattr(message, "role", None) == "tool"
            else getattr(message, "tool_messages", None) or []
        )
        for row in candidates:
            if getattr(row, "role", None) != "tool":
                continue
            if not row.id or row.id in results:
                raise DBStateTraceError("trajectory_replay_failed")
            results[row.id] = row
    return results


def _apply_history(
    environment: Any, messages: list[Any], expected_results: Mapping[str, Any]
) -> None:
    """Apply initial-state history using the pinned environment semantics."""

    for message in messages:
        if not getattr(message, "is_tool_call", lambda: False)():
            continue
        for call in message.tool_calls:
            if not call.id or call.id not in expected_results:
                raise DBStateTraceError("trajectory_replay_failed")
            if not environment._has_tool(call.name):
                continue  # Upstream Environment.set_state treats hallucinations as no-op.
            if not environment._is_mutating_tool(call.name):
                continue
            observed = environment.get_response(call)
            expected = expected_results[call.id]
            if observed.error != expected.error or _parse_tool_content(
                observed.content
            ) != _parse_tool_content(expected.content):
                raise DBStateTraceError("tool_replay_mismatch")


def _parse_tool_content(content: str) -> Any:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


def capture_mutation_events(
    environment: Any,
    messages: list[Any],
    *,
    initial_state: Any = None,
) -> dict[str, Any]:
    """Replay typed trajectory messages against one fresh upstream environment."""

    try:
        environment.set_state(
            initialization_data=None
            if initial_state is None
            else initial_state.initialization_data,
            initialization_actions=None
            if initial_state is None
            else initial_state.initialization_actions,
            message_history=[],
        )
        result_by_id = _tool_result_map(messages)
        initial_history = (
            []
            if initial_state is None or initial_state.message_history is None
            else list(initial_state.message_history)
        )
        if initial_history:
            if len(messages) < len(initial_history) or any(
                not _same_history_message(expected, actual)
                for expected, actual in zip(
                    initial_history, messages[: len(initial_history)]
                )
            ):
                raise DBStateTraceError("trajectory_replay_failed")
            _apply_history(environment, initial_history, result_by_id)
        initial_hash = _state_hash(environment)
        state = _state_snapshot(environment)
        events: list[dict[str, Any]] = []
        write_tool_call_count = 0
        for message in messages[len(initial_history) :]:
            if not getattr(message, "is_tool_call", lambda: False)():
                continue
            for call in message.tool_calls:
                if not call.id or call.id not in result_by_id:
                    raise DBStateTraceError("trajectory_replay_failed")
                if not environment._has_tool(call.name):
                    continue
                if not environment._is_mutating_tool(call.name):
                    continue
                write_tool_call_count += 1
                before = state
                observed = environment.get_response(call)
                expected = result_by_id[call.id]
                if observed.error != expected.error or _parse_tool_content(
                    observed.content
                ) != _parse_tool_content(expected.content):
                    raise DBStateTraceError("tool_replay_mismatch")
                state = _state_snapshot(environment)
                changes = _changes(before, state)
                if changes:
                    events.append(
                        {
                            "event_index": len(events),
                            "turn_idx": message.turn_idx,
                            "tool_call_id": call.id,
                            "requestor": call.requestor,
                            "tool_name": call.name,
                            "arguments": _redact_arguments(call.arguments),
                            "changes": changes,
                        }
                    )
        return {
            "initial_state_hash": initial_hash,
            "actual_final_state_hash": _state_hash(environment),
            "events": events,
            "write_tool_call_count": write_tool_call_count,
        }
    except DBStateTraceError:
        raise
    except Exception as exc:
        raise DBStateTraceError("trajectory_replay_failed") from exc


def _same_history_message(left: Any, right: Any) -> bool:
    left_dump, right_dump = left.model_dump(mode="json"), right.model_dump(mode="json")
    for item in (left_dump, right_dump):
        item.pop("timestamp", None)
        item.pop("turn_idx", None)
    return left_dump == right_dump


def _fresh_retail_environment(data_root: Path) -> Any:
    """Create a clean environment from the already verified, pinned Retail DB."""

    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.tools import RetailTools
    from tau2.environment.environment import Environment

    db_path = data_root / "tau2/domains/retail/db.json"
    database = RetailDB.model_validate_json(db_path.read_text(encoding="utf-8"))
    return Environment(domain_name="retail", policy="", tools=RetailTools(database))


def replay_db_state_trace(
    *,
    manifest: Mapping[str, Any],
    task: Any,
    trajectory: Mapping[str, Any],
    trajectory_bytes: bytes,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Reconstruct field-level DB changes with the exact pinned τ-bench tools.

    This imports only τ-bench data/environment/tool classes. It never builds an
    agent, user, evaluator, reviewer, or provider client.
    """

    data_root = Path(data_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise DBStateTraceError("pinned_environment_unavailable")
    manifest_sha = _verify_pinned_sources(manifest, data_root)
    if str(trajectory.get("task_id")) != str(task.id) or not trajectory.get("id"):
        raise DBStateTraceError("trajectory_replay_failed")
    trajectory_sha = hashlib.sha256(trajectory_bytes).hexdigest()

    try:
        from tau2.data_model.simulation import SimulationRun
        from tau2.data_model.tasks import Task

        simulation = SimulationRun.model_validate(dict(trajectory))
        if simulation.messages is None:
            raise ValueError
        task_model = task if isinstance(task, Task) else Task.model_validate(task)
        environment = _fresh_retail_environment(data_root)
        initial = task_model.initial_state
        messages = list(simulation.messages)
        replay = capture_mutation_events(environment, messages, initial_state=initial)
        events = replay["events"]
    except DBStateTraceError:
        raise
    except Exception as exc:
        raise DBStateTraceError("trajectory_replay_failed") from exc

    reward_info = trajectory.get("reward_info")
    db_check = reward_info.get("db_check") if isinstance(reward_info, Mapping) else None
    db_match = (
        db_check.get("db_match")
        if isinstance(db_check, Mapping) and type(db_check.get("db_match")) is bool
        else None
    )
    gold_state = None
    gold_hash = None
    if task_model.evaluation_criteria is not None:
        try:
            # This follows pinned EnvironmentEvaluator's reference-state path:
            # task initial data/actions/history plus evaluation_criteria.actions.
            # The code never calls EnvironmentEvaluator or an evaluator model.
            gold_environment = _fresh_retail_environment(data_root)
            initial = task_model.initial_state
            initial_history = (
                []
                if initial is None or initial.message_history is None
                else list(initial.message_history)
            )
            gold_environment.set_state(
                initialization_data=None
                if initial is None
                else initial.initialization_data,
                initialization_actions=None
                if initial is None
                else initial.initialization_actions,
                message_history=initial_history,
            )
            for action in task_model.evaluation_criteria.actions or ():
                gold_environment.make_tool_call(
                    action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            gold_state = _state_snapshot(gold_environment)
            gold_hash = _state_hash(gold_environment)
        except Exception:  # noqa: BLE001 - gold comparison is optional and fail-closed
            # Expected-state reconstruction is optional. Never discard an
            # otherwise valid actual trace when the reference path fails.
            gold_state = None
            gold_hash = None
    gold_differences = None
    gold_state_available = gold_state is not None
    if gold_state_available:
        reconstructed_match = replay["actual_final_state_hash"] == gold_hash
        # Keep τ-bench's recorded match authoritative. If reconstruction
        # disagrees with the native result, suppress field claims as unsafe.
        if db_match is None or db_match == reconstructed_match:
            gold_differences = [
                {
                    "entity_type": row["entity_type"],
                    "entity_id": row["entity_id"],
                    "field_path": row["field_path"],
                    "actual": row["before"],
                    "expected": row["after"],
                }
                for row in _changes(_state_snapshot(environment), gold_state)
            ]
        else:
            gold_state_available = False
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(task_model.id),
        "episode_id": str(simulation.id),
        "trajectory_ref": "native-simulation.json",
        "source": "deterministic_replay",
        "status": "complete",
        "provenance": {
            "manifest_sha256": manifest_sha,
            "tau_bench_repository": TAU_BENCH_REPOSITORY,
            "tau_bench_commit": TAU_BENCH_COMMIT,
            "tau_bench_package_version": TAU2_PACKAGE_VERSION,
            "trajectory_sha256": trajectory_sha,
        },
        "summary": {
            "initial_state_hash": replay["initial_state_hash"],
            "actual_final_state_hash": replay["actual_final_state_hash"],
            "gold_final_state_hash": gold_hash if gold_state_available else None,
            "db_match": db_match,
            "write_tool_call_count": replay["write_tool_call_count"],
            "mutation_event_count": len(events),
            "field_change_count": sum(len(event["changes"]) for event in events),
            "final_comparison": {
                "db_match": db_match,
                "gold_state_available": gold_state_available,
                "differences": gold_differences,
            },
        },
        "events": events,
    }
    document["trace_sha256"] = hashlib.sha256(
        canonical_json(document).encode("utf-8")
    ).hexdigest()
    return document


def unavailable_trace(
    *,
    task_id: str,
    episode_id: str,
    manifest: Mapping[str, Any] | None,
    trajectory_bytes: bytes | None,
    reason_code: str = "trace_not_generated",
) -> dict[str, Any]:
    """Build an artifact-safe status record without exception text."""

    reason = (
        reason_code
        if reason_code
        in {
            "pinned_environment_unavailable",
            "trajectory_replay_failed",
            "unsupported_domain",
            "state_serialization_failed",
            "tool_replay_mismatch",
            "trace_not_generated",
        }
        else "trace_not_generated"
    )
    upstream = manifest.get("upstream", {}) if isinstance(manifest, Mapping) else {}
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(task_id),
        "episode_id": str(episode_id),
        "trajectory_ref": "native-simulation.json",
        "source": "deterministic_replay",
        "status": "unavailable",
        "reason_code": reason,
        "provenance": {
            "manifest_sha256": manifest.get("manifest_sha256")
            if isinstance(manifest, Mapping)
            else None,
            "tau_bench_commit": upstream.get("commit")
            if isinstance(upstream, Mapping)
            else None,
            "trajectory_sha256": None
            if trajectory_bytes is None
            else hashlib.sha256(trajectory_bytes).hexdigest(),
        },
        "events": [],
    }
    document["trace_sha256"] = hashlib.sha256(
        canonical_json(document).encode("utf-8")
    ).hexdigest()
    return document


def write_trace_once(path: str | Path, trace: Mapping[str, Any]) -> None:
    """Write one deterministic trace without ever replacing an existing artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes a complete artifact and fails
        # atomically if another worker already created the destination.
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
