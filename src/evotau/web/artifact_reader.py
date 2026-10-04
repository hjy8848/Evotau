"""Read current Phase 0 and alternating-run artifacts for the local Console."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from ..records import EpisodeRecord
from ..tau_provenance import sha256_json

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_SECRET_KEY = re.compile(
    r"(api[_-]?key|token|secret|password|authorization|credential|session[_-]?id)", re.IGNORECASE,
)
_SECRET_TEXT = re.compile(
    r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+|\b(sk-[A-Za-z0-9_-]{12,})"
    r"|\b((?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*)[^\s,;]+"
)


class ArtifactReadError(ValueError):
    """An artifact is missing, malformed, unsafe, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    experiment_id: str
    phase: str
    status: str
    real_provider_enabled: bool
    manifest_sha256: str
    started_at: str
    provider_attempts: int | None
    provider_attempt_cap: int | None
    episode_count: int
    generation_count: int
    heldout_sealed: bool
    path: Path


class ArtifactReader:
    """Read only the current run schema; prior experiment directories stay archival."""

    def __init__(self, runs_root: str | Path, *, project_root: str | Path | None = None):
        self.runs_root = Path(runs_root).expanduser().absolute()
        self.project_root = Path(project_root or self.runs_root.parent.parent).expanduser().absolute()

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        if not self.runs_root.exists():
            return ()
        if self.runs_root.is_symlink() or not self.runs_root.is_dir():
            raise ArtifactReadError("run directory is not a regular directory")
        result = []
        for manifest_path in sorted(self.runs_root.glob("*/manifest.json")):
            run_path = manifest_path.parent
            if run_path.is_symlink() or not run_path.is_dir():
                raise ArtifactReadError("run directory contains a symlink or non-directory")
            manifest = self._read_json(manifest_path)
            if manifest.get("phase") in {"0-integration-proof", "alternating-self-evolution"}:
                result.append(self._summary_for(run_path))
        return tuple(sorted(result, key=lambda row: row["started_at"], reverse=True))

    def get_run(self, run_id: str, *, live_status: str | None = None) -> dict[str, Any]:
        try:
            return self._get_run_impl(run_id, live_status=live_status)
        except FileNotFoundError:
            raise
        except ArtifactReadError:
            raise
        except (KeyError, TypeError, AttributeError, OSError, ValueError, IndexError, OverflowError) as exc:
            raise ArtifactReadError("artifact is malformed or inconsistent and has been hidden") from exc

    def _get_run_impl(self, run_id: str, *, live_status: str | None = None) -> dict[str, Any]:
        run_path = self._resolve_run_path(run_id)
        manifest_path = run_path / "manifest.json"
        manifest = self._read_json(manifest_path)
        self._verify_manifest(manifest)
        phase = manifest.get("phase")
        if phase not in {"0-integration-proof", "alternating-self-evolution"}:
            raise ArtifactReadError("this archived experiment uses an unsupported Console schema")
        result_name = "phase0-result.json" if phase == "0-integration-proof" else "alternating-result.json"
        result_path = run_path / result_name
        result = self._read_json(result_path) if result_path.exists() else None
        if result is not None:
            self._verify_result_manifest(result, manifest)
            if result.get("experiment_id") not in (None, manifest.get("experiment_id")):
                raise ArtifactReadError("result and manifest experiment IDs differ; run is hidden")

        heldout_ids = self._heldout_ids(manifest)
        complete = result is not None and result.get("status") == "complete"
        if result is not None and result.get("status") == "incomplete":
            status = "failed"
        elif complete:
            status = "complete"
        elif live_status in {"running", "paused", "failed"}:
            status = live_status
        else:
            status = "incomplete" if self._has_resume_artifacts(run_path, phase) else "not_started"

        heldout_sealed = bool(heldout_ids) and not complete
        episodes = self._load_episodes(run_path, manifest, result, complete=complete)
        generations = self._generation_commits(manifest, result)
        strategies = self._load_strategies(result, generations)
        latest = max(generations, key=lambda item: item.get("generation", -1), default=None)
        current_customer_id = None if latest is None else latest.get("customer_after", {}).get("strategy_id")
        current_service_id = None if latest is None else latest.get("service_after", {}).get("strategy_id")
        current_customer = strategies["customer"].get(str(current_customer_id))
        current_service = strategies["service"].get(str(current_service_id))
        if result and result.get("final_customer"):
            current_customer = result["final_customer"]
            current_customer_id = _strategy_id(current_customer)
        if result and result.get("final_service"):
            current_service = result["final_service"]
            current_service_id = _strategy_id(current_service)
        if phase == "0-integration-proof" and result:
            strategy = result.get("strategy") or {}
            current_customer = strategy.get("customer")
            current_service = strategy.get("service")
            current_customer_id = _strategy_id(current_customer) if current_customer else None
            current_service_id = _strategy_id(current_service) if current_service else None
        budget = self._budget_summary(manifest, result, run_path)
        return {
            "run_id": run_id,
            "path": run_path,
            "experiment_id": str(manifest.get("experiment_id", run_path.name)),
            "phase": phase,
            "status": status,
            "episode_limit": _manifest_int(manifest, "max_episodes"),
            "max_concurrency": 1,
            "manifest": _safe_manifest_for_display(manifest, sealed=heldout_sealed),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "real_provider_enabled": manifest.get("real_provider_enabled") is True,
            "upstream_commit": _upstream_commit(manifest),
            "started_at": self._timestamp(manifest_path),
            "result": None if heldout_sealed else redact_secrets(result),
            "budget": budget,
            "episodes": episodes,
            "generation_commits": redact_secrets(generations),
            "current_generation": (
                max((item.get("generation", -1) for item in generations), default=-1) + 1
                if status in {"running", "paused", "incomplete"} else None
            ),
            "current_episode": self._active_episode(run_path, status, heldout_sealed),
            "current_customer_id": current_customer_id,
            "current_service_id": current_service_id,
            "current_customer_strategy": redact_secrets(current_customer),
            "current_service_strategy": redact_secrets(current_service),
            "strategies": redact_secrets(strategies),
            "heldout_sealed": heldout_sealed,
            "heldout_task_count": len(heldout_ids),
            "raw_artifacts": redact_secrets({
                "manifest": _safe_manifest_for_display(manifest, sealed=heldout_sealed),
                "result": None if heldout_sealed else result,
            }),
        }

    def episode(self, run_id: str, episode_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        for episode in run["episodes"]:
            if episode["episode_id"] == episode_id:
                return {**episode, "run": run}
        raise FileNotFoundError("episode artifact is unavailable")

    def _summary_for(self, run_path: Path) -> dict[str, Any]:
        run_id = run_path.relative_to(self.runs_root).as_posix()
        run = self.get_run(run_id)
        return {
            "run_id": run_id,
            "experiment_id": run["experiment_id"],
            "phase": run["phase"],
            "status": run["status"],
            "real_provider_enabled": run["real_provider_enabled"],
            "manifest_sha256": run["manifest_sha256"],
            "started_at": run["started_at"],
            "provider_attempts": run["budget"].get("attempts"),
            "provider_attempt_cap": run["budget"].get("cap"),
            "episode_count": len(run["episodes"]),
            "generation_count": len(run["generation_commits"]),
            "heldout_sealed": run["heldout_sealed"],
            "path": run_path,
        }

    def _resolve_run_path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not run_id.strip() or "\\" in run_id:
            raise FileNotFoundError("run is unavailable")
        candidate = self.runs_root / run_id
        _reject_symlink_components(self.runs_root, candidate)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.runs_root)
        except ValueError as exc:
            raise FileNotFoundError("run is unavailable") from exc
        if not resolved.is_dir():
            raise FileNotFoundError("run is unavailable")
        return resolved

    @staticmethod
    def _read_json(path: Path, *, expected: type = dict) -> Any:
        if path.is_symlink() or not path.is_file():
            raise ArtifactReadError("artifact is not a regular file")
        try:
            if path.stat().st_size > _MAX_ARTIFACT_BYTES:
                raise ArtifactReadError("artifact exceeds the local viewer size limit")
            value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
        except ArtifactReadError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ArtifactReadError("artifact is malformed and has been hidden") from exc
        if not isinstance(value, expected):
            raise ArtifactReadError("artifact has an unexpected top-level shape")
        return value

    @staticmethod
    def _verify_manifest(manifest: dict[str, Any]) -> None:
        digest = manifest.get("manifest_sha256")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if not isinstance(digest, str) or sha256_json(payload) != digest:
            raise ArtifactReadError("manifest fingerprint is invalid; run is hidden")
        if not manifest.get("experiment_id"):
            raise ArtifactReadError("manifest has no experiment ID")

    @staticmethod
    def _verify_result_manifest(result: dict[str, Any], manifest: dict[str, Any]) -> None:
        if (result.get("schema_version") != 1
                or result.get("manifest_sha256") != manifest.get("manifest_sha256")
                or result.get("status") not in {"complete", "incomplete"}):
            raise ArtifactReadError("result does not match the frozen manifest schema")

    @staticmethod
    def _heldout_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
        panels = manifest.get("task_panels")
        if isinstance(panels, dict):
            return tuple(str(item) for item in panels.get("H", ()))
        selection = manifest.get("task_selection", {})
        if not isinstance(selection, dict):
            raise ArtifactReadError("manifest task selection is malformed")
        return tuple(str(item) for item in selection.get("heldout", ()))

    @staticmethod
    def _upstream_commit(manifest: dict[str, Any]) -> str | None:
        upstream = manifest.get("upstream")
        return upstream.get("commit") if isinstance(upstream, dict) else None

    @staticmethod
    def _timestamp(path: Path) -> str:
        from datetime import datetime

        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()

    @staticmethod
    def _has_resume_artifacts(run_path: Path, phase: str) -> bool:
        return any((run_path / item).exists() for item in ("run-context.json", "episodes")) or phase.startswith("0-")

    def _active_episode(self, run_path: Path, status: str, heldout_sealed: bool) -> dict[str, Any] | None:
        if status != "running" or heldout_sealed:
            return None
        root = run_path / "episodes"
        if not root.exists():
            return None
        self._assert_regular_dir(root)
        active = []
        artifacts = {"episode-record.json", "run-telemetry.json", "native-simulation.json", "incomplete-run.json"}
        for attempt in root.iterdir():
            if attempt.is_symlink() or not attempt.is_dir():
                raise ArtifactReadError("active episode directory is unsafe")
            if not any((attempt / name).exists() for name in artifacts):
                active.append((attempt.stat().st_mtime, attempt.name))
        if not active:
            return None
        started, attempt_id = max(active)
        from datetime import datetime

        return {
            "attempt_id": attempt_id,
            "started_at": datetime.fromtimestamp(started, tz=UTC).isoformat(),
            "label": "episode 正在准备 / 执行",
        }

    def _load_episodes(
        self,
        run_path: Path,
        manifest: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        complete: bool,
    ) -> list[dict[str, Any]]:
        if manifest.get("phase") == "0-integration-proof":
            if result is None or not result.get("simulation_file"):
                return []
            trajectory_ref = str(result["simulation_file"])
            trajectory_path = self._contained_path(run_path, trajectory_ref)
            trajectory = self._read_json(trajectory_path)
            if result.get("simulation_id") not in (None, trajectory.get("id")):
                raise ArtifactReadError("Phase 0 result and trajectory do not match")
            strategy = result.get("strategy") or {}
            record = {
                "episode_id": str(trajectory.get("id") or "phase0"),
                "task_id": str(trajectory.get("task_id", result.get("task_id", "unknown"))),
                "seed": trajectory.get("seed", manifest.get("seed")),
                "status": result.get("status", "incomplete"),
                "task_success": _reward_success(result.get("native_reward")),
                "native_reward": result.get("native_reward"),
                "trajectory_ref": trajectory_ref,
                "customer_strategy": strategy.get("customer"),
                "service_strategy": strategy.get("service"),
                "panel_name": "phase0",
            }
            return [self._episode_view(run_path, record, trajectory, None)]

        heldout = set(self._heldout_ids(manifest))
        root = run_path / "episodes"
        if not root.exists():
            return []
        self._assert_regular_dir(root)
        views = []
        for directory in sorted(root.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                raise ArtifactReadError("episode store contains an unsafe entry")
            incomplete_path = directory / "incomplete-run.json"
            record_path = directory / "episode-record.json"
            if not record_path.exists():
                if incomplete_path.exists():
                    incomplete = self._read_json(incomplete_path)
                    task_id = str(incomplete.get("task_id", "unknown"))
                    panel = str(incomplete.get("panel_name", ""))
                    if not complete and (task_id in heldout or "heldout" in panel.lower()):
                        continue
                    views.append({
                        "episode_id": directory.name,
                        "task_id": task_id,
                        "seed": incomplete.get("seed"),
                        "status": "incomplete",
                        "panel_name": incomplete.get("panel_name"),
                        "trajectory": None,
                        "messages": [],
                        "tool_calls": 0,
                        "telemetry": None,
                    })
                continue
            record = EpisodeRecord.from_dict(self._read_json(record_path)).to_dict()
            trajectory_ref = record.get("trajectory_ref")
            telemetry_path = directory / "run-telemetry.json"
            if not isinstance(trajectory_ref, str) or not telemetry_path.is_file():
                raise ArtifactReadError("completed episode is missing its trajectory or telemetry")
            trajectory_path = self._contained_path(run_path, trajectory_ref)
            trajectory = self._read_json(trajectory_path)
            telemetry = self._read_json(telemetry_path)
            if (
                record["episode_id"] != trajectory.get("id")
                or record["task_id"] != str(trajectory.get("task_id"))
                or record["seed"] != trajectory.get("seed")
                or telemetry.get("simulation_id") != record["episode_id"]
            ):
                raise ArtifactReadError("episode record, telemetry, and trajectory disagree")
            panel = str(telemetry.get("panel_name", ""))
            if not complete and (record["task_id"] in heldout or "heldout" in panel.lower()):
                continue
            record["panel_name"] = telemetry.get("panel_name")
            record["generation"] = _generation_from_panel(panel)
            views.append(self._episode_view(run_path, record, trajectory, telemetry))
        return views

    def _episode_view(
        self, run_path: Path, record: dict[str, Any], trajectory: dict[str, Any],
        telemetry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        messages = normalize_trajectory_messages(trajectory)
        trajectory_ref = record.get("trajectory_ref")
        db_state_trace = self._db_state_trace_for_episode(
            run_path, record, trajectory, trajectory_ref,
        )
        started_at = None
        if isinstance(trajectory_ref, str):
            try:
                started_at = self._timestamp(self._contained_path(run_path, trajectory_ref).parent)
            except ArtifactReadError:
                pass
        return {
            "episode_id": str(record.get("episode_id", trajectory.get("id", "unknown"))),
            "task_id": str(record.get("task_id", trajectory.get("task_id", "unknown"))),
            "seed": record.get("seed", trajectory.get("seed")),
            "generation": record.get("generation"),
            "panel_name": record.get("panel_name"),
            "status": str(record.get("status", "complete")),
            "task_success": record.get("task_success"),
            "native_reward": record.get("native_reward"),
            "customer_strategy_id": record.get("customer_strategy_id"),
            "service_strategy_id": record.get("service_strategy_id"),
            "customer_strategy": record.get("customer_strategy"),
            "service_strategy": record.get("service_strategy"),
            "trajectory": redact_secrets(trajectory),
            "messages": messages,
            "db_state_trace": db_state_trace,
            "telemetry": redact_secrets(telemetry),
            "tool_calls": sum(message.get("kind") == "tool_call" for message in messages),
            "budget": None if telemetry is None else telemetry.get("budget_after"),
            "raw_record": redact_secrets(record),
            "trajectory_root": run_path,
            "run_path": run_path,
            "started_at": started_at,
        }

    def _db_state_trace_for_episode(
        self,
        run_path: Path,
        record: dict[str, Any],
        trajectory: dict[str, Any],
        trajectory_ref: Any,
    ) -> dict[str, Any]:
        unavailable = {"schema_version": 1, "status": "unavailable", "reason_code": "trace_not_generated", "events": []}
        if not isinstance(trajectory_ref, str):
            return unavailable
        try:
            trajectory_path = self._contained_path(run_path, trajectory_ref)
            trace_path = trajectory_path.parent / "db-state-trace.json"
            if trace_path.is_symlink() or not trace_path.exists():
                return unavailable
            trace = self._read_json(trace_path)
            manifest = self._read_json(run_path / "manifest.json")
            trace_hash = trace.get("trace_sha256")
            fingerprint_payload = dict(trace)
            fingerprint_payload.pop("trace_sha256", None)
            provenance = trace.get("provenance")
            if (
                trace.get("schema_version") != 1
                or trace.get("source") != "deterministic_replay"
                or trace_hash != sha256_json(fingerprint_payload)
                or not isinstance(provenance, dict)
                or provenance.get("manifest_sha256") != manifest.get("manifest_sha256")
                or provenance.get("trajectory_sha256") != hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
                or str(trace.get("task_id")) != str(trajectory.get("task_id"))
                or str(trace.get("episode_id")) != str(trajectory.get("id"))
                or trace.get("status") not in {"complete", "unavailable"}
            ):
                return unavailable
            if trace.get("status") == "complete" and (
                not isinstance(trace.get("events"), list) or not isinstance(trace.get("summary"), dict)
            ):
                return unavailable
            return redact_secrets(trace)
        except (ArtifactReadError, OSError, TypeError, ValueError, KeyError):
            return unavailable

    def _budget_summary(
        self, manifest: dict[str, Any], result: dict[str, Any] | None, run_path: Path,
    ) -> dict[str, Any]:
        snapshot = None if result is None else result.get("provider_usage", result.get("provider_budget"))
        if snapshot is None:
            telemetry_path = self._latest_telemetry_path(run_path)
            if telemetry_path is not None:
                telemetry = self._read_json(telemetry_path)
                snapshot = telemetry.get("budget_after")
        if not isinstance(snapshot, dict):
            return {
                "attempts": None, "cap": _manifest_int(manifest, "request_budget_cap"),
                "prompt_tokens": None, "completion_tokens": None, "model_usage": [],
                "available": False, "successes": None, "failures": None,
                "denied": None, "cache_hits": None,
            }
        unavailable = snapshot.get("usage_unavailable", 0) > 0
        return {
            "attempts": snapshot.get("attempts"),
            "successes": snapshot.get("successes"),
            "failures": snapshot.get("failures"),
            "denied": snapshot.get("denied"),
            "cache_hits": snapshot.get("cache_hits"),
            "cap": snapshot.get("cap", _manifest_int(manifest, "request_budget_cap")),
            "prompt_tokens": None if unavailable else snapshot.get("prompt_tokens"),
            "completion_tokens": None if unavailable else snapshot.get("completion_tokens"),
            "usage_unavailable": snapshot.get("usage_unavailable"),
            "model_usage": redact_secrets(snapshot.get("model_usage", [])),
            "available": True,
            "cost": "Cost unavailable until a frozen price schedule is supplied.",
        }

    @staticmethod
    def _latest_telemetry_path(run_path: Path) -> Path | None:
        paths = list(run_path.glob("episodes/*/run-telemetry.json"))
        paths = [path for path in paths if path.is_file() and not path.is_symlink()]
        return max(paths, key=lambda item: item.stat().st_mtime) if paths else None

    def _generation_commits(
        self, manifest: dict[str, Any], result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if result is not None and isinstance(result.get("generations"), list):
            return result["generations"]
        checkpoint = self._load_checkpoint(manifest)
        generations = None if checkpoint is None else checkpoint.get("generations")
        return generations if isinstance(generations, list) else []

    def _load_checkpoint(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        relative = manifest.get("checkpoint_path")
        if not isinstance(relative, str):
            return None
        path = self._contained_path(self.project_root, relative, require_exists=False)
        if not path.exists():
            return None
        return self._read_json(path)

    @staticmethod
    def _load_strategies(
        result: dict[str, Any] | None, generations: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        strategies: dict[str, dict[str, Any]] = {"customer": {}, "service": {}}

        def add(side: str, value: Any) -> None:
            if not isinstance(value, dict):
                return
            strategy_text = value.get("text", value.get("strategy"))
            if not isinstance(strategy_text, str):
                return
            strategy_id = value.get("strategy_id") or _strategy_id({"text": strategy_text})
            strategies[side][str(strategy_id)] = {"text": strategy_text}

        if isinstance(result, dict):
            for side in ("customer", "service"):
                add(side, result.get(f"initial_{side}"))
                add(side, result.get(f"final_{side}"))
            phase0_strategy = result.get("strategy")
            if isinstance(phase0_strategy, dict):
                add("customer", phase0_strategy.get("customer"))
                add("service", phase0_strategy.get("service"))
        for generation in generations:
            for field in ("before", "after"):
                for side in ("customer", "service"):
                    add(side, generation.get(f"{side}_{field}"))
            add("service", (generation.get("service_phase") or {}).get("proposed_strategy"))
        return strategies

    @staticmethod
    def _contained_path(root: Path, relative: str, *, require_exists: bool = True) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
            raise ArtifactReadError("artifact reference escapes its allowed directory")
        path = root / candidate
        _reject_symlink_components(root, path)
        resolved = path.resolve(strict=require_exists)
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactReadError("artifact reference escapes its allowed directory") from exc
        if require_exists and not resolved.is_file():
            raise ArtifactReadError("referenced artifact is unavailable")
        return resolved

    @staticmethod
    def _assert_regular_dir(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ArtifactReadError("artifact directory is unsafe")


def normalize_trajectory_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Render actual τ-bench messages and tool results without losing unknown forms."""

    rows = trajectory.get("messages")
    if rows is None and isinstance(trajectory.get("ticks"), list):
        rows = []
        for tick in trajectory["ticks"]:
            if isinstance(tick, dict):
                rows.extend(tick.get("messages") or ())
    if not isinstance(rows, list):
        return []
    result = []
    for index, message in enumerate(rows):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", message.get("sender", "unknown"))).lower()
        content = message.get("content", message.get("text", ""))
        turn = message.get("turn_idx", message.get("turn_index", index))
        tool_calls = message.get("tool_calls") or ()
        if role in {"user", "customer"}:
            result.append({"kind": "message", "role": "user", "content": content, "turn_index": turn})
        elif role in {"assistant", "agent", "service"}:
            result.append({"kind": "message", "role": "assistant", "content": content, "turn_index": turn, "tool_calls": tool_calls})
            for call_index, call in enumerate(tool_calls):
                if isinstance(call, dict):
                    result.append({
                        "kind": "tool_call", "role": "assistant", "turn_index": turn,
                        "tool_id": call.get("id", call_index),
                        "tool_name": call.get("name") or (call.get("function") or {}).get("name", "tool"),
                        "arguments": call.get("arguments") or (call.get("function") or {}).get("arguments", {}),
                    })
        elif role in {"tool", "environment", "env"}:
            result.append({
                "kind": "tool_result", "role": "environment", "turn_index": turn,
                "tool_id": message.get("tool_call_id") or message.get("id", index),
                "tool_name": message.get("name", "tool"), "content": content,
            })
        else:
            result.append({
                "kind": "message", "role": role, "turn_index": turn,
                "content": content or message,
            })
    return redact_secrets(result)


def redact_secrets(value: Any, *, _key: str = "") -> Any:
    if _SECRET_KEY.search(_key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(key): redact_secrets(item, _key=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return _SECRET_TEXT.sub(lambda match: (match.group(1) or "") + "[REDACTED]", value)
    return value


def _reward_success(reward: Any) -> bool | None:
    return reward >= 1.0 if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None


def _generation_from_panel(panel_name: Any) -> int | None:
    match = re.search(r"generation[-_: ]*(\d+)", str(panel_name or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _strategy_id(strategy: dict[str, Any]) -> str:
    return sha256_json(strategy)[:16]


def _upstream_commit(manifest: dict[str, Any]) -> str | None:
    upstream = manifest.get("upstream")
    return upstream.get("commit") if isinstance(upstream, dict) else None


def _manifest_int(manifest: dict[str, Any], field: str) -> int | None:
    value = manifest.get(field)
    if value is None and isinstance(manifest.get("experiment"), dict):
        value = manifest["experiment"].get(field)
    return value if type(value) is int else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_symlink_components(root: Path, path: Path) -> None:
    base = root.resolve()
    try:
        relative = path.absolute().relative_to(base)
    except ValueError as exc:
        raise ArtifactReadError("artifact path escapes its allowed directory") from exc
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactReadError("artifact paths cannot pass through symlinks")


def _safe_manifest_for_display(manifest: dict[str, Any], *, sealed: bool) -> dict[str, Any]:
    value = dict(manifest)
    selection = value.get("task_selection")
    if sealed and isinstance(selection, dict):
        selection = dict(selection)
        heldout = selection.get("heldout", ())
        selection["heldout"] = f"[SEALED: {len(heldout)} tasks]" if isinstance(heldout, list) else "[SEALED]"
        value["task_selection"] = selection
    panels = value.get("task_panels")
    if sealed and isinstance(panels, dict):
        panels = dict(panels)
        heldout = panels.get("H", ())
        panels["H"] = f"[SEALED: {len(heldout)} tasks]" if isinstance(heldout, list) else "[SEALED]"
        value["task_panels"] = panels
    return value
