"""Fail-closed, read-only adapters for EvoTau's existing run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..crossplay import CrossPlayMatrix
from ..manifest import sha256_json
from ..records import EpisodeRecord, FailureRecord

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
    """Read immutable EvoTau files without creating or mutating Core artifacts."""

    def __init__(self, runs_root: str | Path, *, project_root: str | Path | None = None):
        self.runs_root = Path(runs_root).expanduser().absolute()
        self.project_root = Path(project_root or self.runs_root.parent.parent).expanduser().absolute()

    def list_runs(self) -> tuple[dict[str, Any], ...]:
        if not self.runs_root.exists():
            return ()
        if self.runs_root.is_symlink() or not self.runs_root.is_dir():
            raise ArtifactReadError("run directory is not a regular directory")
        paths = set(self.runs_root.glob("*/manifest.json"))
        paths.update(self.runs_root.glob("*/pilot-manifest.json"))
        result = []
        for manifest_path in sorted(paths):
            run_path = manifest_path.parent
            if run_path.is_symlink() or not run_path.is_dir():
                raise ArtifactReadError("run directory contains a symlink or non-directory")
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
        manifest_path = self._manifest_path(run_path)
        manifest = self._read_json(manifest_path, expected=dict)
        self._verify_manifest(manifest)
        result_path = self._result_path(run_path)
        result = self._read_json(result_path, expected=dict) if result_path else None
        if result is not None:
            self._verify_result_manifest(result, manifest)
        phase = str(manifest.get("phase", "unknown"))
        if phase.startswith("4-") and result is not None and result.get("status") != "complete":
            raise ArtifactReadError("Pilot root result is not a completed release index; run is hidden")
        if result is not None and result.get("experiment_id") not in (None, manifest.get("experiment_id")):
            raise ArtifactReadError("result and manifest experiment IDs differ; run is hidden")
        if phase.startswith("4-") and result is not None and result.get("status") == "complete":
            self._validate_pilot_root_index(run_path, manifest, result)
        experiment_id = str(manifest.get("experiment_id", manifest.get("id", run_path.name)))
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

        episodes = self._load_episodes(run_path, manifest, result, complete=complete)
        budget = self._budget_summary(manifest, result, run_path)
        commits = self._generation_commits(manifest, result, run_path)
        failures = self._load_verified_failures(run_path)
        if heldout_ids and not complete:
            heldout = set(heldout_ids)
            failures = [failure for failure in failures if str(failure.get("task_id")) not in heldout]
        provisional = [
            episode for episode in episodes
            if episode.get("policy_violation") is True
            and not episode.get("verified_failure")
        ]
        strategies = self._load_strategies(run_path, manifest, result, commits)
        crossplays = self._load_crossplays(run_path)
        task_review = self._task_review_status(manifest)
        current_generation = (
            max((item.get("generation", -1) for item in commits), default=-1) + 1
            if status in {"running", "paused", "incomplete"} else None
        )
        h_sealed = bool(heldout_ids) and not complete
        current_episode = self._active_episode(run_path, phase, status, heldout_sealed=h_sealed)
        latest_commit = max(commits, key=lambda item: item.get("generation", -1), default=None)
        current_customer_id = latest_commit.get("customer_id") if latest_commit else None
        current_service_id = latest_commit.get("service_id") if latest_commit else None
        current_customer = None
        current_service = None
        if latest_commit:
            current_customer = strategies["customer"].get(str(current_customer_id))
            current_service = strategies["service"].get(str(current_service_id))
        elif isinstance(result, dict) and isinstance(result.get("strategy"), dict):
            current_customer = result["strategy"].get("customer")
            current_service = result["strategy"].get("service")
            current_customer_id = _strategy_id(current_customer) if isinstance(current_customer, dict) else None
            current_service_id = _strategy_id(current_service) if isinstance(current_service, dict) else None
        return {
            "run_id": run_id,
            "path": run_path,
            "experiment_id": experiment_id,
            "phase": phase,
            "status": status,
            "episode_limit": _manifest_int(manifest, "max_episodes"),
            "max_concurrency": _manifest_int(manifest, "max_concurrency"),
            "manifest": _safe_manifest_for_display(manifest, sealed=bool(heldout_ids) and not complete),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "real_provider_enabled": manifest.get("real_provider_enabled") is True,
            "upstream_commit": self._upstream_commit(manifest),
            "started_at": self._timestamp(manifest_path),
            # Before release, the aggregate result may still contain heldout rows
            # even when the normalized episode list has been filtered.
            "result": None if h_sealed else redact_secrets(result),
            "budget": budget,
            "episodes": episodes,
            "generation_commits": redact_secrets(commits),
            "current_generation": current_generation,
            "current_episode": current_episode,
            "current_customer_id": current_customer_id,
            "current_service_id": current_service_id,
            "current_customer_strategy": redact_secrets(current_customer),
            "current_service_strategy": redact_secrets(current_service),
            "verified_failures": redact_secrets(failures),
            "provisional_failures": provisional,
            "strategies": redact_secrets(strategies),
            "crossplays": crossplays,
            "heldout_sealed": h_sealed,
            "heldout_task_count": len(heldout_ids),
            "task_review": task_review,
            "raw_artifacts": self._safe_raw_artifacts(
                manifest, result, sealed=bool(heldout_ids) and not complete,
            ),
        }

    def episode(self, run_id: str, episode_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        for episode in run["episodes"]:
            if episode["episode_id"] == episode_id:
                return {**episode, "run": run}
        raise FileNotFoundError("episode artifact is unavailable")

    def crossplay(self, run_id: str, name: str | None = None) -> dict[str, Any]:
        run = self.get_run(run_id)
        matrices = run["crossplays"]
        if name is None:
            if not matrices:
                raise FileNotFoundError("no validated cross-play matrix is available")
            return {"run": run, "matrices": matrices, "selected": matrices[0]}
        selected = next((item for item in matrices if item["name"] == name), None)
        if selected is None:
            raise FileNotFoundError("cross-play matrix is unavailable")
        return {"run": run, "matrices": matrices, "selected": selected}

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
    def _manifest_path(run_path: Path) -> Path:
        for name in ("manifest.json", "pilot-manifest.json"):
            path = run_path / name
            if path.exists():
                if path.is_symlink():
                    raise ArtifactReadError("manifest cannot be a symlink")
                return path
        raise FileNotFoundError("run manifest is not available yet")

    @staticmethod
    def _result_path(run_path: Path) -> Path | None:
        for name in ("phase0-result.json", "phase3-result.json", "pilot-result.json"):
            path = run_path / name
            if path.exists():
                if path.is_symlink():
                    raise ArtifactReadError("result artifact cannot be a symlink")
                return path
        return None

    @staticmethod
    def _read_json(path: Path, *, expected: type = dict) -> Any:
        if path.is_symlink() or not path.is_file():
            raise ArtifactReadError("artifact is not a regular file")
        try:
            size = path.stat().st_size
            if size > _MAX_ARTIFACT_BYTES:
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
        if not manifest.get("experiment_id", manifest.get("id")):
            raise ArtifactReadError("manifest has no experiment ID")

    @staticmethod
    def _verify_result_manifest(result: dict[str, Any], manifest: dict[str, Any]) -> None:
        if result.get("schema_version") != 1:
            raise ArtifactReadError("unsupported result schema; run is hidden")
        if result.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise ArtifactReadError("result and manifest do not match; run is hidden")
        if result.get("status") not in {"complete", "incomplete"}:
            raise ArtifactReadError("result status is malformed; run is hidden")

    @staticmethod
    def _heldout_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
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

    def _has_resume_artifacts(self, run_path: Path, phase: str) -> bool:
        return any((run_path / item).exists() for item in (
            "run-context.json", "episodes", "seed-blocks", "archive.sqlite",
        )) or phase.startswith("0-")

    def _active_episode(
        self, run_path: Path, phase: str, status: str, *, heldout_sealed: bool,
    ) -> dict[str, Any] | None:
        if status != "running" or phase.startswith("0-") or heldout_sealed:
            return None
        roots = [run_path / "episodes"]
        if phase.startswith("4-"):
            roots = list((run_path / "seed-blocks").glob("*/episodes"))
        active = []
        artifact_names = {
            "episode-record.json", "run-telemetry.json", "native-simulation.json",
            "incomplete-run.json", "db-state-trace.json",
        }
        for root in roots:
            if not root.exists():
                continue
            self._assert_regular_dir(root)
            for attempt in root.iterdir():
                if attempt.is_symlink() or not attempt.is_dir():
                    raise ArtifactReadError("active episode directory is unsafe")
                if not any((attempt / name).exists() for name in artifact_names):
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
        phase = str(manifest.get("phase", ""))
        if phase.startswith("0-"):
            return self._phase0_episode(run_path, manifest, result)
        if phase.startswith("4-"):
            return self._pilot_episodes(run_path, manifest, result, complete=complete)
        return self._phase3_episodes(run_path, manifest, result, complete=complete)

    def _phase0_episode(
        self, run_path: Path, manifest: dict[str, Any], result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if result is None:
            return []
        simulation_name = result.get("simulation_file")
        if not simulation_name:
            return []
        trajectory = self._contained_json(run_path, str(simulation_name))
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
            "policy_violation": False,
            "customer_valid": None,
            "strategy_applicable": None,
            "customer_strategy_adherent": None,
            "generation": None,
            "panel_name": "phase0",
            "customer_strategy": strategy.get("customer"),
            "service_strategy": strategy.get("service"),
            "trajectory_ref": str(simulation_name),
            "verified_failure": False,
        }
        return [self._episode_view(run_path, run_path, record, trajectory, None, None)]

    def _phase3_episodes(
        self,
        run_path: Path,
        manifest: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        complete: bool,
    ) -> list[dict[str, Any]]:
        rows = result.get("episodes", ()) if result else None
        entries = []
        if rows is not None:
            if not isinstance(rows, list):
                raise ArtifactReadError("Phase 3 episode index is malformed")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("artifacts"), dict):
                    raise ArtifactReadError("Phase 3 episode row is malformed")
                directory = run_path / "episodes" / str(row.get("attempt_id", ""))
                payloads = self._indexed_artifacts(run_path, row["artifacts"])
                entries.append((directory, payloads, row))
        else:
            root = run_path / "episodes"
            if root.exists():
                self._assert_regular_dir(root)
                for directory in sorted(root.iterdir()):
                    if directory.is_symlink() or not directory.is_dir():
                        raise ArtifactReadError("episode store contains an unsafe entry")
                    payloads = self._optional_episode_artifacts(directory)
                    entries.append((directory, payloads, {}))
        views = []
        for directory, payloads, index in entries:
            record = payloads.get("episode-record.json")
            telemetry = payloads.get("run-telemetry.json")
            trajectory = payloads.get("native-simulation.json")
            incomplete = payloads.get("incomplete-run.json")
            if record is None:
                if incomplete is None:
                    continue
                item = {
                    "episode_id": str(index.get("episode_id") or directory.name),
                    "task_id": str(incomplete.get("task_id", index.get("task_id", "unknown"))),
                    "seed": incomplete.get("seed"),
                    "status": "incomplete",
                    "policy_violation": False,
                    "generation": None,
                    "panel_name": incomplete.get("panel_name", index.get("panel_name")),
                    "verified_failure": False,
                }
                if trajectory is not None:
                    views.append(self._episode_view(run_path, run_path, item, trajectory, telemetry, None))
                else:
                    views.append(item)
                continue
            parsed = EpisodeRecord.from_dict(record)
            if trajectory is None or telemetry is None:
                raise ArtifactReadError("completed episode is missing trajectory or telemetry")
            if (parsed.episode_id != trajectory.get("id")
                    or parsed.task_id != str(trajectory.get("task_id"))
                    or parsed.seed != trajectory.get("seed")
                    or telemetry.get("simulation_id") != parsed.episode_id):
                raise ArtifactReadError("episode record, telemetry, and trajectory disagree")
            item = parsed.to_dict()
            item.update({
                "panel_name": telemetry.get("panel_name"),
                "generation": _generation_from_panel(telemetry.get("panel_name")),
                "verified_failure": parsed.episode_id in {
                    str(row.get("episode_id")) for row in self._load_verified_failures(run_path)
                },
            })
            views.append(self._episode_view(run_path, run_path, item, trajectory, telemetry, None))
        heldout = set(self._heldout_ids(manifest))
        if heldout and not complete:
            views = [item for item in views if str(item.get("task_id")) not in heldout
                     and "heldout" not in str(item.get("panel_name", "")).lower()]
        return views

    def _pilot_episodes(
        self,
        run_path: Path,
        manifest: dict[str, Any],
        result: dict[str, Any] | None,
        *,
        complete: bool,
    ) -> list[dict[str, Any]]:
        block_root = run_path / "seed-blocks"
        if not block_root.exists():
            return []
        self._assert_regular_dir(block_root)
        views = []
        heldout_ids = set(self._heldout_ids(manifest))
        for seed_dir in sorted(block_root.iterdir()):
            if seed_dir.is_symlink() or not seed_dir.is_dir():
                raise ArtifactReadError("Pilot seed-block store contains an unsafe entry")
            seed_result_path = seed_dir / "pilot-seed-result.json"
            seed_result = None
            if seed_result_path.exists():
                seed_result = self._read_json(seed_result_path)
                if result is not None:
                    seed_row = next((item for item in result.get("seed_blocks", ())
                                     if str(item.get("evolution_seed")) == str(seed_result.get("evolution_seed"))), None)
                    if seed_row is None:
                        raise ArtifactReadError("Pilot seed result is absent from the immutable root index")
                    if hashlib.sha256(seed_result_path.read_bytes()).hexdigest() != seed_row.get("result_sha256"):
                        raise ArtifactReadError("Pilot seed result hash does not match root index")
                    indexed_path = self._contained_path(self.project_root, str(seed_row.get("result_path", "")))
                    if indexed_path != seed_result_path.absolute():
                        raise ArtifactReadError("Pilot root index points at a different seed result")
                    if (seed_result.get("manifest_sha256") != result.get("manifest_sha256")
                            or str(seed_result.get("evolution_seed")) != str(seed_row.get("evolution_seed"))):
                        raise ArtifactReadError("Pilot seed result does not match its immutable root index")
            records = seed_result.get("episodes", ()) if seed_result else None
            if records is None:
                episode_root = seed_dir / "episodes"
                if not episode_root.exists():
                    continue
                self._assert_regular_dir(episode_root)
                for episode_dir in sorted(episode_root.iterdir()):
                    if episode_dir.is_symlink() or not episode_dir.is_dir():
                        raise ArtifactReadError("Pilot episode store contains an unsafe entry")
                    incomplete_path = episode_dir / "incomplete-run.json"
                    if incomplete_path.exists():
                        incomplete = self._read_json(incomplete_path)
                        if not complete and (
                            str(incomplete.get("task_id")) in heldout_ids
                            or "heldout" in str(incomplete.get("panel_name", "")).lower()
                        ):
                            continue
                    payloads = self._optional_episode_artifacts(episode_dir)
                    record = payloads.get("episode-record.json")
                    incomplete = payloads.get("incomplete-run.json")
                    telemetry = payloads.get("run-telemetry.json")
                    trajectory = payloads.get("native-simulation.json")
                    if record:
                        record = EpisodeRecord.from_dict(record).to_dict()
                        if telemetry:
                            record["panel_name"] = telemetry.get("panel_name")
                        if trajectory:
                            views.append(self._episode_view(run_path, seed_dir, record, trajectory, telemetry, None))
                    elif incomplete:
                        views.append({
                            "episode_id": episode_dir.name,
                            "task_id": incomplete.get("task_id"),
                            "seed": incomplete.get("seed"),
                            "status": "incomplete",
                            "panel_name": incomplete.get("panel_name"),
                            "policy_violation": False,
                            "verified_failure": False,
                        })
                continue
            if not isinstance(records, list):
                raise ArtifactReadError("Pilot seed episode list is malformed")
            for row in records:
                parsed = EpisodeRecord.from_dict(row)
                if not complete and (
                    parsed.task_id in heldout_ids
                    or "heldout" in str(row.get("panel_name", "")).lower()
                ):
                    continue
                if parsed.trajectory_ref is None:
                    continue
                trajectory_path = self._contained_path(seed_dir, parsed.trajectory_ref)
                trajectory = self._read_json(trajectory_path)
                episode_dir = trajectory_path.parent
                self._verify_pilot_artifact(seed_result, trajectory_path, self.project_root)
                telemetry_path = episode_dir / "run-telemetry.json"
                telemetry = self._read_json(telemetry_path) if telemetry_path.exists() else None
                if telemetry is not None:
                    self._verify_pilot_artifact(seed_result, telemetry_path, self.project_root)
                item = parsed.to_dict()
                item["panel_name"] = telemetry.get("panel_name") if telemetry else None
                item["generation"] = _generation_from_panel(item["panel_name"])
                item["verified_failure"] = parsed.episode_id in {
                    failure["episode_id"] for failure in self._load_verified_failures(run_path)
                }
                views.append(self._episode_view(run_path, seed_dir, item, trajectory, telemetry, None))
        heldout = set(self._heldout_ids(manifest))
        if heldout and not complete:
            views = [item for item in views if str(item.get("task_id")) not in heldout
                     and "heldout" not in str(item.get("panel_name", "")).lower()]
        return views

    def _episode_view(
        self,
        run_path: Path,
        trajectory_root: Path,
        record: dict[str, Any],
        trajectory: dict[str, Any],
        telemetry: dict[str, Any] | None,
        _unused: Any,
    ) -> dict[str, Any]:
        messages = normalize_trajectory_messages(trajectory)
        db_state_trace = self._db_state_trace_for_episode(
            run_path, trajectory_root, record, trajectory,
        )
        started_at = None
        trajectory_ref = record.get("trajectory_ref")
        if isinstance(trajectory_ref, str):
            try:
                trajectory_path = self._contained_path(trajectory_root, trajectory_ref)
                started_at = self._timestamp(trajectory_path.parent)
            except ArtifactReadError:
                started_at = None
        return {
            "episode_id": str(record.get("episode_id", trajectory.get("id", "unknown"))),
            "task_id": str(record.get("task_id", trajectory.get("task_id", "unknown"))),
            "seed": record.get("seed", trajectory.get("seed")),
            "generation": record.get("generation"),
            "panel_name": record.get("panel_name"),
            "status": str(record.get("status", "complete")),
            "task_success": record.get("task_success"),
            "native_reward": record.get("native_reward"),
            "verified_failure": bool(record.get("verified_failure")),
            "policy_violation": record.get("policy_violation") is True,
            "failure_category": _failure_category(record),
            "customer_valid": record.get("customer_valid"),
            "strategy_applicable": record.get("strategy_applicable"),
            "customer_strategy_adherent": record.get("customer_strategy_adherent"),
            "customer_strategy_id": record.get("customer_strategy_id"),
            "service_strategy_id": record.get("service_strategy_id"),
            "customer_strategy": record.get("customer_strategy"),
            "service_strategy": record.get("service_strategy"),
            "evidence": record.get("evidence", ()),
            "trajectory": redact_secrets(trajectory),
            "messages": normalize_trajectory_messages(trajectory),
            "db_state_trace": db_state_trace,
            "telemetry": redact_secrets(telemetry),
            "tool_calls": sum(message.get("kind") == "tool_call" for message in messages),
            "budget": (telemetry or {}).get("budget_after"),
            "raw_record": redact_secrets(record),
            "trajectory_root": trajectory_root,
            "run_path": run_path,
            "started_at": started_at,
        }

    def _db_state_trace_for_episode(
        self,
        run_path: Path,
        trajectory_root: Path,
        record: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> dict[str, Any]:
        unavailable = {
            "schema_version": 1,
            "status": "unavailable",
            "reason_code": "trace_not_generated",
            "events": [],
        }
        reference = record.get("trajectory_ref")
        if not isinstance(reference, str):
            return unavailable
        try:
            trajectory_path = self._contained_path(trajectory_root, reference)
            trace_path = trajectory_path.parent / "db-state-trace.json"
            if trace_path.is_symlink():
                return unavailable
            if not trace_path.exists():
                return unavailable
            trace = self._read_json(trace_path, expected=dict)
            manifest = self._read_json(self._manifest_path(run_path), expected=dict)
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
                not isinstance(trace.get("events"), list)
                or not isinstance(trace.get("summary"), dict)
            ):
                return unavailable
            return redact_secrets(trace)
        except (ArtifactReadError, OSError, TypeError, ValueError, KeyError):
            # The optional observer sidecar never hides a valid episode.
            return unavailable

    def _budget_summary(
        self, manifest: dict[str, Any], result: dict[str, Any] | None, run_path: Path,
    ) -> dict[str, Any]:
        snapshot = result.get("provider_budget") if result else None
        if snapshot is None and result and result.get("seed_blocks"):
            blocks = result["seed_blocks"]
            snapshots = [row.get("provider_budget", {}) for row in blocks]
            snapshot = {
                "attempts": sum(int(row.get("attempts", 0)) for row in snapshots),
                "cap": sum(int(row.get("cap", 0)) for row in snapshots),
                "prompt_tokens": _sum_if_available(snapshots, "prompt_tokens"),
                "completion_tokens": _sum_if_available(snapshots, "completion_tokens"),
                "model_usage": [item for row in snapshots for item in row.get("model_usage", ())],
                "seed_blocks": snapshots,
            }
        if snapshot is None and str(manifest.get("phase", "")).startswith("0-"):
            incomplete = result if result and result.get("status") == "incomplete" else None
            snapshot = None if incomplete is None else incomplete.get("provider_budget")
        if snapshot is None:
            latest = self._latest_telemetry(run_path)
            snapshot = None if latest is None else latest.get("budget_after")
        if not isinstance(snapshot, dict):
            return {"attempts": None, "cap": self._budget_cap(manifest), "prompt_tokens": None,
                    "completion_tokens": None, "model_usage": [], "available": False,
                    "successes": None, "failures": None, "denied": None, "cache_hits": None}
        return {
            "attempts": snapshot.get("attempts"),
            "successes": snapshot.get("successes"),
            "failures": snapshot.get("failures"),
            "denied": snapshot.get("denied"),
            "cache_hits": snapshot.get("cache_hits"),
            "cap": snapshot.get("cap", self._budget_cap(manifest)),
            "prompt_tokens": snapshot.get("prompt_tokens") if snapshot.get("usage_unavailable", 0) == 0 else None,
            "completion_tokens": snapshot.get("completion_tokens") if snapshot.get("usage_unavailable", 0) == 0 else None,
            "usage_unavailable": snapshot.get("usage_unavailable"),
            "model_usage": redact_secrets(snapshot.get("model_usage", [])),
            "available": True,
            "seed_blocks": snapshot.get("seed_blocks"),
            "cost": "Cost unavailable until a frozen price schedule is supplied.",
        }

    @staticmethod
    def _budget_cap(manifest: dict[str, Any]) -> int | None:
        value = manifest.get("request_budget_cap")
        if value is None:
            experiment = manifest.get("experiment", {})
            value = experiment.get("request_budget_cap") if isinstance(experiment, dict) else None
        return value if type(value) is int else None

    def _latest_telemetry(self, run_path: Path) -> dict[str, Any] | None:
        paths = list(run_path.glob("episodes/*/run-telemetry.json"))
        paths.extend(run_path.glob("seed-blocks/*/episodes/*/run-telemetry.json"))
        if not paths:
            return None
        return self._read_json(max(paths, key=lambda item: item.stat().st_mtime))

    def _generation_commits(
        self, manifest: dict[str, Any], result: dict[str, Any] | None, run_path: Path,
    ) -> list[dict[str, Any]]:
        if result and isinstance(result.get("generation_commits"), list):
            return result["generation_commits"]
        checkpoint = self._load_checkpoint(manifest)
        if checkpoint is None:
            return []
        state = checkpoint.get("state")
        if isinstance(state, dict) and isinstance(state.get("commits"), list):
            return state["commits"]
        return []

    def _load_checkpoint(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        path_value = manifest.get("paths", {}).get("checkpoint") if isinstance(manifest.get("paths"), dict) else None
        if not path_value:
            path_value = manifest.get("checkpoint_path")
        if not isinstance(path_value, str):
            return None
        path = self._contained_path(self.project_root, path_value, require_exists=False)
        if not path.exists():
            return None
        return self._read_json(path)

    def _load_verified_failures(self, run_path: Path) -> list[dict[str, Any]]:
        failures = []
        archives = [run_path / "archive.sqlite", *run_path.glob("seed-blocks/*/archive.sqlite")]
        seen: set[str] = set()
        for archive in archives:
            if not archive.exists():
                continue
            if archive.is_symlink() or not archive.is_file():
                raise ArtifactReadError("failure archive is not a regular file")
            uri = f"file:{quote(str(archive), safe='/')}?mode=ro"
            try:
                with sqlite3.connect(uri, uri=True) as db:
                    rows = db.execute(
                        "SELECT payload, protocol_version FROM failures ORDER BY generation, failure_id"
                    ).fetchall()
            except sqlite3.Error as exc:
                raise ArtifactReadError("failure archive is malformed and has been hidden") from exc
            for payload, version in rows:
                if version != 2:
                    continue
                try:
                    parsed = FailureRecord.from_dict(json.loads(payload))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ArtifactReadError("verified-failure archive entry is malformed") from exc
                if parsed.failure_id not in seen:
                    seen.add(parsed.failure_id)
                    failures.append(parsed.to_dict())
        return failures

    def _load_strategies(
        self,
        run_path: Path,
        manifest: dict[str, Any],
        result: dict[str, Any] | None,
        commits: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        customers: dict[str, dict[str, Any]] = {}
        services: dict[str, dict[str, Any]] = {}
        if result and isinstance(result.get("strategy"), dict):
            strategy = result["strategy"]
            if strategy.get("customer"):
                customers[_strategy_id(strategy["customer"])] = strategy["customer"]
            if strategy.get("service"):
                services[_strategy_id(strategy["service"])] = strategy["service"]
        for commit in commits:
            record = commit.get("decision_record") or {}
            for side, target in (("customer", customers), ("service", services)):
                decision = record.get(side, {})
                for field in ("incumbent_before", "incumbent_after"):
                    value = decision.get(field)
                    if isinstance(value, dict):
                        target[_strategy_id(value)] = value
                for proposal in decision.get("proposals", ()):
                    value = proposal.get("strategy")
                    if isinstance(value, dict):
                        target[_strategy_id(value)] = value
        archives = [run_path / "archive.sqlite", *run_path.glob("seed-blocks/*/archive.sqlite")]
        for archive in archives:
            if not archive.exists():
                continue
            if archive.is_symlink() or not archive.is_file():
                raise ArtifactReadError("strategy archive is not a regular file")
            uri = f"file:{quote(str(archive), safe='/')}?mode=ro"
            try:
                with sqlite3.connect(uri, uri=True) as db:
                    for table, target in (("customer_strategies", customers), ("service_strategies", services)):
                        for strategy_id, payload in db.execute(f"SELECT strategy_id, payload FROM {table}"):
                            value = json.loads(payload)
                            if isinstance(value, dict) and _strategy_id(value) == strategy_id:
                                target[strategy_id] = value
            except sqlite3.Error as exc:
                raise ArtifactReadError("strategy archive is malformed and has been hidden") from exc
        return {"customer": customers, "service": services}

    def _load_crossplays(self, run_path: Path) -> list[dict[str, Any]]:
        if run_path.is_symlink():
            raise ArtifactReadError("run directory is unsafe")
        candidates = []
        for name in ("crossplay-matrix.json", "crossplay.json", "crossplay-matrices.json"):
            candidates.extend(run_path.rglob(name))
        matrices = []
        for path in candidates:
            if any(part.startswith(".") for part in path.relative_to(run_path).parts):
                continue
            safe_path = self._contained_path(run_path, path.relative_to(run_path).as_posix())
            document = self._read_json(safe_path)
            rows = document if isinstance(document, list) else [document]
            for index, item in enumerate(rows):
                try:
                    matrix = CrossPlayMatrix.from_dict(item)
                except (TypeError, ValueError) as exc:
                    raise ArtifactReadError("cross-play matrix is invalid; it has been hidden") from exc
                manifest = self._read_json(self._manifest_path(run_path))
                if (self._heldout_ids(manifest)
                        and not self._has_complete_result(run_path)
                        and set(matrix.task_ids) & set(self._heldout_ids(manifest))):
                    continue
                matrices.append({
                    "name": path.relative_to(run_path).as_posix() + (f"#{index}" if len(rows) > 1 else ""),
                    "matrix": redact_secrets(matrix.to_dict()),
                })
        return matrices

    def _task_review_status(self, manifest: dict[str, Any]) -> dict[str, str]:
        review = manifest.get("task_semantic_review")
        if not isinstance(review, dict):
            return {"label": "未提供", "detail": "该阶段不需要 Pilot 人工语义审查。"}
        reviews = review.get("task_reviews", ())
        pairs = review.get("pairwise_reviews", ())
        if not reviews or any(item.get("rationale", "").startswith("REVIEW REQUIRED")
                              for item in (*reviews, *pairs)):
            return {"label": "未完成", "detail": "人工任务语义审查尚未完成。"}
        return {"label": "已记录", "detail": "已记录在冻结 manifest 中；请检查审查者和任务哈希。"}

    def _safe_raw_artifacts(
        self, manifest: dict[str, Any], result: dict[str, Any] | None, *, sealed: bool,
    ) -> dict[str, Any]:
        safe_result = None if sealed else result
        return redact_secrets({
            "manifest": _safe_manifest_for_display(manifest, sealed=sealed),
            "result": safe_result,
        })

    def _has_complete_result(self, run_path: Path) -> bool:
        path = self._result_path(run_path)
        if path is None:
            return False
        result = self._read_json(path)
        return result.get("status") == "complete"

    def _validate_pilot_root_index(
        self, run_path: Path, manifest: dict[str, Any], result: dict[str, Any],
    ) -> None:
        rows = result.get("seed_blocks")
        seeds = manifest.get("evolution_seeds")
        if (not isinstance(rows, list) or not isinstance(seeds, list)
                or len(rows) != len(seeds)):
            raise ArtifactReadError("completed Pilot is missing its full immutable seed index")
        indexed_seeds = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("result_path"), str):
                raise ArtifactReadError("Pilot root seed index is malformed")
            seed = row.get("evolution_seed")
            indexed_seeds.append(seed)
            seed_result_path = self._contained_path(self.project_root, row["result_path"])
            expected_path = run_path / "seed-blocks" / f"seed-{seed}" / "pilot-seed-result.json"
            if seed_result_path != expected_path:
                raise ArtifactReadError("Pilot root index points at a non-canonical seed result path")
            try:
                seed_result_path.relative_to(run_path)
            except ValueError as exc:
                raise ArtifactReadError("Pilot seed result escapes its run directory") from exc
            if seed_result_path.name != "pilot-seed-result.json":
                raise ArtifactReadError("Pilot root index references an unexpected seed artifact")
            raw = seed_result_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row.get("result_sha256"):
                raise ArtifactReadError("Pilot seed result hash does not match root index")
            seed_result = self._read_json(seed_result_path)
            if (seed_result.get("schema_version") != 1
                    or seed_result.get("status") != "complete"
                    or seed_result.get("manifest_sha256") != manifest.get("manifest_sha256")
                    or seed_result.get("evolution_seed") != seed):
                raise ArtifactReadError("Pilot seed result is not bound to its completed root index")
        if indexed_seeds != seeds:
            raise ArtifactReadError("Pilot seed index differs from the frozen evolution seeds")

    def _indexed_artifacts(self, run_path: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for logical, row in artifacts.items():
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ArtifactReadError("artifact index row is malformed")
            path = self._contained_path(run_path, row["path"])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
                raise ArtifactReadError("indexed artifact hash mismatch; run is hidden")
            name = {
                "episode_record": "episode-record.json",
                "run_telemetry": "run-telemetry.json",
                "native_simulation": "native-simulation.json",
                "incomplete_run": "incomplete-run.json",
            }.get(logical, logical + ".json")
            result[name] = self._read_json(path)
        return result

    def _optional_episode_artifacts(self, directory: Path) -> dict[str, Any]:
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactReadError("episode artifact path is unsafe")
        result = {}
        for name in ("episode-record.json", "run-telemetry.json", "native-simulation.json", "incomplete-run.json"):
            path = directory / name
            if path.exists():
                result[name] = self._read_json(path)
        return result

    def _contained_json(self, root: Path, relative: str) -> dict[str, Any]:
        return self._read_json(self._contained_path(root, relative))

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

    @staticmethod
    def _verify_pilot_artifact(
        seed_result: dict[str, Any], path: Path, project_root: Path,
    ) -> None:
        try:
            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise ArtifactReadError("Pilot episode artifact escapes the project directory") from exc
        listed = next((item for item in seed_result.get("artifacts", ())
                       if item.get("path") == relative), None)
        if listed is None:
            raise ArtifactReadError("Pilot episode artifact is absent from the immutable seed index")
        if hashlib.sha256(path.read_bytes()).hexdigest() != listed.get("sha256"):
            raise ArtifactReadError("Pilot episode artifact hash mismatch")


def normalize_trajectory_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Render actual τ-bench Message payloads; unknown forms remain explicit."""

    rows = trajectory.get("messages")
    if rows is None and isinstance(trajectory.get("ticks"), list):
        rows = []
        for tick in trajectory["ticks"]:
            if isinstance(tick, dict):
                rows.extend(tick.get("messages") or ())
    if not isinstance(rows, list):
        return []
    tool_names: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for call in row.get("tool_calls") or ():
            if isinstance(call, dict) and call.get("id"):
                tool_names[str(call["id"])] = str(call.get("name") or "tool")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            result.append({"turn_index": index, "role": "unknown", "kind": "message", "content": "Unsupported message shape", "raw": redact_secrets(row)})
            continue
        turn = row.get("turn_idx")
        if type(turn) is not int:
            turn = index
        tool_messages = row.get("tool_messages")
        if isinstance(tool_messages, list):
            for tool_row in tool_messages:
                if isinstance(tool_row, dict):
                    _append_tool_result(result, tool_row, turn, tool_names)
            continue
        role = str(row.get("role", "unknown"))
        if role == "tool":
            _append_tool_result(result, row, turn, tool_names)
            continue
        calls = row.get("tool_calls") or []
        content = row.get("content")
        if row.get("is_audio") is True and not content:
            content = "[音频消息；不在文本视图中播放]"
        result.append({
            "turn_index": turn if type(turn) is int else index,
            "role": role,
            "kind": "message",
            "content": redact_secrets(content) if isinstance(content, str) else "",
            "timestamp": row.get("timestamp"),
            "tool_calls": [
                {
                    "id": call.get("id"), "name": call.get("name", "tool"),
                    "arguments": redact_secrets(call.get("arguments", {})),
                    "requestor": call.get("requestor", role),
                }
                for call in calls if isinstance(call, dict)
            ],
            "raw": redact_secrets(row),
        })
        for call in calls:
            if isinstance(call, dict):
                result.append({
                    "turn_index": turn if type(turn) is int else index,
                    "role": "tool", "kind": "tool_call",
                    "tool_name": call.get("name", "tool"),
                    "arguments": redact_secrets(call.get("arguments", {})),
                    "tool_id": call.get("id"), "raw": redact_secrets(call),
                })
    return result


def redact_secrets(value: Any, *, _key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact_secrets(item, _key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, _key=_key) for item in value]
    if isinstance(value, str):
        if _SECRET_KEY.search(_key):
            return "[REDACTED]"
        return _SECRET_TEXT.sub(lambda match: (match.group(1) or "") + "[REDACTED]", value)
    return value


def _append_tool_result(
    target: list[dict[str, Any]], row: dict[str, Any], turn: Any, names: dict[str, str],
) -> None:
    tool_id = str(row.get("id", ""))
    tool_turn = turn if type(turn) is int else row.get("turn_idx")
    if type(tool_turn) is not int:
        tool_turn = 0
    target.append({
        "turn_index": tool_turn,
        "role": "tool", "kind": "tool_result",
        "tool_name": names.get(tool_id, row.get("name", "tool result")),
        "tool_id": tool_id, "content": redact_secrets(row.get("content", "")),
        "error": row.get("error") is True,
        "requestor": row.get("requestor"), "raw": redact_secrets(row),
    })


def _reward_success(reward: Any) -> bool | None:
    return reward >= 1.0 if isinstance(reward, (float, int)) and not isinstance(reward, bool) else None


def _generation_from_panel(panel_name: Any) -> int | None:
    match = re.search(r"generation[-_: ]*(\d+)", str(panel_name or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _failure_category(record: dict[str, Any]) -> str | None:
    labels = {
        "missing_identity_verification": "未完成身份/位置验证",
        "missing_explicit_confirmation": "写入前未取得明确确认",
        "incomplete_write_scope": "未收集完整修改范围便执行写入",
    }
    return labels.get(str(record.get("mistake_type")))


def _strategy_id(strategy: dict[str, Any]) -> str:
    return sha256_json(strategy)[:16]


def _sum_if_available(rows: list[dict[str, Any]], field: str) -> int | None:
    if any(row.get("usage_unavailable", 0) for row in rows):
        return None
    values = [row.get(field) for row in rows]
    return sum(values) if all(type(item) is int for item in values) else None


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
    if sealed and isinstance(value.get("task_semantic_review"), dict):
        # The review binds heldout task IDs and pairings; don't let Research mode
        # reveal those identities before the registered release condition.
        value["task_semantic_review"] = {"status": "[SEALED]"}
        value.pop("task_semantic_review_sha256", None)
    return value
