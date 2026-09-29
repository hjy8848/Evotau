"""Preview and launch existing EvoTau CLI runners from the local Console."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..manifest import ExperimentManifest, MechanismManifest, PilotManifest, sha256_json
from ..native_runner import _validate_phase0_parent
from ..phase0 import load_config

_PLUGIN_SPEC = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z")


class RunManagerError(ValueError):
    """A Console run request failed a frozen-input or safety check."""


@dataclass(frozen=True, slots=True)
class RunPreview:
    phase: str
    experiment_id: str
    manifest_sha256: str
    launch_sha256: str
    output_path: str
    task_labels: tuple[str, ...]
    seeds: tuple[int, ...]
    generations: int
    customer_candidates: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    models: tuple[tuple[str, str], ...]
    real_provider_enabled: bool
    provider_plugin: str | None
    tau2_data_path: str | None
    phase0_result_path: str | None
    config_path: Path
    output_directory: Path
    manifest_document: dict[str, Any]

    def to_view(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "launch_sha256": self.launch_sha256,
            "output_path": self.output_path,
            "task_labels": self.task_labels,
            "seeds": self.seeds,
            "generations": self.generations,
            "customer_candidates": self.customer_candidates,
            "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap,
            "provider_retries": self.provider_retries,
            "models": self.models,
            "real_provider_enabled": self.real_provider_enabled,
            "provider_plugin": self.provider_plugin or "未设置",
            "tau2_data_path": self.tau2_data_path or "未设置",
            "phase0_result_path": self.phase0_result_path or "未设置",
            "enforce_communication_protocol": self.manifest_document.get(
                "enforce_communication_protocol",
            ),
        }


@dataclass(slots=True)
class _ManagedProcess:
    run_id: str
    phase: str
    output_directory: Path
    manifest_sha256: str
    pause_file: Path
    process: subprocess.Popen | None
    status: str


class RunManager:
    """Call Phase 0/3/Pilot entrypoints; do not make decisions in the web layer."""

    def __init__(
        self,
        project_root: str | Path,
        runs_root: str | Path,
        *,
        tau2_data_dir: str | Path | None = None,
        popen: Any = subprocess.Popen,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.tau2_data_dir = (
            Path(tau2_data_dir).expanduser().resolve() if tau2_data_dir else
            (Path(os.environ["TAU2_DATA_DIR"]).expanduser().resolve()
             if os.environ.get("TAU2_DATA_DIR") else None)
        )
        self._popen = popen
        self._jobs: dict[str, _ManagedProcess] = {}
        self._lock = threading.RLock()

    def preview(
        self,
        *,
        phase: str,
        config_path: str | Path,
        phase0_result_path: str | Path | None = None,
        provider_plugin: str | None = None,
        tau2_data_dir: str | Path | None = None,
    ) -> RunPreview:
        config = self._config_path(config_path)
        try:
            raw = load_config(config)
            if phase == "0-integration-proof":
                manifest = ExperimentManifest.from_mapping(raw)
                task_labels = tuple(
                    [f"E: {item}" for item in manifest.evolution_task_ids]
                    + [f"V: {item}" for item in manifest.validation_task_ids]
                )
                seeds = (manifest.seed,)
                generations = 1
                candidates = 0
            elif phase == "3-two-generation-smoke":
                manifest = MechanismManifest.from_mapping(raw)
                task_labels = (f"E: {manifest.evolution_task_id}", f"V: {manifest.validation_task_id}")
                seeds = (manifest.seed,)
                generations = manifest.generations
                candidates = manifest.customer_candidates
            elif phase == "4-pilot":
                manifest = PilotManifest.from_mapping(raw)
                heldout_count = len(manifest.heldout_task_ids)
                task_labels = (
                    *(f"E: {item}" for item in manifest.evolution_task_ids),
                    *(f"V: {item}" for item in manifest.validation_task_ids),
                    *((f"H panel sealed · {heldout_count} tasks",) if heldout_count else ()),
                )
                seeds = manifest.evolution_seeds
                generations = manifest.generations
                candidates = manifest.customer_candidates
            else:
                raise RunManagerError("请选择 Phase 0、Phase 3 或 Pilot 配置。")
        except RunManagerError:
            raise
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise RunManagerError(f"配置无法通过 EvoTau manifest 校验（{type(exc).__name__}）。") from exc

        if manifest.phase != phase:
            raise RunManagerError("所选入口和配置中的 phase 不一致。")
        output_directory = self._output_path(manifest.output_path)
        role_models = tuple(sorted((str(role), str(model) if model else "未冻结")
                                   for role, model in manifest.role_models))
        preview = RunPreview(
            phase=phase,
            experiment_id=manifest.experiment_id,
            manifest_sha256=manifest.sha256,
            launch_sha256=self._launch_fingerprint(
                manifest.sha256,
                provider_plugin=provider_plugin,
                tau2_data_dir=tau2_data_dir,
                phase0_result_path=phase0_result_path,
            ),
            output_path=str(output_directory),
            task_labels=task_labels,
            seeds=tuple(seeds),
            generations=generations,
            customer_candidates=candidates,
            max_episodes=manifest.max_episodes,
            request_budget_cap=manifest.request_budget_cap,
            provider_retries=manifest.provider_retries,
            models=role_models,
            real_provider_enabled=manifest.real_provider_enabled,
            provider_plugin=provider_plugin,
            tau2_data_path=(str(self._data_path(tau2_data_dir))
                            if self._data_path(tau2_data_dir) is not None else None),
            phase0_result_path=(str(self._contained_run_file(phase0_result_path))
                                if phase0_result_path else None),
            config_path=config,
            output_directory=output_directory,
            manifest_document=manifest.to_document(),
        )
        self._validate_launch_inputs(
            preview,
            phase0_result_path=phase0_result_path,
            provider_plugin=provider_plugin,
            tau2_data_dir=tau2_data_dir,
            for_start=False,
        )
        return preview

    def start(
        self,
        *,
        phase: str,
        config_path: str | Path,
        confirmed_manifest_sha256: str,
        confirmed_launch_sha256: str | None = None,
        phase0_result_path: str | Path | None = None,
        provider_plugin: str | None = None,
        tau2_data_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        preview = self.preview(
            phase=phase, config_path=config_path,
            phase0_result_path=phase0_result_path,
            provider_plugin=provider_plugin,
            tau2_data_dir=tau2_data_dir,
        )
        if preview.manifest_sha256 != confirmed_manifest_sha256:
            raise RunManagerError("配置或源码在预览后发生变化，请重新预览。")
        if (confirmed_launch_sha256 is None
                or not hmac.compare_digest(preview.launch_sha256, confirmed_launch_sha256)):
            raise RunManagerError("启动参数在预览后发生变化，请重新预览。")
        self._validate_launch_inputs(
            preview,
            phase0_result_path=phase0_result_path,
            provider_plugin=provider_plugin,
            tau2_data_dir=tau2_data_dir,
            for_start=True,
        )
        run_id = preview.output_directory.relative_to(self.runs_root).as_posix()
        with self._lock:
            current = self._job_status(run_id)
            if current == "running":
                raise RunManagerError("该 immutable run 已经在运行。")
            if self._completed_result_exists(preview):
                raise RunManagerError("该 immutable run 已完成，Console 只允许只读打开。")
            self._check_existing_output(preview)
            control_dir = self.runs_root / ".evotau-console" / "controls"
            control_dir.mkdir(parents=True, exist_ok=True)
            pause_file = control_dir / f"{preview.manifest_sha256}.stop"
            if pause_file.exists():
                if pause_file.is_symlink():
                    raise RunManagerError("暂停控制文件路径不安全。")
                pause_file.unlink()
            command = self._command(
                preview,
                phase0_result_path=phase0_result_path,
                provider_plugin=provider_plugin,
                tau2_data_dir=tau2_data_dir,
                pause_file=pause_file,
            )
            env = os.environ.copy()
            python_path = str(self.project_root / "src")
            env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
            try:
                process = self._popen(
                    command,
                    cwd=self.project_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
            except OSError as exc:
                raise RunManagerError(f"EvoTau runner could not be started ({type(exc).__name__}).") from exc
            job = _ManagedProcess(
                run_id=run_id,
                phase=phase,
                output_directory=preview.output_directory,
                manifest_sha256=preview.manifest_sha256,
                pause_file=pause_file,
                process=process,
                status="running",
            )
            self._jobs[run_id] = job
        return {"run_id": run_id, "status": "running", "preview": preview.to_view()}

    def pause(self, run_id: str) -> str:
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None or self._job_status(run_id) != "running":
                raise RunManagerError("当前没有可暂停的活动 run。")
            if job.phase.startswith("0-"):
                raise RunManagerError("Phase 0 只有一个 episode；请让它安全完成。")
            job.pause_file.parent.mkdir(parents=True, exist_ok=True)
            if not job.pause_file.exists():
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(job.pause_file, flags, 0o600)
                os.close(descriptor)
            return "已请求暂停。当前 episode 会完成；runner 将在下一次 episode 派发前停下。"

    def status_for_run(self, run_id: str) -> str | None:
        with self._lock:
            return self._job_status(run_id)

    def _job_status(self, run_id: str) -> str | None:
        job = self._jobs.get(run_id)
        if job is None:
            return None
        if job.process is None:
            return job.status
        code = job.process.poll()
        if code is None:
            job.status = "running"
        elif code == 75:
            job.status = "paused"
            job.process = None
        elif code == 0:
            job.status = "complete" if self._result_file(job) else "failed"
            job.process = None
        else:
            job.status = "failed"
            job.process = None
        return job.status

    def _validate_launch_inputs(
        self,
        preview: RunPreview,
        *,
        phase0_result_path: str | Path | None,
        provider_plugin: str | None,
        tau2_data_dir: str | Path | None,
        for_start: bool,
    ) -> None:
        if not preview.config_path.is_file() or preview.config_path.is_symlink():
            raise RunManagerError("实验配置必须是仓库内的普通文件。")
        if for_start:
            if not preview.real_provider_enabled:
                raise RunManagerError("该 frozen manifest 已禁用真实 provider；Console 不会替它启用。")
            if any(model == "未冻结" for _role, model in preview.models):
                raise RunManagerError("真实 run 需要在 frozen manifest 中固定所有角色模型。")
            data_dir = self._data_path(tau2_data_dir)
            if data_dir is None or not data_dir.is_dir() or data_dir.is_symlink():
                raise RunManagerError("启动前需要指定固定 τ-bench data 目录。")
            if (preview.phase != "0-integration-proof"
                    and (not provider_plugin or not _PLUGIN_SPEC.fullmatch(provider_plugin))):
                raise RunManagerError("Phase 3/Pilot 需要冻结 provider plugin 的 module:factory。")
        if preview.phase == "3-two-generation-smoke":
            if not phase0_result_path:
                if for_start:
                    raise RunManagerError("Phase 3 必须绑定一份已完成的 Phase 0 结果。")
                return
            path = self._contained_run_file(phase0_result_path)
            if for_start and not path.is_file():
                raise RunManagerError("Phase 0 parent result 不存在。")
            if path.is_file():
                try:
                    config = load_config(preview.config_path)
                    manifest = MechanismManifest.from_mapping(config)
                    _validate_phase0_parent(path, manifest)
                except Exception as exc:
                    if for_start:
                        raise RunManagerError(
                            f"Phase 0 parent 未通过绑定校验（{type(exc).__name__}）。"
                        ) from exc

    def _launch_fingerprint(
        self,
        manifest_sha256: str,
        *,
        provider_plugin: str | None,
        tau2_data_dir: str | Path | None,
        phase0_result_path: str | Path | None,
    ) -> str:
        data_path = None
        if tau2_data_dir:
            resolved_data_path = self._data_path(tau2_data_dir)
            data_path = str(resolved_data_path) if resolved_data_path is not None else None
        parent_path, parent_sha256 = None, None
        if phase0_result_path:
            source = self._contained_run_file(phase0_result_path)
            parent_path = str(source)
            parent_sha256 = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        return sha256_json({
            "manifest_sha256": manifest_sha256,
            "provider_plugin": provider_plugin,
            "tau2_data_path": data_path,
            "phase0_result_path": parent_path,
            "phase0_result_sha256": parent_sha256,
        })

    def _command(
        self,
        preview: RunPreview,
        *,
        phase0_result_path: str | Path | None,
        provider_plugin: str | None,
        tau2_data_dir: str | Path | None,
        pause_file: Path,
    ) -> list[str]:
        command = [sys.executable, "-m", {
            "0-integration-proof": "evotau.phase0_run",
            "3-two-generation-smoke": "evotau.phase3_run",
            "4-pilot": "evotau.pilot_run",
        }[preview.phase], "--config", str(preview.config_path)]
        data_dir = self._data_path(tau2_data_dir)
        if data_dir is not None:
            command.extend(("--tau2-data-dir", str(data_dir)))
        if preview.phase == "3-two-generation-smoke":
            command.extend(("--phase0-result", str(self._contained_run_file(phase0_result_path))))
        if preview.phase != "0-integration-proof":
            command.extend(("--provider-plugin", str(provider_plugin)))
            command.extend(("--stop-before-next-episode-file", str(pause_file)))
        return command

    def _config_path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        if candidate.is_symlink():
            raise RunManagerError("配置文件不能是符号链接。")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.project_root)
        except (OSError, ValueError) as exc:
            raise RunManagerError("配置文件必须位于 EvoTau 仓库内。") from exc
        if not resolved.is_file():
            raise RunManagerError("配置文件不存在。")
        return resolved

    def _output_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
            raise RunManagerError("manifest 输出路径不安全。")
        lexical_path = self.project_root / candidate
        current = lexical_path.absolute()
        while current != self.project_root.parent and current != current.parent:
            if current.is_symlink():
                raise RunManagerError("manifest 输出路径不能经过符号链接。")
            if current == self.runs_root:
                break
            current = current.parent
        path = lexical_path.resolve()
        try:
            path.relative_to(self.runs_root)
        except ValueError as exc:
            raise RunManagerError("manifest 输出必须位于 Console 的实验结果目录内。") from exc
        current = path
        while current != self.runs_root.parent and current.exists():
            if current.is_symlink():
                raise RunManagerError("manifest 输出路径不能经过符号链接。")
            current = current.parent
        return path

    def _contained_run_file(self, value: str | Path | None) -> Path:
        if value is None:
            raise RunManagerError("Phase 0 parent result is required.")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        if candidate.is_symlink():
            raise RunManagerError("parent artifact cannot be a symlink.")
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(self.runs_root)
        except ValueError as exc:
            raise RunManagerError("parent artifact must be under the local run directory.") from exc
        return resolved

    def _data_path(self, value: str | Path | None) -> Path | None:
        if value:
            candidate = Path(value).expanduser()
            return candidate.resolve()
        return self.tau2_data_dir

    @staticmethod
    def _completed_result_exists(preview: RunPreview) -> bool:
        name = {
            "0-integration-proof": "phase0-result.json",
            "3-two-generation-smoke": "phase3-result.json",
            "4-pilot": "pilot-result.json",
        }[preview.phase]
        path = preview.output_directory / name
        return path.is_file()

    @staticmethod
    def _result_file(job: _ManagedProcess) -> bool:
        name = "phase0-result.json" if job.phase.startswith("0-") else (
            "phase3-result.json" if job.phase.startswith("3-") else "pilot-result.json"
        )
        return (job.output_directory / name).is_file()

    @staticmethod
    def _check_existing_output(preview: RunPreview) -> None:
        output = preview.output_directory
        if not output.exists():
            return
        if output.is_symlink() or not output.is_dir():
            raise RunManagerError("immutable run output is not a regular directory.")
        manifest_name = "pilot-manifest.json" if preview.phase == "4-pilot" else "manifest.json"
        saved_path = output / manifest_name
        if not saved_path.exists():
            if any(output.iterdir()):
                raise RunManagerError("拒绝复用没有匹配 frozen manifest 的非空输出目录。")
            return
        if saved_path.is_symlink():
            raise RunManagerError("已有 manifest 路径不安全。")
        try:
            saved = load_config(saved_path) if saved_path.suffix in {".yaml", ".yml"} else __import__("json").loads(
                saved_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RunManagerError("已有 immutable manifest 无法读取。") from exc
        if saved != preview.manifest_document:
            raise RunManagerError("已有 immutable run 属于另一份 manifest，拒绝启动。")
