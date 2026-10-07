"""Thin native τ-bench episode runner used by the alternating method."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .alternating import _write_json_atomic, _write_json_once
from .alternating_manifest import AlternatingManifest
from .budget import (
    BudgetSnapshot,
    EpisodeUsageTracker,
    ModelUsageSnapshot,
    RequestBudget,
)
from .communication import observe_communication_protocol
from .episode_execution import StopBeforeEpisodeDispatch
from .observability import EpisodeTelemetry
from .provider_diagnostics import safe_error
from .records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from .service_skills import (
    ServiceSkillMemory,
    ServiceSkillMemoryV2,
    render_selected_service_skills,
    skill_token_count,
)
from .strategies import (
    PromptStrategy,
    ServiceCarrier,
    render_customer_strategy,
    render_service_strategy,
)
from .tau_adapter import build_phase0_orchestrator, run_with_budget
from .tau_provenance import (
    role_model_args_for_runtime,
    sha256_json,
    write_manifest_once,
)


class NativeEpisodeRunError(RuntimeError):
    """A native run failed while preserving its actionable original exception text."""

    def __init__(
        self, original: Exception, *, task_id: str, panel_name: str, diagnostics_ref: str,
    ) -> None:
        self.task_id = task_id
        self.panel_name = panel_name
        self.diagnostics_ref = diagnostics_ref
        self.original_type = type(original).__name__
        self.original_message = safe_error(original)
        super().__init__(f"{self.original_type}: {self.original_message}")


class TauBenchEpisodeRunner:
    """Run and save one native τ-bench episode for alternating evolution."""

    def __init__(
        self,
        *,
        manifest: AlternatingManifest,
        config: Mapping[str, Any],
        data_dir: str | Path,
        request_budget: RequestBudget,
        output_directory: str | Path | None = None,
        stop_before_next_episode_file: str | Path | None = None,
        include_heldout: bool = False,
        task_objects: Mapping[str, Any] | None = None,
    ) -> None:
        if not manifest.real_provider_enabled:
            raise RuntimeError("native τ-bench runs require real_provider_enabled")
        if request_budget.snapshot().cap != manifest.request_budget_cap:
            raise ValueError("request budget cap must match the frozen alternating manifest")
        self.v2_policy = None if manifest.skill_evolution_v2_json is None else json.loads(manifest.skill_evolution_v2_json)
        self.models = dict(manifest.role_models)
        if set(self.models) != {"agent", "customer", "evaluator", "evolver"} or any(
            not self.models[name] for name in self.models
        ):
            raise ValueError(
                "alternating runs require frozen agent, customer, evaluator, and evolver models"
            )
        self.model_args = role_model_args_for_runtime(manifest.role_model_args)
        experiment = config.get("experiment", {})
        config_manifest = AlternatingManifest.from_mapping(config)
        if experiment.get("id") != manifest.experiment_id:
            raise ValueError("run configuration does not match the frozen alternating manifest")
        config_manifest.bind_saved_provenance(manifest.to_document())
        self.manifest = manifest
        self.request_budget = request_budget
        self.data_root = Path(data_dir).expanduser().resolve()
        if task_objects is None:
            raise ValueError("pass the split-aware E/V or final H task panel explicitly")
        allowed_task_ids = set(
            manifest.evolution_task_ids + manifest.validation_task_ids
            + (manifest.heldout_task_ids if include_heldout else ())
        )
        self.tasks = {str(key): value for key, value in task_objects.items()}
        if set(self.tasks) - allowed_task_ids:
            raise ValueError("loaded task objects include a task outside the active panels")
        self.service_policy_text = (
            self.data_root / "tau2/domains/retail/policy.md"
        ).read_text(encoding="utf-8")
        self.output_directory = Path(output_directory or manifest.output_path)
        self.stop_before_next_episode_file = (
            None if stop_before_next_episode_file is None
            else Path(stop_before_next_episode_file).expanduser().absolute()
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._completed_episode_cache: dict[
            str, tuple[EpisodeRecord, BudgetSnapshot, bool]
        ] = {}
        self._completed_episode_cache_lock = Lock()
        self._condition_locks: dict[str, Lock] = {}
        self._panel_reference_lock = Lock()
        manifest_path = self.output_directory / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.manifest = manifest.bind_saved_provenance(saved)
        else:
            other_files = tuple(self.output_directory.iterdir())
            if other_files:
                raise FileExistsError("refusing to use a non-empty run directory without its manifest")
            write_manifest_once(manifest_path, manifest)
        self._load_completed_episode_cache()

    def __call__(
        self,
        *,
        task_id: str,
        seed: int,
        customer: PromptStrategy | None,
        service: ServiceCarrier,
        panel_name: str,
    ) -> EpisodeRecord:
        expected_carrier = (
            ServiceSkillMemoryV2 if self.manifest.service_carrier == "skill_memory_v2" else
            ServiceSkillMemory
            if self.manifest.service_carrier == "skill_memory_v1"
            else PromptStrategy
        )
        if not isinstance(service, expected_carrier):
            raise TypeError(f"episode Service must use the {self.manifest.service_carrier} carrier")
        if str(task_id) not in self.tasks or seed < 0 or not panel_name.strip():
            raise ValueError("native episode task, seed and panel name must be valid")
        key_sha = sha256_json(self._episode_key(task_id, seed, customer, service))
        with self._completed_episode_cache_lock:
            condition_lock = self._condition_locks.setdefault(key_sha, Lock())
        # Identical concurrent jobs share a completed condition. Reservation is
        # acquired only after the cache check, never for a cached episode.
        with condition_lock:
            with self._completed_episode_cache_lock:
                cached = self._completed_episode_cache.get(key_sha)
            reused = cached is not None
            if cached is not None:
                record, usage, recovered = cached
                if recovered:
                    self.request_budget.absorb_usage(usage)
                    with self._completed_episode_cache_lock:
                        self._completed_episode_cache[key_sha] = (record, usage, False)
            else:
                with self.request_budget.track_episode_usage() as episode_usage:
                    reservation = self.request_budget.reserve_episode_dispatch()
                    with self.request_budget.use_episode_reservation(reservation):
                        record = self._run_episode(
                            task_id=task_id, seed=seed, customer=customer, service=service,
                            panel_name=panel_name, episode_usage=episode_usage,
                        )
            self._record_panel_reference(panel_name, key_sha, record, reused=reused)
            return record

    @staticmethod
    def _episode_key(
        task_id: str, seed: int, customer: PromptStrategy | None, service: ServiceCarrier,
    ) -> dict[str, Any]:
        return {
            "task_id": str(task_id), "seed": seed,
            "customer_strategy_id": customer_strategy_id(customer),
            "service_strategy_id": service_strategy_id(service),
        }

    def has_completed_episode(
        self, *, task_id: str, seed: int, customer: PromptStrategy | None,
        service: ServiceCarrier, panel_name: str,
    ) -> bool:
        key = self._episode_key(task_id, seed, customer, service)
        with self._completed_episode_cache_lock:
            return sha256_json(key) in self._completed_episode_cache

    def _record_panel_reference(
        self, panel_name: str, key_sha: str, record: EpisodeRecord, *, reused: bool,
    ) -> None:
        path = self.output_directory / "episode-panel-references.json"
        reference_key = sha256_json({"panel_name": panel_name, "condition_sha256": key_sha})
        with self._panel_reference_lock:
            document = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                "schema_version": 1, "manifest_sha256": self.manifest.sha256, "references": {},
            }
            if document.get("manifest_sha256") != self.manifest.sha256:
                raise ValueError("panel references belong to a different frozen manifest")
            references = document["references"]
            if reference_key not in references:
                references[reference_key] = {
                    "panel_name": panel_name, "condition_sha256": key_sha,
                    "episode_id": record.episode_id, "task_id": record.task_id,
                    "seed": record.seed, "reused": reused,
                }
                _write_json_atomic(path, document)

    def _run_episode(
        self,
        *,
        task_id: str,
        seed: int,
        customer: PromptStrategy | None,
        service: ServiceCarrier,
        panel_name: str,
        episode_usage: EpisodeUsageTracker,
    ) -> EpisodeRecord:
        try:
            task = self.tasks[str(task_id)]
        except KeyError as exc:
            raise ValueError(f"task {task_id!r} is outside the frozen task panels") from exc
        if seed < 0 or not panel_name.strip():
            raise ValueError("native episode seed and panel name must be valid")
        episode_key = self._episode_key(task_id, seed, customer, service)
        episode_key_sha256 = sha256_json(episode_key)
        if self.stop_before_next_episode_file is not None:
            signal = self.stop_before_next_episode_file
            if signal.is_symlink():
                raise RuntimeError("pause signal path cannot be a symlink")
            if signal.exists():
                raise StopBeforeEpisodeDispatch(
                    "paused safely before dispatching the next native episode"
                )
        budget_remaining = self.request_budget.snapshot().remaining
        if (
            budget_remaining is not None
            and budget_remaining <= 0
            and not self.request_budget.has_current_episode_reservation()
        ):
            raise RuntimeError(
                "native episode refused before dispatch: request budget is exhausted"
            )

        attempt_id = uuid4().hex
        episode_directory = self.output_directory / "episodes" / attempt_id
        episode_directory.mkdir(parents=True, exist_ok=False)
        simulation_path = episode_directory / "native-simulation.json"
        record_path = episode_directory / "episode-record.json"
        telemetry_path = episode_directory / "run-telemetry.json"
        live = EpisodeTelemetry(
            episode_directory / "active-episode.json", manifest_sha256=self.manifest.sha256,
            attempt_id=attempt_id, task_id=str(task_id), panel_name=panel_name,
        )
        before = self.request_budget.snapshot()
        prompt_hashes: dict[str, Any] = {}
        simulation_payload: dict[str, Any] | None = None
        orchestrator = None
        activation_rows = []
        activation_options = None
        runtime_service = service
        if isinstance(service, ServiceSkillMemoryV2):
            from .skill_activation import RenderAllSkillSelector, SkillActivator

            def activation_sink(value):
                value = {
                    "schema_version": 3,
                    "manifest_sha256": self.manifest.sha256,
                    "episode_key_sha256": episode_key_sha256,
                    "turn": len(activation_rows),
                    **value,
                }
                value["artifact_sha256"] = sha256_json(value)
                _write_json_once(
                    episode_directory
                    / "skill-activation"
                    / f"turn-{len(activation_rows):04d}.json",
                    value,
                )
                activation_rows.append(value)

            selector = (
                RenderAllSkillSelector()
                if self.v2_policy["service_skill_runtime"] == "render_all_v1"
                else SkillActivator(
                    model=self.v2_policy["activator"]["model"],
                    model_args=(
                        self.v2_policy["activator"]["model_args"]
                        if self.v2_policy["activator"]["model_args"].get("api_protocol")
                        == "responses"
                        else role_model_args_for_runtime(
                            (
                                (
                                    "activator",
                                    tuple(
                                        self.v2_policy["activator"][
                                            "model_args"
                                        ].items()
                                    ),
                                ),
                            )
                        )["activator"]
                    ),
                    request_budget=self.request_budget,
                )
            )
            activation_options = {
                "activator": selector,
                "max_active_skills": self.v2_policy["max_active_service_skills"],
                "sink": activation_sink,
            }

        def on_orchestrator(orchestrator: Any) -> None:

            prompt_hashes["agent"] = hashlib.sha256(
                orchestrator.agent.system_prompt.encode("utf-8")
            ).hexdigest()
            native_agent_class = type(orchestrator.agent).__mro__[1]
            native_agent_property = getattr(native_agent_class, "system_prompt", None)
            if (
                not isinstance(native_agent_property, property)
                or native_agent_property.fget is None
            ):
                raise TypeError(
                    "EvoTau Service wrapper no longer directly subclasses the native LLMAgent"
                )
            native_agent_prompt = native_agent_property.fget(orchestrator.agent)
            service_block = render_service_strategy(runtime_service)
            expected_agent_prompt = (
                native_agent_prompt if not service_block
                else f"{native_agent_prompt}\n\n{service_block}"
            )
            if orchestrator.agent.system_prompt != expected_agent_prompt:
                raise ValueError(
                    "EvoTau Service overlay changed the native tau-bench system prompt"
                )
            prompt_hashes["service_native_system_prompt"] = hashlib.sha256(
                native_agent_prompt.encode("utf-8")
            ).hexdigest()
            prompt_hashes["service_overlay"] = hashlib.sha256(
                service_block.encode("utf-8")
            ).hexdigest()
            if hasattr(service, "skills"):
                prompt_hashes["service_skill_count"] = len(service.skills)
                prompt_hashes["service_skill_rendered_chars"] = len(service_block)
                prompt_hashes["service_skill_rendered_tokens"] = (
                    skill_token_count(service_block)
                    if isinstance(service, ServiceSkillMemoryV2)
                    else None
                )
            prompt_hashes["customer"] = hashlib.sha256(
                orchestrator.user.system_prompt.encode("utf-8")
            ).hexdigest()
            # Test-only injected orchestrators may not model the native tau
            # UserSimulator object. The concrete builder always provides
            # `instructions` and the direct subclass property below.
            if not hasattr(orchestrator.user, "instructions"):
                return
            native_user_class = type(orchestrator.user).__mro__[1]
            native_property = getattr(native_user_class, "system_prompt", None)
            if (
                not isinstance(native_property, property)
                or native_property.fget is None
            ):
                raise TypeError(
                    "EvoTau Customer wrapper no longer directly subclasses the native UserSimulator"
                )
            native_prompt = native_property.fget(orchestrator.user)
            skill_block = render_customer_strategy(customer)
            expected_prompt = (
                native_prompt
                if not skill_block
                else f"{native_prompt}\n\n{skill_block}"
            )
            if orchestrator.user.system_prompt != expected_prompt:
                raise ValueError(
                    "EvoTau Customer overlay changed the native tau-bench system prompt"
                )
            source_scenario = str(task.user_scenario)
            runtime_scenario = getattr(orchestrator.user, "instructions", None)
            if runtime_scenario != source_scenario:
                raise ValueError(
                    "EvoTau Customer wrapper changed the native task scenario"
                )
            prompt_hashes["customer_native_system_prompt"] = hashlib.sha256(
                native_prompt.encode("utf-8")
            ).hexdigest()
            prompt_hashes["customer_scenario_source"] = hashlib.sha256(
                source_scenario.encode("utf-8")
            ).hexdigest()
            prompt_hashes["customer_scenario_runtime"] = hashlib.sha256(
                str(runtime_scenario).encode("utf-8")
            ).hexdigest()

        def on_simulation(simulation: Any) -> None:
            nonlocal simulation_payload
            simulation_payload = simulation.model_dump(mode="json")
            _write_json_once(simulation_path, simulation_payload)

        with self.request_budget.record_provider_calls(
            episode_directory / "provider-calls.jsonl"
        ):
            try:
                orchestrator = build_phase0_orchestrator(
                    task=deepcopy(task),
                    agent_model=self.models["agent"],
                    customer_model=self.models["customer"],
                    agent_model_args=self.model_args["agent"],
                    customer_model_args=self.model_args["customer"],
                    seed=seed,
                    max_steps=self.manifest.max_steps,
                    customer_strategy=customer,
                    service_strategy=runtime_service,
                    **({"service_activation": activation_options} if activation_options is not None else {}),
                    enforce_communication_protocol=self.manifest.enforce_communication_protocol,
                )
                on_orchestrator(orchestrator)
                with live.observe(orchestrator):
                    simulation = run_with_budget(
                        orchestrator,
                        self.request_budget,
                        evaluator_model=self.models["evaluator"],
                        evaluator_model_args=self.model_args["evaluator"],
                        run_reviewer=False,
                        on_simulation=on_simulation,
                    )
                simulation_payload = simulation.model_dump(mode="json")
                if simulation_payload.get("task_id") is None or str(
                    simulation_payload["task_id"]
                ) != str(task_id):
                    raise ValueError(
                        "native τ-bench returned a simulation for a different task"
                    )
                actual_seed = simulation_payload.get("seed")
                if actual_seed is not None and int(actual_seed) != seed:
                    raise ValueError(
                        "native τ-bench returned a simulation for a different seed"
                    )
                if simulation_payload.get("reward_info") is None:
                    reward = None
                else:
                    reward = simulation_payload["reward_info"].get("reward")
                if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
                    raise ValueError("native task-success score is missing or invalid; episode is unscored")
                else:
                    native_reward = float(reward)
                    task_success = native_reward >= 1.0
                    status = EpisodeStatus.COMPLETE

                if simulation_payload is None:
                    raise RuntimeError(
                        "native τ-bench did not provide a serialized simulation"
                    )
                protocol_observation = observe_communication_protocol(
                    simulation_payload.get("messages") or (),
                    enforcement_enabled=self.manifest.enforce_communication_protocol,
                )
                episode_id = str(simulation_payload.get("id") or attempt_id)
                record = EpisodeRecord(
                    episode_id=episode_id,
                    task_id=str(task_id),
                    seed=seed,
                    customer_strategy_id=customer_strategy_id(customer),
                    service_strategy_id=service_strategy_id(service),
                    status=status,
                    task_success=task_success,
                    native_reward=native_reward,
                    termination_reason=simulation_payload.get("termination_reason"),
                    trajectory_ref=simulation_path.relative_to(
                        self.output_directory
                    ).as_posix(),
                    tool_calls=_count_tool_calls(
                        simulation_payload.get("messages") or ()
                    ),
                    enforce_communication_protocol=self.manifest.enforce_communication_protocol,
                    mixed_text_tool_call_messages=(
                        protocol_observation["mixed_text_tool_call_message_count"]
                    ),
                    raw_review={},
                    **(
                        {
                            "total_steps": max(
                                (
                                    m.get("turn_idx", i)
                                    for i, m in enumerate(
                                        simulation_payload.get("messages") or ()
                                    )
                                ),
                                default=-1,
                            )
                            + 1,
                            "prompt_tokens": episode_usage.snapshot(
                                cap=None
                            ).prompt_tokens,
                            "completion_tokens": episode_usage.snapshot(
                                cap=None
                            ).completion_tokens,
                            "tool_errors": sum(
                                1
                                for m in simulation_payload.get("messages", [])
                                if m.get("role") == "tool" and m.get("error")
                            ),
                            "activated_skill_ids": tuple(
                                sorted(
                                    {
                                        ident
                                        for row in activation_rows
                                        for ident in row["decision"]["active_skill_ids"]
                                    }
                                )
                            ),
                        }
                        if isinstance(service, ServiceSkillMemoryV2)
                        else {}
                    ),
                )
                after = self.request_budget.snapshot()
                exact_episode_usage = episode_usage.snapshot(
                    cap=self.request_budget.snapshot().cap
                )
                _write_json_once(
                    telemetry_path,
                    {
                        "attempt_id": attempt_id,
                        "simulation_id": episode_id,
                        "episode_key": episode_key,
                        "episode_key_sha256": episode_key_sha256,
                        "panel_name": panel_name,
                        "customer_strategy_id": record.customer_strategy_id,
                        "service_strategy_id": record.service_strategy_id,
                        "rendered_prompt_sha256": prompt_hashes,
                        "budget_before": before.to_dict(),
                        "budget_after": after.to_dict(),
                        "budget_delta": _snapshot_delta(before, after),
                        "episode_budget_delta": exact_episode_usage.to_dict(),
                        "communication_protocol_observation": protocol_observation,
                    },
                )
                if isinstance(service, ServiceSkillMemoryV2):
                    activation_trace = {
                        "schema_version": 3,
                        "manifest_sha256": self.manifest.sha256,
                        "episode_key_sha256": episode_key_sha256,
                        "episode_id": record.episode_id,
                        "activation_mode": self.v2_policy["service_skill_runtime"],
                        "activator": self.v2_policy["activator"],
                        "decisions": activation_rows,
                        "activation_catalog_tokens": sum(
                            row["activation_catalog_tokens"] for row in activation_rows
                        ),
                        "activated_guidance_tokens": sum(
                            row["activated_guidance_tokens"] for row in activation_rows
                        ),
                        "rendered_skill_tokens": sum(
                            row["activated_guidance_tokens"] for row in activation_rows
                        ),
                        "full_memory_tokens": skill_token_count(
                            render_selected_service_skills(
                                service, [s.skill_id for s in service.skills]
                            )
                        ),
                    }
                    activation_trace["artifact_sha256"] = sha256_json(activation_trace)
                    _write_json_once(
                        episode_directory / "skill-activation-trace.json",
                        activation_trace,
                    )
                _write_json_once(record_path, record.to_dict())
                with self._completed_episode_cache_lock:
                    self._completed_episode_cache[episode_key_sha256] = (
                        record, exact_episode_usage, False,
                    )
                live.publish("complete", activity="native_evaluation_complete")
                return record
            except BaseException as exc:
                live.publish("failed", activity="native_episode_failed")
                partial_error = None
                partial_path = episode_directory / "partial-simulation.json"
                if orchestrator is not None:
                    try:
                        getter = getattr(orchestrator, "get_trajectory", None)
                        messages = getter() if callable(getter) else getattr(orchestrator, "trajectory", [])
                        _write_json_once(partial_path, {
                            "id": str(getattr(orchestrator, "simulation_id", None) or attempt_id),
                            "task_id": str(task_id), "seed": seed, "status": "incomplete",
                            "attempt_id": attempt_id,
                            "messages": [
                                message.model_dump(mode="json") if hasattr(message, "model_dump")
                                else dict(message) for message in messages
                            ],
                        })
                    except Exception as diagnostic_error:  # noqa: BLE001 - preserve the original native error
                        partial_error = safe_error(diagnostic_error)
                after = self.request_budget.snapshot()
                exact_episode_usage = episode_usage.snapshot(
                    cap=self.request_budget.snapshot().cap
                )
                _write_json_once(
                    episode_directory / "incomplete-run.json",
                    {
                        "attempt_id": attempt_id,
                        "task_id": str(task_id),
                        "seed": seed,
                        "episode_key": episode_key,
                        "episode_key_sha256": episode_key_sha256,
                        "panel_name": panel_name,
                        **_exception_details(exc),
                        "native_simulation_saved": simulation_path.exists(),
                        "partial_trajectory_ref": (
                            partial_path.relative_to(self.output_directory).as_posix()
                            if partial_path.exists()
                            else None
                        ),
                        "partial_trajectory_error": partial_error,
                        "provider_calls_ref": (
                            (episode_directory / "provider-calls.jsonl")
                            .relative_to(self.output_directory)
                            .as_posix()
                        ),
                        "communication_protocol_observation": (
                            None
                            if simulation_payload is None
                            else observe_communication_protocol(
                                simulation_payload.get("messages") or (),
                                enforcement_enabled=self.manifest.enforce_communication_protocol,
                            )
                        ),
                        "rendered_prompt_sha256": prompt_hashes,
                        "budget_before": before.to_dict(),
                        "budget_after": after.to_dict(),
                        "budget_delta": _snapshot_delta(before, after),
                        "episode_budget_delta": exact_episode_usage.to_dict(),
                    },
                )
                if not isinstance(exc, Exception):
                    exc.task_id = str(task_id)
                    exc.panel_name = panel_name
                    exc.diagnostics_ref = (episode_directory / "incomplete-run.json").relative_to(self.output_directory).as_posix()
                    raise
                raise NativeEpisodeRunError(
                    exc,
                    task_id=str(task_id),
                    panel_name=panel_name,
                    diagnostics_ref=(episode_directory / "incomplete-run.json")
                    .relative_to(self.output_directory)
                    .as_posix(),
                ) from exc

    def _load_completed_episode_cache(self) -> None:
        episode_root = self.output_directory / "episodes"
        if not episode_root.exists():
            return
        if episode_root.is_symlink() or not episode_root.is_dir():
            raise ValueError("native episode cache root must be a regular directory")
        for directory in sorted(episode_root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("native episode cache contains a non-directory entry")
            incomplete_path = directory / "incomplete-run.json"
            telemetry_path = directory / "run-telemetry.json"
            record_path = directory / "episode-record.json"
            if incomplete_path.exists():
                if not incomplete_path.is_file():
                    raise ValueError("native incomplete-run artifact is not a regular file")
                incomplete = json.loads(incomplete_path.read_text(encoding="utf-8"))
                key = incomplete.get("episode_key")
                key_sha = incomplete.get("episode_key_sha256")
                if isinstance(key, dict) and key_sha != sha256_json(key):
                    raise ValueError("incomplete native episode has an invalid frozen episode key")
                # An incomplete attempt is diagnostic evidence, not a cache hit.
                # Re-dispatching this exact frozen key is safe; successful keys
                # are still loaded below and reused without provider calls.
                continue
            if not telemetry_path.exists() and not record_path.exists():
                continue
            if (not telemetry_path.is_file() or not record_path.is_file()
                    or (directory / "native-simulation.json").is_symlink()):
                raise ValueError("native episode cache has incomplete telemetry or record artifacts")
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            key = telemetry.get("episode_key")
            key_sha = telemetry.get("episode_key_sha256")
            if not isinstance(key, dict) or key_sha != sha256_json(key):
                continue
            record = EpisodeRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))
            if record.task_id not in self.tasks:
                # Initial E/V resume must not deserialize H conversations. The
                # final runner can load them after fresh challenge generation.
                continue
            if record.status != EpisodeStatus.COMPLETE or type(record.task_success) is not bool:
                raise ValueError("native cache contains an incomplete or unscored episode")
            if set(key) != {"task_id", "seed", "customer_strategy_id", "service_strategy_id"}:
                raise ValueError("native cache uses a different condition schema")
            self.load_trajectory(record)
            if key_sha in self._completed_episode_cache:
                raise ValueError("native episode cache contains duplicate frozen episode keys")
            if (key.get("task_id") != record.task_id or key.get("seed") != record.seed
                    or key.get("customer_strategy_id") != record.customer_strategy_id
                    or key.get("service_strategy_id") != record.service_strategy_id):
                raise ValueError("native episode cache key differs from its saved EpisodeRecord")
            before = BudgetSnapshot(**telemetry["budget_before"])
            after = BudgetSnapshot(**telemetry["budget_after"])
            saved_episode_usage = telemetry.get("episode_budget_delta")
            usage = (
                BudgetSnapshot(**saved_episode_usage)
                if isinstance(saved_episode_usage, dict)
                else _budget_snapshot_difference(after, before)
            )
            self._completed_episode_cache[key_sha] = (
                record, usage, not self.request_budget.live_usage_restored,
            )

    def load_trajectory(self, episode: EpisodeRecord) -> Mapping[str, Any] | None:
        """Load one saved native simulation after enforcing output-directory containment."""
        if episode.trajectory_ref is None:
            return None
        output_root = self.output_directory.resolve()
        trajectory_path = (output_root / episode.trajectory_ref).resolve()
        try:
            trajectory_path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError("episode trajectory reference escapes its run directory") from exc
        if not trajectory_path.is_file():
            raise FileNotFoundError("episode trajectory referenced by repair evidence is unavailable")
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict)
                or str(payload.get("id")) != episode.episode_id
                or str(payload.get("task_id")) != episode.task_id
                or int(payload.get("seed", -1)) != episode.seed):
            raise ValueError("saved native simulation does not match its EpisodeRecord")
        if not isinstance(payload.get("messages"), list):
            raise TypeError("saved native simulation is missing its message list")
        return payload


def _count_tool_calls(messages: Any) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "tool":
            count += 1
        elif role == "multi_tool":
            count += len(message.get("tool_messages") or ())
    return count


def _snapshot_delta(before: BudgetSnapshot, after: BudgetSnapshot) -> dict[str, int]:
    return {
        name: getattr(after, name) - getattr(before, name)
        for name in (
            "attempts", "successes", "failures", "denied", "prompt_tokens",
            "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
        )
    }


def _budget_snapshot_difference(after: BudgetSnapshot, before: BudgetSnapshot) -> BudgetSnapshot:
    """Return a valid additive usage snapshot for a completed native episode."""

    names = (
        "attempts", "successes", "failures", "denied", "prompt_tokens",
        "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
    )
    differences = {name: getattr(after, name) - getattr(before, name) for name in names}
    if any(value < 0 for value in differences.values()):
        raise ValueError("native episode budget counters moved backwards")
    before_models = {row.model_id: row for row in before.model_usage}
    after_models = {row.model_id: row for row in after.model_usage}
    model_rows = []
    for model_id in sorted(set(before_models) | set(after_models)):
        old = before_models.get(model_id, ModelUsageSnapshot(model_id))
        new = after_models.get(model_id, ModelUsageSnapshot(model_id))
        values = {name: getattr(new, name) - getattr(old, name) for name in names}
        if any(value < 0 for value in values.values()):
            raise ValueError("native episode model counters moved backwards")
        if any(values.values()):
            model_rows.append(ModelUsageSnapshot(model_id=model_id, **values))
    if not model_rows and any(differences.values()):
        model_rows.append(ModelUsageSnapshot(model_id="__unattributed__", **differences))
    return BudgetSnapshot(cap=after.cap, model_usage=tuple(model_rows), **differences)


def _exception_details(exc: BaseException) -> dict[str, Any]:
    """Keep the failure reason and a short cause chain, with a bounded traceback."""
    chain = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 6:
        seen.add(id(current))
        chain.append({
            "type": type(current).__name__,
            "message": safe_error(current),
            "repr": safe_error(Exception(repr(current))),
        })
        current = current.__cause__ or current.__context__
    cause = exc.__cause__ or exc.__context__
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "failure_type": type(exc).__name__,
        "failure_message": safe_error(exc),
        "failure_repr": safe_error(Exception(repr(exc))),
        "cause_type": None if cause is None else type(cause).__name__,
        "cause_message": None if cause is None else safe_error(cause),
        "exception_chain": chain,
        "traceback": safe_error(Exception(formatted[-16_000:]), limit=16_000),
        "traceback_truncated": len(formatted) > 16_000,
    }
