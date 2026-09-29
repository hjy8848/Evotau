"""Manifest-bound native τ-bench episode adapter for the Phase 1–3 controller."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from .budget import (
    BudgetSnapshot,
    EpisodeUsageTracker,
    ModelUsageSnapshot,
    RequestBudget,
)
from .checkpoint import load_checkpoint, manifest_fingerprint
from .communication import observe_communication_protocol
from .customer_evolver import LLMCustomerEvolver, OperatorSelector
from .db_state_trace import (
    DBStateTraceError,
    replay_db_state_trace,
    unavailable_trace,
    write_trace_once,
)
from .episode_execution import StopBeforeEpisodeDispatch
from .manifest import (
    MechanismManifest,
    PilotManifest,
    role_model_args_for_runtime,
    sha256_json,
    write_manifest_once,
)
from .phase0_run import _load_pinned_tasks, _write_json_once
from .records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from .strategies import CustomerStrategy, ServiceStrategy
from .tau_adapter import build_phase0_orchestrator, run_with_budget


@dataclass(frozen=True, slots=True)
class IndependentEpisodeAudit:
    """Independent human/model annotation; τ-bench's native review is not ground truth."""

    verifier_ref: str
    customer_valid: bool
    strategy_applicable: bool
    customer_strategy_adherent: bool | None
    policy_violation: bool
    invalid_repeated_write_calls: int
    policy_rule_id: str | None = None
    mistake_type: str | None = None
    workflow_stage: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.verifier_ref.strip():
            raise ValueError("independent episode audit requires a verifier reference")
        if type(self.customer_valid) is not bool or type(self.strategy_applicable) is not bool:
            raise ValueError("independent audit must explicitly judge Customer validity and strategy applicability")
        if self.strategy_applicable and type(self.customer_strategy_adherent) is not bool:
            raise ValueError("applicable strategy behavior requires an adherence judgment")
        if not self.strategy_applicable and self.customer_strategy_adherent is not None:
            raise ValueError("not_applicable strategy behavior must not be scored for adherence")
        if type(self.policy_violation) is not bool:
            raise ValueError("independent audit must explicitly judge policy violation")
        if (type(self.invalid_repeated_write_calls) is not int
                or self.invalid_repeated_write_calls < 0):
            raise ValueError("independent audit must report a non-negative repeated-write count")
        if self.policy_violation and not all(
            (self.policy_rule_id, self.mistake_type, self.workflow_stage, self.evidence)
        ):
            raise ValueError("policy violation audit requires a rule, mistake, stage, and trajectory evidence")


AuditProvider = Callable[
    [Any, Any, CustomerStrategy | None, ServiceStrategy, str], IndependentEpisodeAudit
]


class NativeEpisodeRunError(RuntimeError):
    """A native run failed; provider or credential details are kept out of records."""


class TauBenchEpisodeRunner:
    """Run native τ-bench text episodes for `TwoGenerationSmoke`.

    The callback is optional for collection-only work. Without independent
    annotations, returned records are explicitly uncertain and cannot promote
    failures or affect Customer fitness. The callback runs inside the shared
    LiteLLM request-budget instrumentation.
    """

    def __init__(
        self,
        *,
        manifest: MechanismManifest | PilotManifest,
        config: Mapping[str, Any],
        data_dir: str | Path,
        request_budget: RequestBudget,
        audit_provider: AuditProvider | None = None,
        service_token_counter: Callable[[str], int] | None = None,
        output_directory: str | Path | None = None,
        stop_before_next_episode_file: str | Path | None = None,
    ) -> None:
        if not manifest.real_provider_enabled:
            raise RuntimeError("native Phase 3 runs require explicit real_provider_enabled opt-in")
        if request_budget.snapshot().cap != manifest.request_budget_cap:
            raise ValueError("shared request budget cap must match the frozen mechanism manifest")
        self.models = dict(manifest.role_models)
        if set(self.models) != {"agent", "customer", "reviewer", "evaluator", "evolver"} or any(
            not self.models[name] for name in self.models
        ):
            raise ValueError(
                "native Phase 3 runs require frozen agent, customer, reviewer, evaluator, and evolver models"
            )
        self.model_args = {role: dict(args) for role, args in manifest.role_model_args}
        experiment = config.get("experiment", {})
        config_manifest = (
            PilotManifest.from_mapping(config)
            if isinstance(manifest, PilotManifest)
            else MechanismManifest.from_mapping(config)
        )
        if experiment.get("id") != manifest.experiment_id or config_manifest.sha256 != manifest.sha256:
            raise ValueError("run configuration does not match the frozen mechanism manifest")
        self.manifest = manifest
        self.request_budget = request_budget
        self.audit_provider = audit_provider
        self.service_token_counter = service_token_counter
        self.data_root = Path(data_dir).expanduser().resolve()
        self.tasks = _load_pinned_tasks(
            manifest,
            data_dir=self.data_root,
            task_selection=experiment.get("task_selection", {}),
            task_ids=(
                manifest.evolution_task_ids + manifest.validation_task_ids + manifest.heldout_task_ids
                if isinstance(manifest, PilotManifest)
                else (manifest.evolution_task_id, manifest.validation_task_id)
            ),
        )
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
        manifest_path = self.output_directory / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved != manifest.to_document():
                raise ValueError("existing run directory belongs to a different frozen manifest")
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
        customer: CustomerStrategy | None,
        service: ServiceStrategy,
        panel_name: str,
    ) -> EpisodeRecord:
        with self.request_budget.track_episode_usage() as episode_usage:
            return self._run_episode(
                task_id=task_id,
                seed=seed,
                customer=customer,
                service=service,
                panel_name=panel_name,
                episode_usage=episode_usage,
            )

    def has_completed_episode(
        self, *, task_id: str, seed: int, customer: CustomerStrategy | None,
        service: ServiceStrategy, panel_name: str,
    ) -> bool:
        key = {
            "task_id": str(task_id),
            "seed": seed,
            "customer_strategy_id": customer_strategy_id(customer),
            "service_strategy_id": service_strategy_id(service),
            "panel_name": panel_name,
        }
        with self._completed_episode_cache_lock:
            return sha256_json(key) in self._completed_episode_cache

    def _run_episode(
        self,
        *,
        task_id: str,
        seed: int,
        customer: CustomerStrategy | None,
        service: ServiceStrategy,
        panel_name: str,
        episode_usage: EpisodeUsageTracker,
    ) -> EpisodeRecord:
        try:
            task = self.tasks[str(task_id)]
        except KeyError as exc:
            raise ValueError(f"task {task_id!r} is outside the frozen task panels") from exc
        if seed < 0 or not panel_name.strip():
            raise ValueError("native episode seed and panel name must be valid")
        if service.rules and self.service_token_counter is None:
            raise ValueError("non-empty ServiceStrategy requires a model-matched token counter")
        episode_key = {
            "task_id": str(task_id),
            "seed": seed,
            "customer_strategy_id": customer_strategy_id(customer),
            "service_strategy_id": service_strategy_id(service),
            "panel_name": panel_name,
        }
        episode_key_sha256 = sha256_json(episode_key)
        with self._completed_episode_cache_lock:
            cached = self._completed_episode_cache.get(episode_key_sha256)
            if cached is not None and cached[2]:
                self._completed_episode_cache[episode_key_sha256] = (
                    cached[0], cached[1], False,
                )
        if cached is not None:
            record, usage, recovered = cached
            if recovered:
                try:
                    self.request_budget.absorb_usage(usage)
                except Exception:
                    with self._completed_episode_cache_lock:
                        self._completed_episode_cache[episode_key_sha256] = (
                            record, usage, True,
                        )
                    raise
            return record
        if self.stop_before_next_episode_file is not None:
            signal = self.stop_before_next_episode_file
            if signal.is_symlink():
                raise RuntimeError("pause signal path cannot be a symlink")
            if signal.exists():
                raise StopBeforeEpisodeDispatch(
                    "paused safely before dispatching the next native episode"
                )
        if (self.request_budget.snapshot().remaining <= 0
                and not self.request_budget.has_current_episode_reservation()):
            raise RuntimeError("native episode refused before dispatch: request budget is exhausted")

        attempt_id = uuid4().hex
        episode_directory = self.output_directory / "episodes" / attempt_id
        episode_directory.mkdir(parents=True, exist_ok=False)
        simulation_path = episode_directory / "native-simulation.json"
        record_path = episode_directory / "episode-record.json"
        telemetry_path = episode_directory / "run-telemetry.json"
        before = self.request_budget.snapshot()
        prompt_hashes: dict[str, str] = {}
        simulation_payload: dict[str, Any] | None = None
        audit_result: IndependentEpisodeAudit | None = None

        def on_orchestrator(orchestrator: Any) -> None:
            import hashlib

            prompt_hashes["agent"] = hashlib.sha256(
                orchestrator.agent.system_prompt.encode("utf-8")
            ).hexdigest()
            prompt_hashes["customer"] = hashlib.sha256(
                orchestrator.user.system_prompt.encode("utf-8")
            ).hexdigest()

        def on_simulation(simulation: Any) -> None:
            nonlocal simulation_payload
            simulation_payload = simulation.model_dump(mode="json")
            _write_json_once(simulation_path, simulation_payload)

        def after_review(simulation: Any, orchestrator: Any) -> None:
            nonlocal audit_result
            if self.audit_provider is None:
                return
            audit_result = self.audit_provider(
                simulation, orchestrator.task, customer, service, panel_name
            )
            if not isinstance(audit_result, IndependentEpisodeAudit):
                raise TypeError("audit_provider must return IndependentEpisodeAudit")

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
                service_strategy=service,
                service_token_counter=self.service_token_counter,
                enforce_communication_protocol=self.manifest.enforce_communication_protocol,
            )
            on_orchestrator(orchestrator)
            simulation = run_with_budget(
                orchestrator,
                self.request_budget,
                reviewer_model=self.models["reviewer"],
                reviewer_model_args=self.model_args["reviewer"],
                evaluator_model=self.models["evaluator"],
                evaluator_model_args=self.model_args["evaluator"],
                on_simulation=on_simulation,
                after_review=after_review,
            )
            simulation_payload = simulation.model_dump(mode="json")
            if simulation_payload.get("task_id") is None or str(simulation_payload["task_id"]) != str(task_id):
                raise ValueError("native τ-bench returned a simulation for a different task")
            actual_seed = simulation_payload.get("seed")
            if actual_seed is not None and int(actual_seed) != seed:
                raise ValueError("native τ-bench returned a simulation for a different seed")
            if simulation_payload.get("reward_info") is None:
                reward = None
            else:
                reward = simulation_payload["reward_info"].get("reward")
            if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
                native_reward = None
                task_success = None
                status = EpisodeStatus.UNCERTAIN
            else:
                native_reward = float(reward)
                task_success = native_reward >= 1.0
                if audit_result is None:
                    status = EpisodeStatus.UNCERTAIN
                elif not audit_result.customer_valid:
                    status = EpisodeStatus.INVALID_CUSTOMER
                elif audit_result.strategy_applicable and not audit_result.customer_strategy_adherent:
                    status = EpisodeStatus.INVALID_STRATEGY
                else:
                    status = EpisodeStatus.COMPLETE

            if simulation_payload is None:
                raise RuntimeError("native τ-bench did not provide a serialized simulation")
            protocol_observation = observe_communication_protocol(
                simulation_payload.get("messages") or (),
                enforcement_enabled=self.manifest.enforce_communication_protocol,
            )
            episode_id = str(simulation_payload.get("id") or attempt_id)
            db_trace_telemetry = _write_db_state_trace(
                manifest=self.manifest,
                task=task,
                trajectory=simulation_payload,
                trajectory_path=simulation_path,
                episode_directory=episode_directory,
                data_root=self.data_root,
            )
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
                customer_valid=None if audit_result is None else audit_result.customer_valid,
                strategy_applicable=None if audit_result is None else audit_result.strategy_applicable,
                customer_strategy_adherent=None if audit_result is None else audit_result.customer_strategy_adherent,
                policy_violation=False if audit_result is None else audit_result.policy_violation,
                invalid_repeated_write_calls=(
                    None if audit_result is None else audit_result.invalid_repeated_write_calls
                ),
                policy_rule_id=None if audit_result is None else audit_result.policy_rule_id,
                mistake_type=None if audit_result is None else audit_result.mistake_type,
                workflow_stage=None if audit_result is None else audit_result.workflow_stage,
                evidence=() if audit_result is None else audit_result.evidence,
                trajectory_ref=simulation_path.relative_to(self.output_directory).as_posix(),
                audit_ref=None if audit_result is None else audit_result.verifier_ref,
                tool_calls=_count_tool_calls(simulation_payload.get("messages") or ()),
                enforce_communication_protocol=self.manifest.enforce_communication_protocol,
                mixed_text_tool_call_messages=(
                    protocol_observation["mixed_text_tool_call_message_count"]
                ),
                raw_review={
                    "native_review": simulation_payload.get("review"),
                    "auth_classification": simulation_payload.get("auth_classification"),
                },
            )
            after = self.request_budget.snapshot()
            exact_episode_usage = episode_usage.snapshot(cap=self.request_budget.snapshot().cap)
            _write_json_once(telemetry_path, {
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
                "independent_audit": None if audit_result is None else _audit_dict(audit_result),
                **db_trace_telemetry,
            })
            _write_json_once(record_path, record.to_dict())
            with self._completed_episode_cache_lock:
                self._completed_episode_cache[episode_key_sha256] = (
                    record, exact_episode_usage, False,
                )
            return record
        except Exception as exc:
            after = self.request_budget.snapshot()
            exact_episode_usage = episode_usage.snapshot(cap=self.request_budget.snapshot().cap)
            _write_json_once(episode_directory / "incomplete-run.json", {
                "attempt_id": attempt_id,
                "task_id": str(task_id),
                "seed": seed,
                "episode_key": episode_key,
                "episode_key_sha256": episode_key_sha256,
                "panel_name": panel_name,
                "failure_type": type(exc).__name__,
                "native_simulation_saved": simulation_path.exists(),
                "communication_protocol_observation": (
                    None if simulation_payload is None else observe_communication_protocol(
                        simulation_payload.get("messages") or (),
                        enforcement_enabled=self.manifest.enforce_communication_protocol,
                    )
                ),
                "rendered_prompt_sha256": prompt_hashes,
                "budget_before": before.to_dict(),
                "budget_after": after.to_dict(),
                "budget_delta": _snapshot_delta(before, after),
                "episode_budget_delta": exact_episode_usage.to_dict(),
            })
            raise NativeEpisodeRunError(type(exc).__name__) from exc

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
                if isinstance(key, dict) and key_sha == sha256_json(key):
                    raise ValueError(
                        "an interrupted native episode has the same frozen task/seed/strategy/panel key; "
                        "refusing an unregistered duplicate provider run"
                    )
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
            self._completed_episode_cache[key_sha] = (record, usage, True)

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
        return payload


def run_native_phase3(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    phase0_result_path: str | Path,
    audit_provider: AuditProvider,
    customer_proposal_provider: OperatorSelector | None = None,
    provider_provenance: Mapping[str, Any] | None = None,
    service_transition: Callable[..., Any] | None = None,
    service_proposal_provider: Callable[..., Any] | None = None,
    service_repair_audit_provider: Callable[..., Any] | None = None,
    stop_before_next_episode_file: str | Path | None = None,
) -> tuple[tuple[Any, ...], BudgetSnapshot]:
    """Run the frozen two-generation controller on native τ-bench episodes.

    A config must explicitly enable the provider and freeze all four role
    models. `audit_provider` must return independent Customer validity,
    adherence, and policy-attribution judgments; the native τ-bench reviewer
    alone is intentionally insufficient. A verified failure requires either
    a complete `service_transition` callback or separate repair proposal and
    audit providers that are assembled into the built-in paired gate.
    """

    from .archive import FailureArchive
    from .lifecycle import TwoGenerationSmoke
    from .phase0_run import _parse_strategies, load_config

    config = load_config(config_path)
    manifest = MechanismManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError("native Phase 3 is disabled in the frozen manifest")
    if audit_provider is None:
        raise ValueError("native Phase 3 requires an independent EpisodeAudit provider")
    phase0_budget, run_context = _validate_phase0_parent(phase0_result_path, manifest)
    customer, service = _parse_strategies(config["experiment"])
    customer = customer or CustomerStrategy()
    role_models = dict(manifest.role_models)
    if customer_proposal_provider is None:
        customer_proposal_provider = LLMCustomerEvolver(
            model=role_models["evolver"],
            model_args=role_model_args_for_runtime(manifest.role_model_args)["evolver"],
        )
    agent_model = role_models["agent"]
    from litellm import token_counter

    service_token_counter = lambda text: token_counter(model=agent_model, text=text)
    if service_transition is None:
        if service_proposal_provider is None or service_repair_audit_provider is None:
            raise ValueError(
                "native Phase 3 requires a ServiceTransition or both repair proposal and audit providers"
            )
        from .service_transition import GatedServiceTransition

        service_transition = GatedServiceTransition(
            evolution_task_id=manifest.evolution_task_id,
            validation_task_id=manifest.validation_task_id,
            seed=manifest.seed,
            initial_service=service,
            proposal_provider=service_proposal_provider,
            audit_provider=service_repair_audit_provider,
            token_counter=service_token_counter,
        )
    elif service_proposal_provider is not None or service_repair_audit_provider is not None:
        raise ValueError("pass either a complete ServiceTransition or repair providers, not both")
    run_context = {
        **run_context,
        "provider_provenance": _provider_provenance_document(
            provider_provenance,
            {
                "audit_provider": audit_provider,
                "customer_proposal_provider": customer_proposal_provider,
                "service_transition": service_transition,
            },
        ),
    }
    budget = RequestBudget(manifest.request_budget_cap)
    output_directory = Path(manifest.output_path)
    context_path = output_directory / "run-context.json"
    checkpoint_path = Path(manifest.checkpoint_path)
    resuming = checkpoint_path.exists()
    if context_path.exists():
        saved_context = json.loads(context_path.read_text(encoding="utf-8"))
        if saved_context != run_context:
            raise ValueError("existing Phase 3 run context differs from frozen parent or provider provenance")
    elif resuming:
        raise FileNotFoundError("Phase 3 checkpoint exists without its immutable run-context.json")
    if not resuming:
        budget.absorb_usage(phase0_budget)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_dir,
        request_budget=budget,
        audit_provider=audit_provider,
        service_token_counter=service_token_counter,
        stop_before_next_episode_file=stop_before_next_episode_file,
    )
    archive = FailureArchive(Path(manifest.output_path) / "archive.sqlite")
    controller = TwoGenerationSmoke(
        manifest=manifest,
        checkpoint_path=manifest.checkpoint_path,
        runner=runner,
        task_ids=(manifest.evolution_task_id, manifest.validation_task_id),
        seed=manifest.seed,
        request_budget=budget,
        failure_archive=archive,
        manifest_context=run_context,
    )
    if not context_path.exists():
        _write_json_once(context_path, run_context)
    commits = controller.run(
        customer,
        service,
        service_transition=service_transition,
        customer_proposal_provider=customer_proposal_provider,
        candidates_per_generation=manifest.customer_candidates,
    )
    final_budget = budget.snapshot()
    result = _phase3_result_document(
        manifest=manifest,
        output_directory=output_directory,
        run_context=run_context,
        commits=commits,
        provider_budget=final_budget,
    )
    _write_or_verify_immutable_json(output_directory / "phase3-result.json", result)
    return commits, final_budget


def run_native_pilot(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    callback_factory: Callable[[Mapping[str, Any], PilotManifest, int], Mapping[str, Any]],
    stop_before_next_episode_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one frozen Pilot condition over its independent seed blocks.

    The callback factory constructs per-seed providers without making requests.
    Each seed gets an isolated checkpoint, archive, trajectory store, and shared
    request budget. Validation and sealed-H episodes run only after the final
    generation commits.
    """

    from .archive import FailureArchive
    from .customer_evolver import LLMCustomerEvolver
    from .lifecycle import TwoGenerationSmoke
    from .manifest import PilotManifest, write_manifest_once
    from .phase0_run import _parse_strategies, load_config

    config = load_config(config_path)
    manifest = PilotManifest.from_mapping(config)
    if not manifest.real_provider_enabled:
        raise RuntimeError("native Pilot execution is disabled in the frozen manifest")
    if manifest.condition not in {
        "adaptive_coevolution", "one_shot_repair", "random_mutation", "static_customer",
        "adaptive_customer", "frozen_service", "frozen_customer",
        "no_historical_replay",
    }:
        raise ValueError(
            "native multi-generation Pilot condition is not supported"
        )
    final_episode_count = len(manifest.validation_task_ids) + len(manifest.heldout_task_ids)
    if final_episode_count >= manifest.max_episodes:
        raise ValueError("Pilot episode cap must leave evolution slots before final V/H")
    data_root = Path(data_dir).expanduser().resolve()
    task_selection = config["experiment"]["task_selection"]
    all_task_ids = (
        manifest.evolution_task_ids + manifest.validation_task_ids + manifest.heldout_task_ids
    )
    _load_pinned_tasks(
        manifest, data_dir=data_root, task_selection=task_selection, task_ids=all_task_ids,
    )
    customer, service = _parse_strategies(config["experiment"])
    customer = customer or CustomerStrategy()
    role_models = dict(manifest.role_models)
    from litellm import token_counter

    service_token_counter = lambda text: token_counter(model=role_models["agent"], text=text)
    output_root = Path(manifest.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest_path = output_root / "pilot-manifest.json"
    if root_manifest_path.exists():
        if json.loads(root_manifest_path.read_text(encoding="utf-8")) != manifest.to_document():
            raise ValueError("existing Pilot output belongs to a different frozen manifest")
    else:
        write_manifest_once(root_manifest_path, manifest)

    seed_results = []
    for seed_index, seed in enumerate(manifest.evolution_seeds):
        # Reserve a disjoint 100k range for every independent seed block. The
        # current schedule uses relative offsets below 51k for final H.
        episode_seed_base = (seed_index + 1) * 100_000
        static_schedule = (
            _build_static_customer_schedule(
                manifest, episode_seed_base=episode_seed_base,
            )
            if manifest.condition == "static_customer" else None
        )
        seed_dir = output_root / "seed-blocks" / f"seed-{seed}"
        checkpoint_path = Path(manifest.checkpoint_path) / f"seed-{seed}.json"
        returned = callback_factory(config, manifest, episode_seed_base)
        if not isinstance(returned, Mapping):
            raise TypeError("Pilot callback factory must return a mapping")
        callbacks = {
            key: value for key, value in returned.items()
            if key != "provider_provenance" and value is not None
        }
        provider_provenance = returned.get("provider_provenance")
        allowed = {
            "audit_provider", "customer_proposal_provider", "service_transition",
            "service_proposal_provider", "service_repair_audit_provider",
        }
        if set(callbacks) - allowed or "audit_provider" not in callbacks:
            raise ValueError("Pilot callbacks must include audit_provider and use supported roles")
        if any(not callable(value) for value in callbacks.values()):
            raise TypeError("every Pilot provider callback must be callable")
        repair_keys = {"service_proposal_provider", "service_repair_audit_provider"}
        has_transition = "service_transition" in callbacks
        if manifest.condition in {"random_mutation", "static_customer"}:
            control = manifest.condition
            if set(callbacks) != {"audit_provider"}:
                raise ValueError(
                    f"{control} control accepts only audit_provider; it freezes Service and "
                    "cannot load a failure-aware Customer Evolver"
                )
        elif manifest.condition in {"frozen_service", "adaptive_customer"}:
            if has_transition or set(callbacks) & repair_keys:
                raise ValueError(
                    f"{manifest.condition} cannot load any Service repair callback"
                )
        elif manifest.condition == "frozen_customer":
            if "customer_proposal_provider" in callbacks:
                raise ValueError("frozen_customer cannot load a Customer Evolver callback")
            has_repair_callbacks = repair_keys <= set(callbacks)
            if bool(set(callbacks) & repair_keys) != has_repair_callbacks:
                raise ValueError("Pilot repair proposal and audit callbacks must be supplied together")
            if has_transition == has_repair_callbacks:
                raise ValueError("Pilot callbacks require a ServiceTransition or both repair callbacks")
        elif manifest.condition == "no_historical_replay":
            has_repair_callbacks = repair_keys <= set(callbacks)
            if bool(set(callbacks) & repair_keys) != has_repair_callbacks:
                raise ValueError("Pilot repair proposal and audit callbacks must be supplied together")
            if has_transition and has_repair_callbacks:
                raise ValueError("Pilot must use one Service transition callback route")
            if has_transition:
                from .service_transition import GatedServiceTransition

                transition = callbacks["service_transition"]
                if (not isinstance(transition, GatedServiceTransition)
                        or transition.include_historical_replay):
                    raise ValueError(
                        "no_historical_replay requires GatedServiceTransition with historical replay disabled"
                    )
            elif not has_repair_callbacks:
                raise ValueError("no_historical_replay requires a Service transition or both repair callbacks")
        else:
            has_repair_callbacks = repair_keys <= set(callbacks)
            if bool(set(callbacks) & repair_keys) != has_repair_callbacks:
                raise ValueError("Pilot repair proposal and audit callbacks must be supplied together")
            if has_transition == has_repair_callbacks:
                raise ValueError("Pilot callbacks require a ServiceTransition or both repair callbacks")
        if (manifest.condition not in {"random_mutation", "static_customer", "frozen_customer"}
                and "customer_proposal_provider" not in callbacks):
            callbacks["customer_proposal_provider"] = LLMCustomerEvolver(
                model=role_models["evolver"],
                model_args=role_model_args_for_runtime(manifest.role_model_args)["evolver"],
            )
        if (manifest.condition not in {
            "random_mutation", "static_customer", "frozen_service", "adaptive_customer",
        }
                and not has_transition):
            from .service_transition import GatedServiceTransition

            callbacks["service_transition"] = GatedServiceTransition(
                evolution_task_id=manifest.evolution_task_ids[0],
                validation_task_id=manifest.validation_task_ids[0],
                evolution_task_ids=manifest.evolution_task_ids,
                validation_task_ids=manifest.validation_task_ids,
                seed=episode_seed_base,
                initial_service=service,
                proposal_provider=callbacks.pop("service_proposal_provider"),
                audit_provider=callbacks.pop("service_repair_audit_provider"),
                token_counter=service_token_counter,
                include_historical_replay=manifest.condition != "no_historical_replay",
            )
        from .service_baselines import OneShotServiceTransition

        if (manifest.condition == "one_shot_repair"
                and not isinstance(callbacks["service_transition"], OneShotServiceTransition)):
            raise ValueError("one_shot_repair Pilot requires the frozen OneShotServiceTransition wrapper")
        if (manifest.condition == "adaptive_coevolution"
                and isinstance(callbacks["service_transition"], OneShotServiceTransition)):
            raise ValueError("adaptive_coevolution Pilot cannot use the one-shot baseline wrapper")
        run_context = {
            "schema_version": 1,
            "pilot_manifest_sha256": manifest.sha256,
            "condition": manifest.condition,
            "evolution_seed": seed,
            "episode_seed_base": episode_seed_base,
            "task_panels": {
                "E": list(manifest.evolution_task_ids),
                "V": list(manifest.validation_task_ids),
                "H": list(manifest.heldout_task_ids),
            },
            "static_customer_schedule_sha256": (
                None if static_schedule is None else static_schedule["sha256"]
            ),
            "provider_provenance": _provider_provenance_document(
                provider_provenance,
                {
                    key: callbacks[key]
                    for key in (
                        "audit_provider", "customer_proposal_provider", "service_transition",
                    ) if key in callbacks
                },
            ),
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        seed_dir.mkdir(parents=True, exist_ok=True)
        if static_schedule is not None:
            _write_or_verify_immutable_json(
                seed_dir / "static-customer-schedule.json", static_schedule,
            )
        seed_manifest_path = seed_dir / "manifest.json"
        if seed_manifest_path.exists():
            if json.loads(seed_manifest_path.read_text(encoding="utf-8")) != manifest.to_document():
                raise ValueError("Pilot seed directory belongs to a different frozen manifest")
        else:
            write_manifest_once(seed_manifest_path, manifest)
        context_path = seed_dir / "run-context.json"
        if context_path.exists():
            if json.loads(context_path.read_text(encoding="utf-8")) != run_context:
                raise ValueError("existing Pilot run context differs from frozen providers or design")
        elif checkpoint_path.exists():
            raise FileNotFoundError("Pilot seed checkpoint exists without its immutable run context")
        else:
            _write_json_once(context_path, run_context)
        result_path = seed_dir / "pilot-seed-result.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            _verify_pilot_seed_result(
                prior, result_path=result_path,
                expected_manifest_sha256=manifest.sha256,
                expected_context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
                expected_evolution_seed=seed,
                expected_episode_seed_base=episode_seed_base,
                expected_static_schedule_sha256=(
                    None if static_schedule is None else static_schedule["sha256"]
                ),
            )
            seed_results.append(prior)
            continue

        budget = RequestBudget(manifest.request_budget_cap)
        runner = TauBenchEpisodeRunner(
            manifest=manifest, config=config, data_dir=data_root,
            request_budget=budget, audit_provider=callbacks["audit_provider"],
            service_token_counter=service_token_counter, output_directory=seed_dir,
            stop_before_next_episode_file=stop_before_next_episode_file,
        )
        controller = None
        commits = ()
        static_evolution_count = 0
        static_episode_attempt_count = 0
        static_discovery_episodes: list[EpisodeRecord] = []
        static_reproduction_episodes: list[EpisodeRecord] = []
        static_verified_failures: dict[str, FailureRecord] = {}

        def dispatch_static_episode(_runner=runner, **kwargs) -> EpisodeRecord:
            nonlocal static_episode_attempt_count
            if static_episode_attempt_count >= manifest.max_episodes:
                raise RuntimeError("static Pilot episode cap reached before scheduled dispatch")
            static_episode_attempt_count += 1
            return _runner(**kwargs)

        if static_schedule is not None:
            for index, assignment in enumerate(static_schedule["evolution"]):
                source = dispatch_static_episode(
                    task_id=assignment["task_id"], seed=assignment["seed"],
                    customer=CustomerStrategy(**assignment["customer_strategy"]),
                    service=service, panel_name=assignment["panel_name"],
                )
                static_discovery_episodes.append(source)
                if source.has_attributable_failure_candidate and source.audit_ref:
                    reproduction = dispatch_static_episode(
                        task_id=assignment["task_id"],
                        seed=episode_seed_base + 60_000 + index,
                        customer=CustomerStrategy(**assignment["customer_strategy"]),
                        service=service,
                        panel_name=f"pilot-static-reproduction-g{assignment['generation']}-{index + 1}",
                    )
                    static_reproduction_episodes.append(reproduction)
                    if (reproduction.has_attributable_failure_candidate and reproduction.audit_ref):
                        try:
                            failure = FailureRecord.verify(
                                source,
                                reproduction_episode=reproduction,
                                generation=assignment["generation"],
                                verifier=source.audit_ref,
                                reproduction_verifier=reproduction.audit_ref,
                            )
                        except ValueError:
                            pass
                        else:
                            static_verified_failures.setdefault(failure.failure_id, failure)
            static_evolution_count = len(static_schedule["evolution"])
            final_customer = None
            final_service = service
        else:
            controller = TwoGenerationSmoke(
                manifest=manifest, checkpoint_path=str(checkpoint_path), runner=runner,
                task_ids=(manifest.evolution_task_ids[0], manifest.validation_task_ids[0]),
                evolution_task_ids=manifest.evolution_task_ids, seed=seed,
                episode_seed_base=episode_seed_base,
                request_budget=budget, failure_archive=FailureArchive(seed_dir / "archive.sqlite"),
                manifest_context=run_context,
                max_episodes=manifest.max_episodes - final_episode_count,
            )
            # Pilot has no upstream Phase 0 integration episode. The controller's
            # manifest_context default reserves one slot for Phase 3's Phase 0
            # parent, so start Pilot's per-seed episode accounting at zero.
            controller.episode_attempts = 0
            commits = controller.run(
                customer, service,
                service_transition=callbacks.get("service_transition"),
                customer_proposal_provider=callbacks.get("customer_proposal_provider"),
                customer_proposal_mode=(
                    "random_mutation" if manifest.condition == "random_mutation"
                    else "failure_conditioned"
                ),
                allow_frozen_service=manifest.condition in {
                    "random_mutation", "frozen_service", "adaptive_customer",
                },
                freeze_customer=manifest.condition == "frozen_customer",
                allow_strategy_revisit=True,
                candidates_per_generation=manifest.customer_candidates,
            )
            checkpoint_hash = manifest_fingerprint({
                "manifest": manifest.to_payload(), "run_context": run_context,
            })
            checkpoint = load_checkpoint(checkpoint_path, expected_manifest_hash=checkpoint_hash)
            final_customer = CustomerStrategy(**checkpoint.state["customer"])
            from .lifecycle import _service_from_dict

            final_service = _service_from_dict(checkpoint.state["service"])
        final_panels_path = seed_dir / "pilot-final-panels.json"
        if final_panels_path.exists():
            final_panels = json.loads(final_panels_path.read_text(encoding="utf-8"))
            if static_schedule is not None:
                panels_match = (
                    final_panels.get("static_customer_schedule_sha256") == static_schedule["sha256"]
                    and final_panels.get("service_strategy_id") == service_strategy_id(final_service)
                )
            else:
                panels_match = (
                    final_panels.get("customer_strategy_id") == customer_strategy_id(final_customer)
                    and final_panels.get("service_strategy_id") == service_strategy_id(final_service)
                )
            if final_panels.get("evolution_seed") != seed or not panels_match:
                raise ValueError("saved Pilot final panels differ from committed generation state")
            budget.absorb_usage(BudgetSnapshot(**final_panels["provider_budget_delta"]))
        else:
            before_final = budget.snapshot()
            validation_records = []
            heldout_records = []
            if static_schedule is not None:
                for assignment in static_schedule["validation"]:
                    validation_records.append(dispatch_static_episode(
                        task_id=assignment["task_id"], seed=assignment["seed"],
                        customer=CustomerStrategy(**assignment["customer_strategy"]),
                        service=final_service, panel_name=assignment["panel_name"],
                    ).to_dict())
                # Heldout tasks stay sealed until the frozen static schedule is complete.
                for assignment in static_schedule["heldout"]:
                    heldout_records.append(dispatch_static_episode(
                        task_id=assignment["task_id"], seed=assignment["seed"],
                        customer=CustomerStrategy(**assignment["customer_strategy"]),
                        service=final_service, panel_name=assignment["panel_name"],
                    ).to_dict())
            else:
                for index, task_id in enumerate(manifest.validation_task_ids):
                    validation_records.append(runner(
                        task_id=task_id, seed=episode_seed_base + 40_000 + index,
                        customer=final_customer, service=final_service,
                        panel_name=f"pilot-final-validation-{index + 1}",
                    ).to_dict())
                # Heldout tasks stay sealed until all generations and validation measurements finish.
                for index, task_id in enumerate(manifest.heldout_task_ids):
                    heldout_records.append(runner(
                        task_id=task_id, seed=episode_seed_base + 50_000 + index,
                        customer=final_customer, service=final_service,
                        panel_name=f"pilot-final-heldout-{index + 1}",
                    ).to_dict())
            delta = _budget_snapshot_difference(budget.snapshot(), before_final)
            final_panels = {
                "schema_version": 1,
                "evolution_seed": seed,
                "customer_strategy_id": (
                    None if final_customer is None else customer_strategy_id(final_customer)
                ),
                "service_strategy_id": service_strategy_id(final_service),
                "static_customer_schedule_sha256": (
                    None if static_schedule is None else static_schedule["sha256"]
                ),
                "validation_records": validation_records,
                "heldout_records": heldout_records,
                "provider_budget_delta": delta.to_dict(),
            }
            _write_json_once(final_panels_path, final_panels)

        episode_records = [
            EpisodeRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))).to_dict()
            for path in sorted((seed_dir / "episodes").glob("*/episode-record.json"))
        ]
        expected_episode_count = (
            (
                static_evolution_count + len(static_reproduction_episodes)
                if controller is None else controller.episode_attempts
            )
            + len(manifest.validation_task_ids)
            + len(manifest.heldout_task_ids)
        )
        if len(episode_records) != expected_episode_count:
            raise ValueError("Pilot episode artifacts do not reconcile with run and final-panel counts")
        rq1_study_run = None
        if manifest.condition in {"adaptive_customer", "random_mutation", "static_customer"}:
            rq1_study_run = _build_rq1_pilot_study_run(
                manifest=manifest,
                evolution_seed=seed,
                provider_attempts=budget.snapshot().attempts,
                commits=commits,
                static_discovery_episodes=tuple(static_discovery_episodes),
                static_reproduction_episodes=tuple(static_reproduction_episodes),
                static_verified_failures=tuple(static_verified_failures.values()),
            )
        seed_result = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": manifest.experiment_id,
            "condition": manifest.condition,
            "manifest_sha256": manifest.sha256,
            "run_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            "evolution_seed": seed,
            "episode_seed_base": episode_seed_base,
            "static_customer_schedule_sha256": (
                None if static_schedule is None else static_schedule["sha256"]
            ),
            "generation_commits": [
                {
                    "generation": item.generation, "customer_id": item.customer_id,
                    "service_id": item.service_id, "customer_evolved": item.customer_evolved,
                    "service_evolved": item.service_evolved, "completed": item.completed,
                    "note": item.note, "decision_record": item.decision_record,
                }
                for item in commits
            ],
            "provider_budget": budget.snapshot().to_dict(),
            "episode_count": len(episode_records),
            "episodes": episode_records,
            "rq1_study_run": rq1_study_run,
            "final_panels": {
                "validation": final_panels["validation_records"],
                "heldout": final_panels["heldout_records"],
            },
            "artifacts": _pilot_artifact_rows(
                seed_dir,
                None if controller is None else checkpoint_path,
                result_path,
            ),
        }
        _write_or_verify_immutable_json(result_path, seed_result)
        seed_results.append(seed_result)

    result_document = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "condition": manifest.condition,
        "manifest_sha256": manifest.sha256,
        "evolution_seeds": list(manifest.evolution_seeds),
        "generation_count": manifest.generations,
        "seed_blocks": [
            {
                "evolution_seed": item["evolution_seed"],
                "episode_seed_base": item["episode_seed_base"],
                "static_customer_schedule_sha256": item.get("static_customer_schedule_sha256"),
                "result_path": str(
                    output_root / "seed-blocks" / f"seed-{item['evolution_seed']}"
                    / "pilot-seed-result.json"
                ),
                "result_sha256": hashlib.sha256((
                    output_root / "seed-blocks" / f"seed-{item['evolution_seed']}"
                    / "pilot-seed-result.json"
                ).read_bytes()).hexdigest(),
                "episode_count": item["episode_count"],
                "provider_budget": item["provider_budget"],
            }
            for item in seed_results
        ],
        "rq1_study_runs": [
            item["rq1_study_run"] for item in seed_results
            if item.get("rq1_study_run") is not None
        ],
    }
    _write_or_verify_immutable_json(output_root / "pilot-result.json", result_document)
    return result_document


def _validate_phase0_parent(
    phase0_result_path: str | Path,
    manifest: MechanismManifest,
) -> tuple[BudgetSnapshot, dict[str, Any]]:
    """Bind Phase 3 to one complete, compatible Phase 0 native integration run."""

    result_path = Path(phase0_result_path).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Phase 0 result does not exist: {result_path}")
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    phase0_manifest_path = result_path.parent / "manifest.json"
    if not phase0_manifest_path.is_file():
        raise FileNotFoundError("Phase 0 result is missing its adjacent immutable manifest.json")
    phase0_document = json.loads(phase0_manifest_path.read_text(encoding="utf-8"))
    recorded_manifest_hash = phase0_document.get("manifest_sha256")
    phase0_payload = {key: value for key, value in phase0_document.items() if key != "manifest_sha256"}
    if not isinstance(recorded_manifest_hash, str) or sha256_json(phase0_payload) != recorded_manifest_hash:
        raise ValueError("Phase 0 manifest fingerprint is invalid")
    if result.get("schema_version") != 1 or result.get("status") != "complete":
        raise ValueError("Phase 3 requires a completed schema-version-1 Phase 0 result")
    if result.get("manifest_sha256") != recorded_manifest_hash:
        raise ValueError("Phase 0 result and adjacent manifest fingerprints differ")
    if result.get("experiment_id") != phase0_document.get("experiment_id"):
        raise ValueError("Phase 0 result experiment ID does not match its manifest")
    if phase0_document.get("phase") != "0-integration-proof":
        raise ValueError("Phase 3 parent artifact is not a Phase 0 integration proof")
    upstream = phase0_document.get("upstream") or {}
    phase0_selection = phase0_document.get("task_selection") or {}
    if upstream.get("commit") != manifest.upstream_commit:
        raise ValueError("Phase 0 and Phase 3 upstream τ-bench commits differ")
    if phase0_document.get("domain") != manifest.domain or phase0_document.get("communication_mode") != manifest.communication_mode:
        raise ValueError("Phase 0 and Phase 3 domain/runtime protocols differ")
    phase0_protocol_mode = phase0_document.get("enforce_communication_protocol")
    if type(phase0_protocol_mode) is not bool:
        raise ValueError(
            "Phase 0 parent does not record communication-protocol enforcement; regenerate it"
        )
    if phase0_protocol_mode != manifest.enforce_communication_protocol:
        raise ValueError("Phase 0 and Phase 3 communication-protocol enforcement differs")
    if phase0_document.get("evaluation_type") != manifest.evaluation_type:
        raise ValueError("Phase 0 and Phase 3 evaluation modes differ")
    if (phase0_document.get("real_provider_enabled") is not True
            or phase0_document.get("max_episodes") != 1
            or not 1 <= int(phase0_document.get("request_budget_cap", 0)) <= 70):
        raise ValueError("Phase 0 parent must enable one episode within the 70-attempt cap")
    if phase0_selection.get("evolution") != [manifest.evolution_task_id]:
        raise ValueError("Phase 0 E task does not match the Phase 3 evolution task")
    if phase0_document.get("seed") != manifest.seed:
        raise ValueError("Phase 0 and Phase 3 seeds differ")
    if phase0_document.get("source_blob_sha1") != dict(manifest.source_blob_sha1):
        raise ValueError("Phase 0 and Phase 3 pinned source fingerprints differ")
    phase3_models = dict(manifest.role_models)
    phase0_models = {role: model for role, model in phase3_models.items() if role != "evolver"}
    if phase0_document.get("role_models") != phase0_models:
        raise ValueError("Phase 0 and Phase 3 frozen role models differ")
    phase3_model_args = {role: dict(args) for role, args in manifest.role_model_args}
    phase0_model_args = {role: args for role, args in phase3_model_args.items() if role != "evolver"}
    if phase0_document.get("role_model_args") != phase0_model_args:
        raise ValueError("Phase 0 and Phase 3 frozen role model arguments differ")
    if result.get("task_id") is None or str(result["task_id"]) != manifest.evolution_task_id:
        raise ValueError("Phase 0 result task ID does not match the Phase 3 evolution task")
    simulation_name = result.get("simulation_file")
    if not isinstance(simulation_name, str) or not simulation_name:
        raise ValueError("Phase 0 result has no native simulation file reference")
    simulation_path = (result_path.parent / simulation_name).resolve()
    if not simulation_path.is_relative_to(result_path.parent):
        raise ValueError("Phase 0 simulation reference escapes its result directory")
    if not simulation_path.is_file():
        raise FileNotFoundError("Phase 0 result references a missing native simulation")
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    simulation_id = result.get("simulation_id")
    if (not simulation_id or simulation.get("id") != simulation_id
            or str(simulation.get("task_id")) != manifest.evolution_task_id):
        raise ValueError("Phase 0 native simulation ID or task does not match its result")
    budget_data = result.get("provider_budget")
    if not isinstance(budget_data, Mapping):
        raise TypeError("Phase 0 result has no provider-budget accounting")
    try:
        phase0_budget = BudgetSnapshot(**budget_data)
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 0 provider-budget snapshot is malformed") from exc
    if phase0_budget.cap != int(phase0_document["request_budget_cap"]) or phase0_budget.attempts < 1:
        raise ValueError("Phase 0 provider-budget snapshot is inconsistent with its manifest")
    context = {
        "phase0_result_path": str(result_path),
        "phase0_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "phase0_manifest_sha256": recorded_manifest_hash,
        "phase0_simulation_id": simulation_id,
        "phase0_provider_budget": phase0_budget.to_dict(),
    }
    return phase0_budget, context


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


def _write_db_state_trace(
    *, manifest: Any, task: Any, trajectory: Mapping[str, Any],
    trajectory_path: Path, episode_directory: Path, data_root: Path,
) -> dict[str, Any]:
    """Generate an isolated display sidecar; any observer failure is non-fatal."""

    started = perf_counter()
    trajectory_bytes = b""
    try:
        trajectory_bytes = trajectory_path.read_bytes()
        trace = replay_db_state_trace(
            manifest=manifest.to_document(),
            task=task,
            trajectory=trajectory,
            trajectory_bytes=trajectory_bytes,
            data_dir=data_root,
        )
    except DBStateTraceError as exc:
        trace = unavailable_trace(
            task_id=str(trajectory.get("task_id", getattr(task, "id", "unknown"))),
            episode_id=str(trajectory.get("id", episode_directory.name)),
            manifest=manifest.to_document(),
            trajectory_bytes=trajectory_bytes or None,
            reason_code=exc.reason_code,
        )
    except Exception:  # noqa: BLE001 - observer failures cannot invalidate episodes
        trace = unavailable_trace(
            task_id=str(trajectory.get("task_id", getattr(task, "id", "unknown"))),
            episode_id=str(trajectory.get("id", episode_directory.name)),
            manifest=manifest.to_document(),
            trajectory_bytes=trajectory_bytes or None,
            reason_code="trace_not_generated",
        )
    artifact_saved = False
    try:
        write_trace_once(episode_directory / "db-state-trace.json", trace)
        artifact_saved = True
    except Exception:  # noqa: BLE001 - sidecar write errors never invalidate episodes
        # A sidecar storage failure must not convert a completed episode into a
        # Core failure. Console readers treat a missing sidecar as unavailable.
        artifact_saved = False
    summary = trace.get("summary", {}) if isinstance(trace, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {
        "db_trace_status": trace.get("status", "unavailable") if artifact_saved else "unavailable",
        "db_trace_generation_seconds": round(perf_counter() - started, 6),
        "db_trace_event_count": int(summary.get("mutation_event_count", 0)) if artifact_saved else 0,
        "db_trace_field_change_count": int(summary.get("field_change_count", 0)) if artifact_saved else 0,
    }


def _provider_provenance_document(
    plugin_provenance: Mapping[str, Any] | None,
    callbacks: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind provider callback source files and any plugin package to run resume."""

    callback_sources: dict[tuple[str, str], dict[str, str]] = {}
    callback_configurations: dict[tuple[str, str], dict[str, Any]] = {}
    visited: set[int] = set()

    def visit(label: str, value: Any) -> None:
        if value is None:
            return
        already_expanded = id(value) in visited
        visited.add(id(value))
        provenance_hook = getattr(value, "__evotau_provenance__", None)
        if callable(provenance_hook):
            try:
                supplied_config = provenance_hook()
                if not isinstance(supplied_config, Mapping):
                    raise TypeError("provenance hook must return a mapping")
                normalized_config = json.loads(json.dumps(
                    dict(supplied_config), sort_keys=True, ensure_ascii=False, allow_nan=False,
                ))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"provider callback {label} returned invalid resume provenance"
                ) from exc
            identity = (
                str(getattr(value, "__module__", type(value).__module__)),
                str(getattr(value, "__qualname__", type(value).__qualname__)),
            )
            callback_configurations[(label, ":".join(identity))] = {
                "label": label,
                "module": identity[0],
                "qualname": identity[1],
                "configuration": normalized_config,
                "configuration_sha256": sha256_json(normalized_config),
            }
        if callable(value):
            target = value if inspect.isfunction(value) or inspect.ismethod(value) else type(value)
            try:
                source_path = inspect.getsourcefile(target)
            except (TypeError, OSError) as exc:
                raise ValueError(f"provider callback {label} has no inspectable source file") from exc
            if source_path is None:
                raise ValueError(f"provider callback {label} has no source file")
            original_path = Path(source_path).expanduser()
            if original_path.is_symlink() or not original_path.is_file():
                raise ValueError(f"provider callback {label} source is not a regular file")
            path = original_path.resolve()
            module_name = getattr(target, "__module__", type(value).__module__)
            qualname = getattr(target, "__qualname__", type(value).__qualname__)
            row = {
                "module": str(module_name),
                "qualname": str(qualname),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            callback_sources[(label, f"{row['module']}:{row['qualname']}")] = row
            if already_expanded:
                return
            if is_dataclass(value) and not isinstance(value, type):
                for item in fields(value):
                    nested = getattr(value, item.name)
                    if callable(nested) or (is_dataclass(nested) and not isinstance(nested, type)):
                        visit(f"{label}.{item.name}", nested)
        elif is_dataclass(value) and not isinstance(value, type) and not already_expanded:
            for item in fields(value):
                nested = getattr(value, item.name)
                if callable(nested) or (is_dataclass(nested) and not isinstance(nested, type)):
                    visit(f"{label}.{item.name}", nested)

    for name, callback in callbacks.items():
        visit(name, callback)
    if not callback_sources:
        raise ValueError("native Phase 3 provider provenance has no callback source files")

    plugin: dict[str, Any] | None = None
    if plugin_provenance is not None:
        plugin = dict(plugin_provenance)
        recorded = plugin.pop("sha256", None)
        if not isinstance(recorded, str) or sha256_json(plugin) != recorded:
            raise ValueError("provider plugin source fingerprint is invalid")
        plugin["sha256"] = recorded

    payload = {
        "schema_version": 1,
        "plugin": plugin,
        "callback_sources": [
            {"label": label, **source}
            for (label, _identity), source in sorted(callback_sources.items())
        ],
        "callback_configurations": [
            configuration
            for (_label, _identity), configuration in sorted(callback_configurations.items())
        ],
    }
    return {**payload, "sha256": sha256_json(payload)}


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


def _phase3_result_document(
    *,
    manifest: MechanismManifest,
    output_directory: Path,
    run_context: Mapping[str, Any],
    commits: tuple[Any, ...],
    provider_budget: BudgetSnapshot,
) -> dict[str, Any]:
    """Index immutable Phase 3 outputs and bind every artifact by SHA-256."""

    output_root = output_directory.resolve()
    manifest_path = output_root / "manifest.json"
    context_path = output_root / "run-context.json"
    checkpoint_path = Path(manifest.checkpoint_path).expanduser().resolve()
    for label, path in (
        ("manifest", manifest_path), ("run context", context_path),
        ("generation checkpoint", checkpoint_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"completed Phase 3 is missing its {label} artifact")

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_context = json.loads(context_path.read_text(encoding="utf-8"))
    if saved_manifest != manifest.to_document() or saved_context != dict(run_context):
        raise ValueError("Phase 3 provenance artifacts changed before result indexing")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_state = checkpoint.get("state")
    if not isinstance(checkpoint_state, dict):
        raise TypeError("Phase 3 generation checkpoint has no state object")
    expected_checkpoint_hash = manifest_fingerprint({
        "manifest": manifest.to_payload(), "run_context": dict(run_context),
    })
    if checkpoint.get("manifest_hash") != expected_checkpoint_hash:
        raise ValueError("Phase 3 generation checkpoint is not bound to its manifest and parent run")
    if checkpoint_state.get("request_budget") != provider_budget.to_dict():
        raise ValueError("Phase 3 final provider budget differs from the checkpoint")
    if checkpoint_state.get("commits") != [
        {
            "generation": item.generation,
            "customer_id": item.customer_id,
            "service_id": item.service_id,
            "customer_evolved": item.customer_evolved,
            "service_evolved": item.service_evolved,
            "completed": item.completed,
            "note": item.note,
            "decision_record": item.decision_record,
        }
        for item in commits
    ]:
        raise ValueError("Phase 3 result commits differ from the generation checkpoint")

    episode_rows: list[dict[str, Any]] = []
    episode_root = output_root / "episodes"
    if episode_root.exists():
        if episode_root.is_symlink() or not episode_root.is_dir():
            raise ValueError("Phase 3 episode artifact root is not a regular directory")
        for directory in sorted(episode_root.iterdir(), key=lambda item: item.name):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Phase 3 episode artifact directory contains an unexpected entry")
            names = (
                "episode-record.json", "run-telemetry.json",
                "native-simulation.json", "incomplete-run.json",
            )
            artifacts: dict[str, dict[str, str]] = {}
            payloads: dict[str, Any] = {}
            for name in names:
                path = directory / name
                if not path.exists():
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Phase 3 episode artifact is not a regular file")
                raw = path.read_bytes()
                artifacts[name.removesuffix(".json").replace("-", "_")] = {
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                payloads[name] = json.loads(raw.decode("utf-8"))

            record = payloads.get("episode-record.json")
            telemetry = payloads.get("run-telemetry.json")
            simulation = payloads.get("native-simulation.json")
            incomplete = payloads.get("incomplete-run.json")
            if record is not None:
                if telemetry is None or simulation is None or incomplete is not None:
                    raise ValueError("completed episode is missing telemetry/simulation or also marked incomplete")
                if (
                    record.get("episode_id") != simulation.get("id")
                    or record.get("task_id") != str(simulation.get("task_id"))
                    or record.get("seed") != simulation.get("seed")
                    or telemetry.get("simulation_id") != record.get("episode_id")
                ):
                    raise ValueError("Phase 3 episode record, telemetry and simulation do not agree")
                status = "complete"
                episode_id = record["episode_id"]
                task_id = record["task_id"]
                seed = record["seed"]
                panel_name = telemetry.get("panel_name")
            elif incomplete is not None:
                status = "incomplete"
                episode_id = None
                task_id = incomplete.get("task_id")
                seed = incomplete.get("seed")
                panel_name = incomplete.get("panel_name")
            else:
                raise ValueError("Phase 3 episode directory has no complete or interruption record")
            episode_rows.append({
                "attempt_id": directory.name,
                "episode_id": episode_id,
                "task_id": task_id,
                "seed": seed,
                "panel_name": panel_name,
                "status": status,
                "artifacts": artifacts,
            })

    completed = sum(row["status"] == "complete" for row in episode_rows)
    if completed < len(commits):
        raise ValueError("Phase 3 result has fewer completed native episodes than committed generations")
    expected_attempts = checkpoint_state.get("episode_attempts")
    if type(expected_attempts) is not int or expected_attempts < 1:
        raise ValueError("Phase 3 checkpoint has an invalid episode-attempt count")
    phase3_attempts = expected_attempts - 1  # The parent Phase 0 episode shares this ceiling.
    if phase3_attempts != len(episode_rows):
        raise ValueError("Phase 3 episode artifacts do not reconcile with the checkpoint attempt count")
    return {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "run_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "phase0_result_sha256": run_context["phase0_result_sha256"],
        "generation_checkpoint": {
            "path": manifest.checkpoint_path,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "generation_commits": [
            {
                "generation": item.generation,
                "customer_id": item.customer_id,
                "service_id": item.service_id,
                "customer_evolved": item.customer_evolved,
                "service_evolved": item.service_evolved,
                "completed": item.completed,
                "note": item.note,
                "decision_record": item.decision_record,
            }
            for item in commits
        ],
        "provider_budget": provider_budget.to_dict(),
        "episode_attempt_count": len(episode_rows),
        "phase3_episode_attempt_count": phase3_attempts,
        "completed_episode_count": completed,
        "incomplete_episode_count": len(episode_rows) - completed,
        "episodes": episode_rows,
    }


def _write_or_verify_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create an immutable summary, accepting only an identical resume result."""

    try:
        _write_json_once(path, value)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("existing Phase 3 result artifact is unreadable") from exc
        if existing != dict(value):
            raise ValueError("existing Phase 3 result artifact differs from the resumed run")


def _pilot_artifact_rows(
    seed_directory: Path, checkpoint_path: Path | None, result_path: Path,
) -> list[dict[str, str]]:
    archive_path = seed_directory / "archive.sqlite"
    if archive_path.is_file():
        # Freeze the durable SQLite image before hashing it. WAL pages are part
        # of the archive state but sidecars are transient and not indexed.
        with sqlite3.connect(archive_path, timeout=5.0) as db:
            checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or checkpoint[0] != 0:
            raise RuntimeError("Pilot archive could not be checkpointed before artifact indexing")
    project_root = Path.cwd().resolve()
    paths = [
        path for path in seed_directory.rglob("*")
        if path.is_file()
        and path.name not in {"archive.sqlite-wal", "archive.sqlite-shm"}
        and path.resolve() != result_path.resolve()
    ]
    if checkpoint_path is not None and checkpoint_path.is_file():
        paths.append(checkpoint_path)
    rows = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        # SQLite removes transient WAL/SHM sidecars as connections close; they
        # are not part of the durable archive artifact and may disappear
        # between rglob() and this validation.
        if path.name in {"archive.sqlite-wal", "archive.sqlite-shm"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Pilot result cannot index a symlink or non-file artifact: {path} "
                f"(symlink={path.is_symlink()}, file={path.is_file()})"
            )
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError("Pilot artifact path escapes the project directory") from exc
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def _build_static_customer_schedule(
    manifest: PilotManifest,
    *,
    episode_seed_base: int,
) -> dict[str, Any]:
    """Materialize a balanced static Customer schedule for the exact Pilot panels."""

    from .baselines import StaticCustomerPortfolio

    if manifest.static_customer_portfolio_json is None:
        raise ValueError("static_customer Pilot is missing its frozen portfolio")
    portfolio = StaticCustomerPortfolio.from_dict(
        json.loads(manifest.static_customer_portfolio_json),
    )
    generation_seeds = tuple(episode_seed_base + generation for generation in range(manifest.generations))
    panel_schedule = portfolio.panel_schedule(
        task_ids=manifest.evolution_task_ids,
        seeds=generation_seeds,
        repeats_per_pair=1 + manifest.customer_candidates,
    )
    strategies = {customer_strategy_id(item): item for item in portfolio.strategies}
    evolution = [
        {
            "panel": "evolution",
            "panel_name": "discovery",
            "generation": generation,
            "task_id": task_id,
            "seed": seed,
            "repeat": repeat,
            "strategy_id": strategy_id,
            "customer_strategy": strategies[strategy_id].to_dict(),
        }
        for task_id, seed, repeat, strategy_id in panel_schedule.assignments
        for generation in (seed - episode_seed_base,)
    ]
    cursor = len(evolution)
    validation = []
    heldout = []
    for panel_name, panel, task_ids, offset in (
        ("validation", validation, manifest.validation_task_ids, 40_000),
        ("heldout", heldout, manifest.heldout_task_ids, 50_000),
    ):
        for index, task_id in enumerate(task_ids):
            strategy_id = portfolio.strategy_ids[cursor % len(portfolio.strategy_ids)]
            panel.append({
                "panel": panel_name,
                "panel_name": f"pilot-final-{panel_name}-{index + 1}",
                "generation": None,
                "task_id": task_id,
                "seed": episode_seed_base + offset + index,
                "repeat": 0,
                "strategy_id": strategy_id,
                "customer_strategy": strategies[strategy_id].to_dict(),
            })
            cursor += 1
    payload = {
        "schema_version": 1,
        "portfolio_id": portfolio.portfolio_id,
        "portfolio_sha256": portfolio.sha256,
        "panel_schedule_sha256": panel_schedule.sha256,
        "episode_seed_base": episode_seed_base,
        "evolution_task_ids": list(manifest.evolution_task_ids),
        "validation_task_ids": list(manifest.validation_task_ids),
        "heldout_task_ids": list(manifest.heldout_task_ids),
        "evolution": evolution,
        "validation": validation,
        "heldout": heldout,
    }
    return {**payload, "sha256": sha256_json(payload)}


def _build_rq1_pilot_study_run(
    *,
    manifest: PilotManifest,
    evolution_seed: int,
    provider_attempts: int,
    commits: tuple[Any, ...],
    static_discovery_episodes: tuple[EpisodeRecord, ...],
    static_reproduction_episodes: tuple[EpisodeRecord, ...],
    static_verified_failures: tuple[FailureRecord, ...],
) -> dict[str, Any]:
    """Export only the frozen E-panel evidence in the RQ1 analyzer's schema."""

    from .study_analysis import StudyRun

    condition = {
        "adaptive_customer": "adaptive_customer",
        "random_mutation": "random_mutation",
        "static_customer": "static_customer",
    }[manifest.condition]
    discovery: dict[str, EpisodeRecord] = {}
    reproductions: dict[str, EpisodeRecord] = {}
    failures: dict[str, FailureRecord] = {}

    def retain(target: dict[str, EpisodeRecord], episode: EpisodeRecord) -> None:
        previous = target.setdefault(episode.episode_id, episode)
        if previous != episode:
            raise ValueError("RQ1 Pilot export found conflicting records for one episode ID")

    if manifest.condition == "static_customer":
        for episode in static_discovery_episodes:
            retain(discovery, episode)
        for episode in static_reproduction_episodes:
            retain(reproductions, episode)
        failures.update((item.failure_id, item) for item in static_verified_failures)
    else:
        for commit in commits:
            decision = commit.decision_record
            customer_decision = decision.get("customer", {})
            for evaluation in customer_decision.get("evaluations", ()):
                panel_name = evaluation.get("panel_name")
                if panel_name == "discovery":
                    for row in evaluation.get("episodes", ()):
                        retain(discovery, EpisodeRecord.from_dict(row))
                    for row in evaluation.get("replication_episodes", ()):
                        retain(reproductions, EpisodeRecord.from_dict(row))
                elif panel_name == "confirmation":
                    for row in evaluation.get("episodes", ()):
                        retain(reproductions, EpisodeRecord.from_dict(row))
            for row in decision.get("verified_failures", ()):
                failure = FailureRecord.from_dict(row)
                failures.setdefault(failure.failure_id, failure)

    run = StudyRun(
        run_id=f"{manifest.experiment_id}:{manifest.condition}:seed-{evolution_seed}",
        seed_block_id=f"seed-{evolution_seed}",
        evolution_seed=evolution_seed,
        condition=condition,
        task_ids=manifest.evolution_task_ids,
        request_budget_cap=manifest.request_budget_cap,
        provider_attempts=provider_attempts,
        episodes=tuple(discovery.values()),
        verified_failures=tuple(failures.values()),
        reproduction_episodes=tuple(reproductions.values()),
    )
    # Match the persisted JSON representation so the first return and an
    # immutable resume compare byte-for-byte at the document level.
    return json.loads(json.dumps(run.to_dict(), ensure_ascii=False))


def _verify_pilot_seed_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    expected_manifest_sha256: str,
    expected_context_sha256: str,
    expected_evolution_seed: int,
    expected_episode_seed_base: int,
    expected_static_schedule_sha256: str | None,
) -> None:
    if (result.get("status") != "complete"
            or result.get("manifest_sha256") != expected_manifest_sha256
            or result.get("run_context_sha256") != expected_context_sha256
            or result.get("evolution_seed") != expected_evolution_seed
            or result.get("episode_seed_base") != expected_episode_seed_base
            or result.get("static_customer_schedule_sha256") != expected_static_schedule_sha256):
        raise ValueError("existing Pilot seed result differs from its manifest or provider context")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError("Pilot seed result artifact index is missing or malformed")
    project_root = Path.cwd().resolve()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("Pilot seed result contains a malformed artifact row")
        path = (project_root / str(item["path"])).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("Pilot seed result artifact escapes the project directory") from exc
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Pilot seed result references a missing or unsafe artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Pilot seed result artifact hash does not match its indexed digest")
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("Pilot seed result path is not a regular file")


def _audit_dict(audit: IndependentEpisodeAudit) -> dict[str, Any]:
    return {
        "verifier_ref": audit.verifier_ref,
        "customer_valid": audit.customer_valid,
        "strategy_applicable": audit.strategy_applicable,
        "customer_strategy_adherent": audit.customer_strategy_adherent,
        "policy_violation": audit.policy_violation,
        "invalid_repeated_write_calls": audit.invalid_repeated_write_calls,
        "policy_rule_id": audit.policy_rule_id,
        "mistake_type": audit.mistake_type,
        "workflow_stage": audit.workflow_stage,
        "evidence": [asdict(item) for item in audit.evidence],
    }
