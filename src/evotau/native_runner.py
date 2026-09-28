"""Manifest-bound native τ-bench episode adapter for the Phase 1–3 controller."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .budget import BudgetSnapshot, RequestBudget
from .manifest import MechanismManifest, sha256_json, write_manifest_once
from .phase0_run import _load_pinned_tasks, _write_json_once
from .records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
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
        manifest: MechanismManifest,
        config: Mapping[str, Any],
        data_dir: str | Path,
        request_budget: RequestBudget,
        audit_provider: AuditProvider | None = None,
        service_token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if not manifest.real_provider_enabled:
            raise RuntimeError("native Phase 3 runs require explicit real_provider_enabled opt-in")
        if request_budget.snapshot().cap != manifest.request_budget_cap:
            raise ValueError("shared request budget cap must match the frozen mechanism manifest")
        self.models = dict(manifest.role_models)
        if set(self.models) != {"agent", "customer", "reviewer"} or any(
            not self.models[name] for name in self.models
        ):
            raise ValueError("native Phase 3 runs require frozen agent, customer, and reviewer models")
        experiment = config.get("experiment", {})
        if (experiment.get("id") != manifest.experiment_id
                or MechanismManifest.from_mapping(config).sha256 != manifest.sha256):
            raise ValueError("run configuration does not match the frozen mechanism manifest")
        self.manifest = manifest
        self.request_budget = request_budget
        self.audit_provider = audit_provider
        self.service_token_counter = service_token_counter
        self.tasks = _load_pinned_tasks(
            manifest,
            data_dir=data_dir,
            task_selection=experiment.get("task_selection", {}),
            task_ids=(manifest.evolution_task_id, manifest.validation_task_id),
        )
        self.output_directory = Path(manifest.output_path)
        self.output_directory.mkdir(parents=True, exist_ok=True)
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

    def __call__(
        self,
        *,
        task_id: str,
        seed: int,
        customer: CustomerStrategy | None,
        service: ServiceStrategy,
        panel_name: str,
    ) -> EpisodeRecord:
        try:
            task = self.tasks[str(task_id)]
        except KeyError as exc:
            raise ValueError(f"task {task_id!r} is outside the frozen E/V task set") from exc
        if seed < 0 or not panel_name.strip():
            raise ValueError("native episode seed and panel name must be valid")
        if service.rules and self.service_token_counter is None:
            raise ValueError("non-empty ServiceStrategy requires a model-matched token counter")
        if self.request_budget.snapshot().remaining <= 0:
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
                task=task,
                agent_model=self.models["agent"],
                customer_model=self.models["customer"],
                seed=seed,
                max_steps=self.manifest.max_steps,
                customer_strategy=customer,
                service_strategy=service,
                service_token_counter=self.service_token_counter,
            )
            on_orchestrator(orchestrator)
            simulation = run_with_budget(
                orchestrator,
                self.request_budget,
                reviewer_model=self.models["reviewer"],
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
                customer_valid=None if audit_result is None else audit_result.customer_valid,
                strategy_applicable=None if audit_result is None else audit_result.strategy_applicable,
                customer_strategy_adherent=None if audit_result is None else audit_result.customer_strategy_adherent,
                policy_violation=False if audit_result is None else audit_result.policy_violation,
                policy_rule_id=None if audit_result is None else audit_result.policy_rule_id,
                mistake_type=None if audit_result is None else audit_result.mistake_type,
                workflow_stage=None if audit_result is None else audit_result.workflow_stage,
                evidence=() if audit_result is None else audit_result.evidence,
                trajectory_ref=simulation_path.relative_to(self.output_directory).as_posix(),
                audit_ref=None if audit_result is None else audit_result.verifier_ref,
                tool_calls=_count_tool_calls(simulation_payload.get("messages") or ()),
                raw_review={
                    "native_review": simulation_payload.get("review"),
                    "auth_classification": simulation_payload.get("auth_classification"),
                },
            )
            after = self.request_budget.snapshot()
            _write_json_once(telemetry_path, {
                "attempt_id": attempt_id,
                "simulation_id": episode_id,
                "panel_name": panel_name,
                "customer_strategy_id": record.customer_strategy_id,
                "service_strategy_id": record.service_strategy_id,
                "rendered_prompt_sha256": prompt_hashes,
                "budget_before": asdict(before),
                "budget_after": asdict(after),
                "budget_delta": _snapshot_delta(before, after),
                "independent_audit": None if audit_result is None else _audit_dict(audit_result),
            })
            _write_json_once(record_path, record.to_dict())
            return record
        except Exception as exc:
            after = self.request_budget.snapshot()
            _write_json_once(episode_directory / "incomplete-run.json", {
                "attempt_id": attempt_id,
                "task_id": str(task_id),
                "seed": seed,
                "panel_name": panel_name,
                "failure_type": type(exc).__name__,
                "native_simulation_saved": simulation_path.exists(),
                "rendered_prompt_sha256": prompt_hashes,
                "budget_before": asdict(before),
                "budget_after": asdict(after),
                "budget_delta": _snapshot_delta(before, after),
            })
            raise NativeEpisodeRunError(type(exc).__name__) from exc


def run_native_phase3(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    phase0_result_path: str | Path,
    audit_provider: AuditProvider,
    service_transition: Callable[..., Any],
) -> tuple[tuple[Any, ...], BudgetSnapshot]:
    """Run the frozen two-generation controller on native τ-bench episodes.

    A config must explicitly enable the provider and freeze all three role
    models. `audit_provider` must return independent Customer validity,
    adherence, and policy-attribution judgments; the native τ-bench reviewer
    alone is intentionally insufficient. A verified failure also requires a
    `service_transition` callback before the generation can commit.
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
    if service_transition is None:
        raise ValueError("native Phase 3 requires an audited ServiceTransition provider")
    phase0_budget, run_context = _validate_phase0_parent(phase0_result_path, manifest)
    customer, service = _parse_strategies(config["experiment"])
    customer = customer or CustomerStrategy()
    agent_model = dict(manifest.role_models)["agent"]
    from litellm import token_counter

    service_token_counter = lambda text: token_counter(model=agent_model, text=text)
    budget = RequestBudget(manifest.request_budget_cap)
    output_directory = Path(manifest.output_path)
    context_path = output_directory / "run-context.json"
    checkpoint_path = Path(manifest.checkpoint_path)
    resuming = checkpoint_path.exists()
    if context_path.exists():
        saved_context = json.loads(context_path.read_text(encoding="utf-8"))
        if saved_context != run_context:
            raise ValueError("existing Phase 3 run context belongs to a different Phase 0 result")
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
        candidates_per_generation=manifest.customer_candidates,
    )
    return commits, budget.snapshot()


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
    if phase0_document.get("role_models") != dict(manifest.role_models):
        raise ValueError("Phase 0 and Phase 3 frozen role models differ")
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
        "phase0_provider_budget": asdict(phase0_budget),
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


def _snapshot_delta(before: BudgetSnapshot, after: BudgetSnapshot) -> dict[str, int]:
    return {
        name: getattr(after, name) - getattr(before, name)
        for name in (
            "attempts", "successes", "failures", "denied", "prompt_tokens",
            "completion_tokens", "usage_responses", "usage_unavailable", "cache_hits",
        )
    }


def _audit_dict(audit: IndependentEpisodeAudit) -> dict[str, Any]:
    return {
        "verifier_ref": audit.verifier_ref,
        "customer_valid": audit.customer_valid,
        "strategy_applicable": audit.strategy_applicable,
        "customer_strategy_adherent": audit.customer_strategy_adherent,
        "policy_violation": audit.policy_violation,
        "policy_rule_id": audit.policy_rule_id,
        "mistake_type": audit.mistake_type,
        "workflow_stage": audit.workflow_stage,
        "evidence": [asdict(item) for item in audit.evidence],
    }
