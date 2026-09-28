"""Provider-agnostic Phase 1–3 mechanism orchestration.

Episode execution is injected. This module never constructs a model client or
silently interprets a task failure as an attributed Service failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .archive import FailureArchive
from .attribution import promote_verified_failure
from .budget import BudgetSnapshot, ProviderBudgetExceeded, RequestBudget
from .checkpoint import (
    EvolutionCheckpoint,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint,
)
from .manifest import MechanismManifest, sha256_json
from .mutation import CustomerCandidate, propose_customer_candidates
from .records import (
    CandidateEvaluation,
    EpisodeRecord,
    FailureRecord,
    customer_strategy_id,
    service_strategy_id,
)
from .selection import SelectionDecision, select_customer
from .service_evolution import GateReport
from .strategies import CustomerStrategy, ServiceRule, ServiceStrategy


class EpisodeRunner(Protocol):
    def __call__(self, *, task_id: str, seed: int, customer: CustomerStrategy | None,
                 service: ServiceStrategy, panel_name: str) -> EpisodeRecord: ...


@dataclass(slots=True)
class ServiceTransitionEpisodeRunner:
    """Episode capability plus the frozen history needed by a gated Service repair."""

    episode_runner: EpisodeRunner
    remaining_episodes: int
    customer_strategies: Mapping[str, CustomerStrategy]
    episode_history: tuple[EpisodeRecord, ...]

    def __post_init__(self) -> None:
        if self.remaining_episodes < 0:
            raise ValueError("remaining episode capacity cannot be negative")

    def __call__(self, *, task_id: str, seed: int, customer: CustomerStrategy | None,
                 service: ServiceStrategy, panel_name: str) -> EpisodeRecord:
        if self.remaining_episodes <= 0:
            raise RuntimeError("episode cap reached before Service gate dispatch")
        self.remaining_episodes -= 1
        return self.episode_runner(
            task_id=task_id, seed=seed, customer=customer, service=service,
            panel_name=panel_name,
        )

    def resolve_customer_strategy(self, strategy_id: str) -> CustomerStrategy | None:
        return self.customer_strategies.get(strategy_id)

    def find_failure_episode(self, failure: FailureRecord) -> EpisodeRecord | None:
        return next((item for item in self.episode_history if item.episode_id == failure.episode_id), None)

    def passing_history(self, *, task_id: str, service_id: str) -> tuple[EpisodeRecord, ...]:
        return tuple(sorted(
            (
                item for item in self.episode_history
                if item.task_id == task_id
                and item.service_strategy_id == service_id
                and item.status.value == "complete"
                and item.task_success is True
                and not item.policy_violation
                and item.customer_valid is True
                and item.strategy_applicable is True
                and item.customer_strategy_adherent is True
            ),
            key=lambda item: (item.seed, item.episode_id),
        ))


@dataclass(frozen=True, slots=True)
class CustomerRound:
    incumbent: CandidateEvaluation
    proposals: tuple[CustomerCandidate, ...]
    candidates: tuple[CandidateEvaluation, ...]
    selection: SelectionDecision
    verified_failures: tuple[FailureRecord, ...]
    confirmation_evaluations: tuple[CandidateEvaluation, ...] = ()


def evaluate_customer_panel(
    runner: EpisodeRunner,
    *,
    task_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    strategy: CustomerStrategy,
    service: ServiceStrategy,
    panel_name: str,
    generation: int,
    strategy_seen_failures: dict[str, str] | None = None,
    failure_verifier: Callable[[EpisodeRecord], str | None] | None = None,
    request_budget: RequestBudget | None = None,
) -> CandidateEvaluation:
    if not task_ids or not seeds:
        raise ValueError("evaluation panel needs at least one task and seed")
    customer_id, service_id = customer_strategy_id(strategy), service_strategy_id(service)
    episodes: list[EpisodeRecord] = []
    failures: list[FailureRecord] = []
    for task_id in task_ids:
        for seed in seeds:
            if request_budget is not None and request_budget.snapshot().remaining <= 0:
                raise ProviderBudgetExceeded("generation stopped before episode: shared request budget exhausted")
            episode = runner(task_id=task_id, seed=seed, customer=strategy, service=service, panel_name=panel_name)
            if episode.task_id != task_id or episode.seed != seed:
                raise ValueError("runner returned episode for a different task or seed")
            if episode.customer_strategy_id != customer_id or episode.service_strategy_id != service_id:
                raise ValueError("runner returned episode with mismatched strategy IDs")
            episodes.append(episode)
            verification_ref = None
            if episode.has_attributable_failure_candidate:
                if failure_verifier is not None:
                    verification_ref = failure_verifier(episode)
                elif strategy_seen_failures is not None:
                    verification_ref = strategy_seen_failures.get(episode.episode_id) or episode.audit_ref
                else:
                    verification_ref = episode.audit_ref
            decision = promote_verified_failure(
                episode, generation=generation,
                independent_verification_ref=verification_ref,
            )
            if decision.failure:
                failures.append(decision.failure)
    return CandidateEvaluation(customer_id, tuple(episodes), tuple(failures), panel_name)


def run_customer_round(
    runner: EpisodeRunner,
    *,
    incumbent: CustomerStrategy,
    service: ServiceStrategy,
    task_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    generation: int,
    proposal_seed: int,
    count: int = 2,
    verification_refs: dict[str, str] | None = None,
    failure_verifier: Callable[[EpisodeRecord], str | None] | None = None,
    already_seen: tuple[str, ...] | frozenset[str] = (),
    confirmation_task_ids: tuple[str, ...] = (),
    confirmation_seeds: tuple[int, ...] = (),
    request_budget: RequestBudget | None = None,
) -> CustomerRound:
    """Evaluate incumbent and K candidates on the same panel; ties retain incumbent."""
    incumbent_eval = evaluate_customer_panel(
        runner, task_ids=task_ids, seeds=seeds, strategy=incumbent, service=service,
        panel_name="discovery", generation=generation, strategy_seen_failures=verification_refs,
        failure_verifier=failure_verifier,
        request_budget=request_budget,
    )
    candidates = propose_customer_candidates(
        incumbent, count, seed=proposal_seed, recent_failures=incumbent_eval.verified_failures,
        already_seen=already_seen,
    )
    evaluations = tuple(evaluate_customer_panel(
        runner, task_ids=task_ids, seeds=seeds, strategy=candidate.strategy, service=service,
        panel_name="discovery", generation=generation, strategy_seen_failures=verification_refs,
        failure_verifier=failure_verifier,
        request_budget=request_budget,
    ) for candidate in candidates)
    confirmations: dict[str, CandidateEvaluation] | None = None
    if confirmation_task_ids and confirmation_seeds:
        best = max((item.fitness for item in evaluations), default=incumbent_eval.fitness)
        if best > incumbent_eval.fitness:
            winner = min((item for item in evaluations if item.fitness == best), key=lambda item: item.strategy_id)
            selected_strategy = next(item.strategy for item in candidates if item.strategy_id == winner.strategy_id)
            incumbent_confirm = evaluate_customer_panel(
                runner, task_ids=confirmation_task_ids, seeds=confirmation_seeds,
                strategy=incumbent, service=service, panel_name="confirmation",
                generation=generation, strategy_seen_failures=verification_refs,
                failure_verifier=failure_verifier,
                request_budget=request_budget,
            )
            winner_confirm = evaluate_customer_panel(
                runner, task_ids=confirmation_task_ids, seeds=confirmation_seeds,
                strategy=selected_strategy, service=service, panel_name="confirmation",
                generation=generation, strategy_seen_failures=verification_refs,
                failure_verifier=failure_verifier,
                request_budget=request_budget,
            )
            confirmations = {incumbent_eval.strategy_id: incumbent_confirm,
                             winner.strategy_id: winner_confirm}
    selection = select_customer(incumbent_eval, evaluations, confirmation=confirmations)
    failures = incumbent_eval.verified_failures + tuple(
        failure for evaluation in evaluations for failure in evaluation.verified_failures
    )
    return CustomerRound(
        incumbent_eval, candidates, evaluations, selection, failures,
        () if confirmations is None else tuple(confirmations.values()),
    )


@dataclass(frozen=True, slots=True)
class GenerationCommit:
    generation: int
    customer_id: str
    service_id: str
    customer_evolved: bool
    service_evolved: bool
    completed: bool
    note: str = ""
    decision_record: dict[str, Any] | None = None


ServiceTransition = Callable[
    [int, CustomerStrategy, ServiceStrategy, tuple[FailureRecord, ...], EpisodeRunner,
     RequestBudget | None],
    tuple[ServiceStrategy, GateReport | None, str],
]


class TwoGenerationSmoke:
    """A small injectable controller for mechanism testing, not an experiment CLI."""

    def __init__(self, *, manifest: dict | MechanismManifest, checkpoint_path: str,
                 runner: EpisodeRunner, task_ids: tuple[str, ...], seed: int,
                 request_budget: RequestBudget | None = None,
                 failure_archive: FailureArchive | None = None,
                 manifest_context: Mapping[str, Any] | None = None):
        if len(task_ids) != 2 or task_ids[0] == task_ids[1]:
            raise ValueError("mechanism smoke requires distinct evolution and validation task IDs")
        self.manifest = manifest
        manifest_payload = manifest.to_payload() if isinstance(manifest, MechanismManifest) else manifest
        self.manifest_context = None if manifest_context is None else dict(manifest_context)
        self.manifest_hash = manifest_fingerprint(
            manifest_payload if self.manifest_context is None else {
                "manifest": manifest_payload, "run_context": self.manifest_context,
            }
        )
        if isinstance(manifest, MechanismManifest):
            if (task_ids != (manifest.evolution_task_id, manifest.validation_task_id)
                    or seed != manifest.seed):
                raise ValueError("controller task IDs and seed must match the frozen manifest")
            if request_budget is not None and request_budget.snapshot().cap != manifest.request_budget_cap:
                raise ValueError("shared request budget cap must match the frozen manifest")
        self.checkpoint_path = checkpoint_path
        self.runner = runner
        self.task_ids = task_ids
        self.seed = seed
        self.request_budget = request_budget
        self.failure_archive = failure_archive
        self.max_episodes = manifest.max_episodes if isinstance(manifest, MechanismManifest) else int(manifest.get("max_episodes", 23))
        # Phase 3's 23-episode ceiling includes the Phase 0 integration episode.
        self.episode_attempts = 1 if self.manifest_context is not None else 0
        self._last_customer: CustomerStrategy | None = None
        self._last_service: ServiceStrategy | None = None
        self._commit_records: list[dict] = []
        self._episode_history: dict[str, EpisodeRecord] = {}
        self._progress: dict | None = None
        if request_budget is not None and request_budget.snapshot().cap > 1800:
            raise ValueError("Phase 3 shared provider-attempt budget cannot exceed 1,800")

    def _run_episode(self, **kwargs) -> EpisodeRecord:
        cache_key = _runner_cache_key(**kwargs)
        if self._progress is not None and cache_key in self._progress["episodes"]:
            return self._progress["episodes"][cache_key]
        if self.episode_attempts >= self.max_episodes:
            raise RuntimeError(f"episode cap {self.max_episodes} reached before dispatch")
        if self.request_budget is not None and self.request_budget.snapshot().remaining <= 0:
            raise ProviderBudgetExceeded("generation stopped before episode: shared request budget exhausted")
        self.episode_attempts += 1
        try:
            episode = self.runner(**kwargs)
            self._episode_history[cache_key] = episode
            if self._progress is not None:
                self._progress["episodes"][cache_key] = episode
            return episode
        finally:
            self._persist_progress()

    def _persist_progress(self) -> None:
        if self._progress is None:
            return
        customer = self._last_customer or CustomerStrategy(**self._progress["customer"])
        service = self._last_service or _service_from_dict(self._progress["service"])
        progress = {
            "generation": self._progress["generation"],
            "customer": self._progress["customer"],
            "service": self._progress["service"],
            "seen_strategy_ids": list(self._progress["seen_strategy_ids"]),
            "episodes": {key: episode.to_dict() for key, episode in self._progress["episodes"].items()},
            "prepared_generation": self._progress.get("prepared_generation"),
        }
        state = {
            "customer": customer.to_dict(), "service": service.to_dict(),
            "commits": list(self._commit_records), "seed": self.seed,
            "episode_attempts": self.episode_attempts,
            "run_context": self.manifest_context,
            "request_budget": _budget_snapshot_dict(self.request_budget),
            "episode_history": {key: episode.to_dict() for key, episode in self._episode_history.items()},
            "progress": progress,
        }
        committed_generation = max((item["generation"] for item in self._commit_records), default=0)
        checkpoint_generation = max(committed_generation, self._progress["generation"])
        save_checkpoint(self.checkpoint_path, EvolutionCheckpoint(
            self.manifest_hash, checkpoint_generation, state,
            tuple(f"generation:{item['generation']}" for item in self._commit_records),
        ))

    def _apply_prepared_archive(self, prepared: dict[str, Any]) -> None:
        _append_prepared_archive(self.failure_archive, prepared["archive"])
        decision = prepared["commit"].get("decision_record")
        if decision is None:
            raise ValueError("prepared generation is missing its immutable decision record")
        coverage = (
            None if self.failure_archive is None
            else self.failure_archive.active_replay_coverage()
        )
        existing = decision.get("active_replay_coverage")
        if existing is None:
            decision["active_replay_coverage"] = coverage
            self._persist_progress()
        elif existing != coverage:
            raise ValueError("prepared generation replay-coverage summary changed during recovery")

    def commit_generation(self, generation: int, customer: CustomerStrategy,
                          service: ServiceStrategy, *, note: str = "",
                          customer_evolved: bool = False, service_evolved: bool = False,
                          decision_record: dict[str, Any] | None = None) -> GenerationCommit:
        if generation not in (0, 1):
            raise ValueError("minimal smoke has exactly two generations (0 and 1)")
        # The controller accepts only immutable strategy snapshots after their
        # selection/gate decisions have completed.
        commit = GenerationCommit(generation, customer_strategy_id(customer),
                                  service_strategy_id(service), customer_evolved, service_evolved,
                                  True, note, decision_record)
        state = {
            "customer": customer.to_dict(), "service": service.to_dict(),
            "commits": [_commit_to_dict(commit)],
            "seed": self.seed,
            "episode_attempts": self.episode_attempts,
            "run_context": self.manifest_context,
            "request_budget": _budget_snapshot_dict(self.request_budget),
            "episode_history": {key: episode.to_dict() for key, episode in self._episode_history.items()},
            "progress": None,
        }
        # Preserve previous generation state while making each commit atomic.
        path = Path(self.checkpoint_path)
        if path.exists():
            from .checkpoint import load_checkpoint
            old = load_checkpoint(path, expected_manifest_hash=self.manifest_hash)
            completed = {item["generation"] for item in old.state.get("commits", [])}
            if generation in completed:
                prior = next(item for item in old.state["commits"] if item["generation"] == generation)
                if prior["customer_id"] == commit.customer_id and prior["service_id"] == commit.service_id:
                    return commit
                raise ValueError("generation already committed with different strategy state")
            if generation != (max(completed) + 1 if completed else 0):
                raise ValueError("generation commits must be sequential")
            state["commits"] = old.state.get("commits", []) + state["commits"]
        elif generation != 0:
            raise ValueError("generation commits must start at generation 0")
        save_checkpoint(self.checkpoint_path, EvolutionCheckpoint(
            self.manifest_hash, generation, state, tuple(f"generation:{item['generation']}" for item in state["commits"]),
        ))
        self._last_customer, self._last_service = customer, service
        self._commit_records = list(state["commits"])
        self._progress = None
        return commit

    def run(self, customer: CustomerStrategy, service: ServiceStrategy, *,
            verification_refs: dict[str, str] | None = None,
            failure_verifier: Callable[[EpisodeRecord], str | None] | None = None,
            service_transition: ServiceTransition | None = None,
            candidates_per_generation: int = 2) -> tuple[GenerationCommit, ...]:
        """Execute two Customer-first generations on E with a fresh confirmation seed.

        `failure_verifier` audits candidate traces independently. `service_transition`
        must return an already gate-checked strategy and report. The report
        retains the proposal, static audit, verification hypothesis, and the
        candidate that was evaluated, including on rejection. It is never
        allowed to bypass the structured gate by returning a changed Service
        with accepted=False.
        """
        if candidates_per_generation != 2:
            raise ValueError("minimal smoke freezes K=2 Customer candidates per generation")
        if isinstance(self.manifest, MechanismManifest):
            if sha256_json(customer.to_dict()) != self.manifest.customer_strategy_sha256:
                raise ValueError("initial Customer strategy does not match the frozen manifest")
            if sha256_json(service.to_dict()) != self.manifest.service_strategy_sha256:
                raise ValueError("initial Service strategy does not match the frozen manifest")
        commits: list[GenerationCommit] = []
        start_generation = 0
        if self._last_customer is None:
            self._last_customer, self._last_service = customer, service
        path = Path(self.checkpoint_path)
        if path.exists():
            restored = load_checkpoint(path, expected_manifest_hash=self.manifest_hash)
            if restored.state.get("run_context") != self.manifest_context:
                raise ValueError("checkpoint Phase 0 run context does not match the current run")
            self.episode_attempts = int(restored.state.get("episode_attempts", 0))
            self._last_customer = customer = CustomerStrategy(**restored.state["customer"])
            self._last_service = service = _service_from_dict(restored.state["service"])
            commits.extend(GenerationCommit(**item) for item in restored.state.get("commits", []))
            self._commit_records = list(restored.state.get("commits", []))
            self._episode_history = {
                key: EpisodeRecord.from_dict(value)
                for key, value in restored.state.get("episode_history", {}).items()
            }
            saved_budget = restored.state.get("request_budget")
            if self.request_budget is not None and saved_budget is not None:
                self.request_budget.restore_usage(BudgetSnapshot(**saved_budget))
            elif saved_budget is not None and saved_budget.get("attempts", 0) > 0:
                raise ValueError("resuming a budgeted run requires restoring its shared RequestBudget")
            progress = restored.state.get("progress")
            if progress is not None:
                self._progress = {
                    "generation": int(progress["generation"]),
                    "customer": progress["customer"], "service": progress["service"],
                    "seen_strategy_ids": tuple(progress["seen_strategy_ids"]),
                    "episodes": {key: EpisodeRecord.from_dict(value)
                                 for key, value in progress["episodes"].items()},
                    "prepared_generation": progress.get("prepared_generation"),
                }
                customer = CustomerStrategy(**self._progress["customer"])
                service = _service_from_dict(self._progress["service"])
                start_generation = self._progress["generation"]
            else:
                start_generation = max((item["generation"] for item in self._commit_records), default=-1) + 1
        elif isinstance(self.manifest, MechanismManifest) and self.manifest.real_provider_enabled and self.request_budget is None:
            raise ValueError("live provider-enabled mechanism runs require a shared RequestBudget")
        for generation in range(start_generation, 2):
            if self._progress is None or self._progress["generation"] != generation:
                self._progress = {
                    "generation": generation, "customer": customer.to_dict(),
                    "service": service.to_dict(),
                    "seen_strategy_ids": tuple(self.failure_archive.customer_strategy_ids()
                                                if self.failure_archive is not None else ()),
                    "episodes": {},
                    "prepared_generation": None,
                }
            prepared = self._progress.get("prepared_generation")
            if prepared is not None:
                customer = CustomerStrategy(**prepared["customer"])
                service = _service_from_dict(prepared["service"])
                self._apply_prepared_archive(prepared)
                expected = GenerationCommit(**prepared["commit"])
                commit = self.commit_generation(
                    generation, customer, service, note=expected.note,
                    customer_evolved=expected.customer_evolved,
                    service_evolved=expected.service_evolved,
                    decision_record=expected.decision_record,
                )
                if commit != expected:
                    raise ValueError("prepared generation differs from its persisted decision record")
                commits.append(commit)
                continue
            seed = self.seed + generation
            round_result = run_customer_round(
                self._run_episode, incumbent=customer, service=service,
                task_ids=(self.task_ids[0],), seeds=(seed,), generation=generation,
                proposal_seed=self.seed + generation * 1009, count=candidates_per_generation,
                verification_refs=verification_refs,
                failure_verifier=failure_verifier,
                already_seen=self._progress["seen_strategy_ids"],
                confirmation_task_ids=(self.task_ids[0],),
                confirmation_seeds=(self.seed + 10_000 + generation,),
                request_budget=self.request_budget,
            )
            old_customer = customer
            if round_result.selection.evolved:
                selected = next(item.strategy for item in round_result.proposals
                                if item.strategy_id == round_result.selection.selected_id)
                customer = selected
            old_service = service
            service_evolved = False
            note = round_result.selection.reason
            gate: GateReport | None = None
            service_note = note
            failure_pool = {failure.failure_id: failure for failure in round_result.verified_failures}
            if self.failure_archive is not None:
                failure_pool.update({
                    failure.failure_id: failure
                    for failure in self.failure_archive.active_representatives(
                        current_generation=generation,
                    )
                })
            if service_transition is not None and failure_pool:
                strategy_snapshots = {
                    customer_strategy_id(item): item
                    for item in (old_customer, customer, *(proposal.strategy for proposal in round_result.proposals))
                }
                observed_episodes = tuple({
                    episode.episode_id: episode
                    for episode in (
                        *self._episode_history.values(),
                        *round_result.incumbent.episodes,
                        *(episode for evaluation in round_result.candidates for episode in evaluation.episodes),
                        *(episode for evaluation in round_result.confirmation_evaluations for episode in evaluation.episodes),
                    )
                }.values())
                if self.failure_archive is not None:
                    for failure in failure_pool.values():
                        strategy_id = failure.customer_strategy_id
                        if strategy_id not in strategy_snapshots:
                            archived_strategy = self.failure_archive.get_customer_strategy(strategy_id)
                            if archived_strategy is not None:
                                strategy_snapshots[strategy_id] = archived_strategy
                transition_runner = ServiceTransitionEpisodeRunner(
                    self._run_episode,
                    max(0, self.max_episodes - self.episode_attempts),
                    strategy_snapshots,
                    observed_episodes,
                )
                if self.request_budget is None:
                    proposed, gate, service_note = service_transition(
                        generation, customer, service, tuple(failure_pool.values()),
                        transition_runner, None,
                    )
                else:
                    from tau2.utils import llm_utils

                    with self.request_budget.instrument_tau_llm_utils(llm_utils):
                        proposed, gate, service_note = service_transition(
                            generation, customer, service, tuple(failure_pool.values()),
                            transition_runner, self.request_budget,
                        )
                accepted = gate is not None and gate.accepted and gate.candidate_strategy == proposed
                if gate is not None and gate.accepted and proposed != gate.candidate_strategy:
                    raise ValueError("accepted repair must return the exact candidate named by its GateReport")
                if proposed != service and (not accepted or not failure_pool):
                    raise ValueError("rejected Service repair must leave incumbent unchanged")
                service, service_evolved, note = proposed, accepted and proposed != old_service, service_note
            elif service_transition is not None:
                note = "no verified failure is available for a Service repair"
                service_note = note
            elif failure_pool:
                raise ValueError(
                    "verified Service failures require an audited ServiceTransition before generation commit"
                )
            decision_record = _generation_decision_record(
                generation=generation,
                old_customer=old_customer,
                new_customer=customer,
                old_service=old_service,
                new_service=service,
                round_result=round_result,
                failure_pool=tuple(failure_pool.values()),
                service_transition_ran=service_transition is not None and bool(failure_pool),
                service_gate=gate,
                service_note=service_note,
            )
            decision_record = json.loads(json.dumps(decision_record, ensure_ascii=False))
            commit = GenerationCommit(
                generation, customer_strategy_id(customer), service_strategy_id(service),
                customer != old_customer, service_evolved, True, note, decision_record,
            )
            archive_payload = _prepared_archive_payload(
                generation=generation,
                old_customer=old_customer,
                proposals=round_result.proposals,
                failures=round_result.verified_failures,
                old_service=old_service,
                new_service=service,
                service_gate=gate,
            )
            self._progress["prepared_generation"] = {
                "customer": customer.to_dict(),
                "service": service.to_dict(),
                "commit": _commit_to_dict(commit),
                "archive": archive_payload,
            }
            self._persist_progress()
            self._apply_prepared_archive(self._progress["prepared_generation"])
            commit = self.commit_generation(
                generation, customer, service, note=note,
                customer_evolved=customer != old_customer,
                service_evolved=service_evolved,
                decision_record=decision_record,
            )
            commits.append(commit)
        return tuple(commits)


def _service_from_dict(value: dict) -> ServiceStrategy:
    return ServiceStrategy(tuple(ServiceRule(
        rule_id=item["rule_id"], policy_ref=item["policy_ref"], trigger=item["trigger"],
        required_execution=item["required_execution"], evidence_refs=tuple(item["evidence_refs"]),
    ) for item in value["rules"]))


def _budget_snapshot_dict(budget: RequestBudget | None) -> dict | None:
    if budget is None:
        return None
    return asdict(budget.snapshot())


def _commit_to_dict(commit: GenerationCommit) -> dict[str, Any]:
    return {
        "generation": commit.generation,
        "customer_id": commit.customer_id,
        "service_id": commit.service_id,
        "customer_evolved": commit.customer_evolved,
        "service_evolved": commit.service_evolved,
        "completed": commit.completed,
        "note": commit.note,
        "decision_record": commit.decision_record,
    }


def _generation_decision_record(
    *,
    generation: int,
    old_customer: CustomerStrategy,
    new_customer: CustomerStrategy,
    old_service: ServiceStrategy,
    new_service: ServiceStrategy,
    round_result: CustomerRound,
    failure_pool: tuple[FailureRecord, ...],
    service_transition_ran: bool,
    service_gate: GateReport | None,
    service_note: str,
) -> dict[str, Any]:
    return {
        "generation": generation,
        "customer": {
            "incumbent_before": old_customer.to_dict(),
            "incumbent_after": new_customer.to_dict(),
            "proposals": [
                {
                    "strategy_id": proposal.strategy_id,
                    "strategy": proposal.strategy.to_dict(),
                    "parent_id": proposal.parent_id,
                    "operator": proposal.operator,
                    "rationale": proposal.rationale,
                    "changed_fields": list(proposal.changed_fields),
                    "expected_behavioral_effect": proposal.expected_behavioral_effect,
                    "supporting_failure_ids": list(proposal.supporting_failure_ids),
                }
                for proposal in round_result.proposals
            ],
            "evaluations": [
                round_result.incumbent.to_dict(),
                *(item.to_dict() for item in round_result.candidates),
                *(item.to_dict() for item in round_result.confirmation_evaluations),
            ],
            "selection": asdict(round_result.selection),
        },
        "service": {
            "incumbent_before": old_service.to_dict(),
            "incumbent_after": new_service.to_dict(),
            "transition_ran": service_transition_ran,
            "failure_ids": [failure.failure_id for failure in failure_pool],
            "note": service_note,
            "gate": None if service_gate is None else service_gate.to_dict(),
        },
        "verified_failures": [failure.to_dict() for failure in round_result.verified_failures],
    }


def _prepared_archive_payload(
    *,
    generation: int,
    old_customer: CustomerStrategy,
    proposals: tuple[CustomerCandidate, ...],
    failures: tuple[FailureRecord, ...],
    old_service: ServiceStrategy,
    new_service: ServiceStrategy,
    service_gate: GateReport | None,
) -> dict[str, Any]:
    customers = [{
        "strategy": old_customer.to_dict(), "parent_id": None,
        "operator": "incumbent", "generation": generation,
    }]
    customers.extend({
        "strategy": proposal.strategy.to_dict(), "parent_id": proposal.parent_id,
        "operator": proposal.operator, "generation": generation,
    } for proposal in proposals)
    services = [{
        "strategy": old_service.to_dict(), "parent_id": None,
        "operator": "incumbent", "generation": generation,
    }]
    if new_service != old_service:
        services.append({
            "strategy": new_service.to_dict(),
            "parent_id": service_strategy_id(old_service),
            "operator": "verified_repair", "generation": generation,
        })
    replayed_target = (
        service_gate is not None
        and any(key.startswith("target-") for key, _passed, _reason in service_gate.unit_results)
    )
    return {
        "customers": customers,
        "services": services,
        "failures": [failure.to_dict() for failure in failures],
        "replays": ([{"failure_id": service_gate.target_failure_id, "generation": generation}]
                    if replayed_target else []),
    }


def _append_prepared_archive(
    archive: FailureArchive | None,
    payload: Mapping[str, Any],
) -> None:
    if archive is None:
        return
    for item in payload["customers"]:
        archive.append_customer_strategy(
            CustomerStrategy(**item["strategy"]), parent_id=item["parent_id"],
            operator=item["operator"], generation=int(item["generation"]),
        )
    for item in payload["services"]:
        archive.append_service_strategy(
            _service_from_dict(item["strategy"]), parent_id=item["parent_id"],
            operator=item["operator"], generation=int(item["generation"]),
        )
    for item in payload["failures"]:
        archive.append(FailureRecord.from_dict(item))
    for replay in payload.get("replays", ()):
        archive.mark_replayed(replay["failure_id"], generation=int(replay["generation"]))


def _runner_cache_key(*, task_id: str, seed: int, customer: CustomerStrategy | None,
                      service: ServiceStrategy, panel_name: str) -> str:
    return sha256_json({"task_id": task_id, "seed": seed,
                        "customer": customer_strategy_id(customer),
                        "service": service_strategy_id(service), "panel": panel_name})
