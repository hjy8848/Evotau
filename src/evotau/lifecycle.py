"""Provider-agnostic Phase 1–3 mechanism orchestration.

Episode execution is injected. This module never constructs a model client or
silently interprets a task failure as an attributed Service failure.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from .archive import FailureArchive
from .attribution import confirm_failure_reproductions
from .baselines import propose_random_mutation_candidates
from .budget import BudgetSnapshot, ProviderBudgetExceeded, RequestBudget
from .checkpoint import (
    EvolutionCheckpoint,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint,
)
from .customer_evolver import (
    OperatorSelector,
    propose_customer_candidates_with_selector,
)
from .episode_execution import (
    EpisodeSpec,
    StopBeforeEpisodeDispatch,
    run_episode_batch,
)
from .manifest import MechanismManifest, PilotManifest, sha256_json
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
    service_policy_text: str | None = None
    trajectory_loader: Callable[[EpisodeRecord], Mapping[str, Any] | None] | None = None

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

    def load_trajectory(self, episode: EpisodeRecord) -> Mapping[str, Any] | None:
        if self.trajectory_loader is None:
            return None
        return self.trajectory_loader(episode)

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
    max_concurrency: int = 1,
    reserve_episode: Callable[[EpisodeSpec], bool] | None = None,
    release_episode: Callable[[EpisodeSpec], None] | None = None,
    needs_provider_request: Callable[[EpisodeSpec], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    episode_finished: Callable[[EpisodeSpec], None] | None = None,
    batch_finished: Callable[[], None] | None = None,
) -> CandidateEvaluation:
    if not task_ids or not seeds:
        raise ValueError("evaluation panel needs at least one task and seed")
    if (len(set(task_ids)) != len(task_ids) or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("evaluation panel task IDs and seeds must be unique and valid")
    (evaluation,) = _evaluate_customer_panels(
        runner,
        task_ids=task_ids,
        seeds=seeds,
        strategies=(strategy,),
        service=service,
        panel_name=panel_name,
        generation=generation,
        strategy_seen_failures=strategy_seen_failures,
        failure_verifier=failure_verifier,
        request_budget=request_budget,
        max_concurrency=max_concurrency,
        reserve_episode=reserve_episode,
        release_episode=release_episode,
        needs_provider_request=needs_provider_request,
        should_pause=should_pause,
        episode_finished=episode_finished,
        batch_finished=batch_finished,
    )
    return evaluation


def _evaluate_customer_panels(
    runner: EpisodeRunner,
    *,
    task_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    strategies: tuple[CustomerStrategy, ...],
    service: ServiceStrategy,
    panel_name: str,
    generation: int,
    strategy_seen_failures: dict[str, str] | None = None,
    failure_verifier: Callable[[EpisodeRecord], str | None] | None = None,
    request_budget: RequestBudget | None = None,
    max_concurrency: int = 1,
    reserve_episode: Callable[[EpisodeSpec], bool] | None = None,
    release_episode: Callable[[EpisodeSpec], None] | None = None,
    needs_provider_request: Callable[[EpisodeSpec], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    episode_finished: Callable[[EpisodeSpec], None] | None = None,
    batch_finished: Callable[[], None] | None = None,
) -> tuple[CandidateEvaluation, ...]:
    if not task_ids or not seeds or not strategies:
        raise ValueError("evaluation panel needs at least one task, seed, and Customer strategy")
    if (len(set(task_ids)) != len(task_ids) or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("evaluation panel task IDs and seeds must be unique and valid")
    strategy_ids = tuple(customer_strategy_id(item) for item in strategies)
    if len(set(strategy_ids)) != len(strategy_ids):
        raise ValueError("a parallel evaluation batch cannot repeat a Customer strategy")
    specs = tuple(
        EpisodeSpec(task_id, seed, strategy, service, panel_name, generation)
        for strategy in strategies
        for task_id in task_ids
        for seed in seeds
    )
    episodes = run_episode_batch(
        runner,
        specs,
        max_concurrency=max_concurrency,
        request_budget=request_budget,
        reserve_episode=reserve_episode,
        release_episode=release_episode,
        needs_provider_request=needs_provider_request,
        should_pause=should_pause,
        episode_finished=episode_finished,
        batch_finished=batch_finished,
    )
    grouped: dict[str, list[EpisodeRecord]] = {customer_strategy_id(item): [] for item in strategies}
    audit_refs: dict[str, list[tuple[str, str]]] = {
        customer_strategy_id(item): [] for item in strategies
    }
    service_id = service_strategy_id(service)
    for spec, episode in zip(specs, episodes, strict=True):
        strategy_id = spec.customer_id
        if episode.task_id != spec.task_id or episode.seed != spec.seed:
            raise ValueError("runner returned episode for a different task or seed")
        if episode.customer_strategy_id != strategy_id or episode.service_strategy_id != service_id:
            raise ValueError("runner returned episode with mismatched strategy IDs")
        grouped[strategy_id].append(episode)
        verification_ref = None
        if episode.has_attributable_failure_candidate:
            if failure_verifier is not None:
                verification_ref = failure_verifier(episode)
            elif strategy_seen_failures is not None:
                verification_ref = strategy_seen_failures.get(episode.episode_id) or episode.audit_ref
            else:
                verification_ref = episode.audit_ref
        if episode.has_attributable_failure_candidate and verification_ref and verification_ref.strip():
            audit_refs[strategy_id].append((episode.episode_id, verification_ref.strip()))
    return tuple(
        CandidateEvaluation(
            customer_strategy_id(strategy), tuple(grouped[customer_strategy_id(strategy)]), (), panel_name,
            tuple(audit_refs[customer_strategy_id(strategy)]),
        )
        for strategy in strategies
    )


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
    prior_failures: tuple[FailureRecord, ...] = (),
    confirmation_task_ids: tuple[str, ...] = (),
    confirmation_seeds: tuple[int, ...] = (),
    request_budget: RequestBudget | None = None,
    max_concurrency: int = 1,
    reserve_episode: Callable[[EpisodeSpec], bool] | None = None,
    release_episode: Callable[[EpisodeSpec], None] | None = None,
    needs_provider_request: Callable[[EpisodeSpec], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    episode_finished: Callable[[EpisodeSpec], None] | None = None,
    batch_finished: Callable[[], None] | None = None,
    proposal_provider: OperatorSelector | None = None,
    proposal_mode: str = "failure_conditioned",
    allow_strategy_revisit: bool = False,
) -> CustomerRound:
    """Evaluate a shared panel, then replay audited signals before scoring failures."""
    if (not task_ids or not seeds or len(set(task_ids)) != len(task_ids)
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Customer discovery requires a unique frozen task/seed panel")
    if bool(confirmation_task_ids) != bool(confirmation_seeds):
        raise ValueError("failure confirmation requires both tasks and fresh seeds")
    if confirmation_task_ids:
        if (confirmation_task_ids != task_ids
                or len(set(confirmation_task_ids)) != len(confirmation_task_ids)
                or len(set(confirmation_seeds)) != len(confirmation_seeds)
                or any(type(seed) is not int or seed < 0 for seed in confirmation_seeds)):
            raise ValueError("failure confirmation must preserve the ordered discovery task panel")
        if set(zip(task_ids, seeds)) & set(zip(confirmation_task_ids, confirmation_seeds)):
            raise ValueError("failure confirmation requires fresh task/seed pairs")
    execution_options = {
        "request_budget": request_budget,
        "max_concurrency": max_concurrency,
        "reserve_episode": reserve_episode,
        "release_episode": release_episode,
        "needs_provider_request": needs_provider_request,
        "should_pause": should_pause,
        "episode_finished": episode_finished,
        "batch_finished": batch_finished,
    }
    incumbent_eval = evaluate_customer_panel(
        runner, task_ids=task_ids, seeds=seeds, strategy=incumbent, service=service,
        panel_name="discovery", generation=generation, strategy_seen_failures=verification_refs,
        failure_verifier=failure_verifier,
        **execution_options,
    )
    if proposal_mode not in {"failure_conditioned", "random_mutation"}:
        raise ValueError("Customer proposal mode must be failure_conditioned or random_mutation")
    if type(allow_strategy_revisit) is not bool:
        raise TypeError("allow_strategy_revisit must be boolean")
    seen_for_proposal = () if allow_strategy_revisit else already_seen
    if proposal_mode == "random_mutation":
        if proposal_provider is not None:
            raise ValueError("random-mutation control cannot use a failure-aware proposal provider")
        candidates = propose_random_mutation_candidates(
            incumbent, count, seed=proposal_seed, already_seen=seen_for_proposal,
        )
    elif proposal_provider is None:
        candidates = propose_customer_candidates(
            incumbent, count, seed=proposal_seed, recent_failures=prior_failures,
            already_seen=seen_for_proposal,
        )
    elif request_budget is None:
        candidates = propose_customer_candidates_with_selector(
            incumbent, count, generation=generation, seed=proposal_seed,
            recent_failures=prior_failures, already_seen=tuple(seen_for_proposal),
            evolution_task_ids=task_ids, operator_selector=proposal_provider,
        )
    else:
        from tau2.utils import llm_utils

        with request_budget.instrument_tau_llm_utils(llm_utils):
            candidates = propose_customer_candidates_with_selector(
                incumbent, count, generation=generation, seed=proposal_seed,
                recent_failures=prior_failures, already_seen=tuple(seen_for_proposal),
                evolution_task_ids=task_ids, operator_selector=proposal_provider,
            )
    evaluations = list(_evaluate_customer_panels(
        runner, task_ids=task_ids, seeds=seeds,
        strategies=tuple(candidate.strategy for candidate in candidates), service=service,
        panel_name="discovery", generation=generation, strategy_seen_failures=verification_refs,
        failure_verifier=failure_verifier, **execution_options,
    ))
    confirmations: dict[str, CandidateEvaluation] | None = None
    confirmation_evaluations: list[CandidateEvaluation] = []
    probe = _choose_failure_replay_probe(incumbent_eval, tuple(evaluations))
    can_replay = bool(confirmation_task_ids and confirmation_seeds)
    if probe is not None and can_replay:
        selected_strategy = next(item.strategy for item in candidates if item.strategy_id == probe.strategy_id)
        incumbent_confirm, probe_confirm = _evaluate_customer_panels(
            runner, task_ids=confirmation_task_ids, seeds=confirmation_seeds,
            strategies=(incumbent, selected_strategy), service=service, panel_name="confirmation",
            generation=generation, strategy_seen_failures=verification_refs,
            failure_verifier=failure_verifier, **execution_options,
        )
        incumbent_eval, incumbent_confirm, _ = confirm_failure_reproductions(
            incumbent_eval, incumbent_confirm, generation=generation,
        )
        probe, probe_confirm, _ = confirm_failure_reproductions(
            probe, probe_confirm, generation=generation,
        )
        evaluations = [probe if item.strategy_id == probe.strategy_id else item for item in evaluations]
        confirmations = {
            incumbent_eval.strategy_id: incumbent_confirm,
            probe.strategy_id: probe_confirm,
        }
        confirmation_evaluations.extend((incumbent_confirm, probe_confirm))
    elif incumbent_eval.provisional_failure_events and can_replay:
        # A verified incumbent failure may justify a Service repair even when no
        # Customer candidate offers a novel or strictly stronger attack signal.
        (incumbent_confirm,) = _evaluate_customer_panels(
            runner, task_ids=confirmation_task_ids, seeds=confirmation_seeds,
            strategies=(incumbent,), service=service, panel_name="confirmation",
            generation=generation, strategy_seen_failures=verification_refs,
            failure_verifier=failure_verifier, **execution_options,
        )
        incumbent_eval, incumbent_confirm, _ = confirm_failure_reproductions(
            incumbent_eval, incumbent_confirm, generation=generation,
        )
        confirmation_evaluations.append(incumbent_confirm)
    selection = select_customer(incumbent_eval, evaluations, confirmation=confirmations)
    failures_by_id = {
        failure.failure_id: failure
        for evaluation in (incumbent_eval, *evaluations, *confirmation_evaluations)
        for failure in evaluation.verified_failures
    }
    return CustomerRound(
        incumbent_eval, candidates, tuple(evaluations), selection,
        tuple(failures_by_id[key] for key in sorted(failures_by_id)),
        tuple(confirmation_evaluations),
    )


def run_frozen_customer_round(
    runner: EpisodeRunner,
    *,
    incumbent: CustomerStrategy,
    service: ServiceStrategy,
    task_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    generation: int,
    verification_refs: dict[str, str] | None = None,
    failure_verifier: Callable[[EpisodeRecord], str | None] | None = None,
    confirmation_task_ids: tuple[str, ...] = (),
    confirmation_seeds: tuple[int, ...] = (),
    request_budget: RequestBudget | None = None,
    max_concurrency: int = 1,
    reserve_episode: Callable[[EpisodeSpec], bool] | None = None,
    release_episode: Callable[[EpisodeSpec], None] | None = None,
    needs_provider_request: Callable[[EpisodeSpec], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    episode_finished: Callable[[EpisodeSpec], None] | None = None,
    batch_finished: Callable[[], None] | None = None,
) -> CustomerRound:
    """Collect fixed-Customer failure evidence without proposing or testing mutations."""

    incumbent_evaluation = evaluate_customer_panel(
        runner,
        task_ids=task_ids,
        seeds=seeds,
        strategy=incumbent,
        service=service,
        panel_name="discovery",
        generation=generation,
        strategy_seen_failures=verification_refs,
        failure_verifier=failure_verifier,
        request_budget=request_budget,
        max_concurrency=max_concurrency,
        reserve_episode=reserve_episode,
        release_episode=release_episode,
        needs_provider_request=needs_provider_request,
        should_pause=should_pause,
        episode_finished=episode_finished,
        batch_finished=batch_finished,
    )
    confirmations: tuple[CandidateEvaluation, ...] = ()
    if incumbent_evaluation.provisional_failure_events and confirmation_task_ids:
        confirmation = evaluate_customer_panel(
            runner,
            task_ids=confirmation_task_ids,
            seeds=confirmation_seeds,
            strategy=incumbent,
            service=service,
            panel_name="confirmation",
            generation=generation,
            strategy_seen_failures=verification_refs,
            failure_verifier=failure_verifier,
            request_budget=request_budget,
            max_concurrency=max_concurrency,
            reserve_episode=reserve_episode,
            release_episode=release_episode,
            needs_provider_request=needs_provider_request,
            should_pause=should_pause,
            episode_finished=episode_finished,
            batch_finished=batch_finished,
        )
        incumbent_evaluation, confirmation, _failures = confirm_failure_reproductions(
            incumbent_evaluation, confirmation, generation=generation,
        )
        confirmations = (confirmation,)
    strategy_id = customer_strategy_id(incumbent)
    selection = SelectionDecision(
        incumbent_id=strategy_id,
        selected_id=strategy_id,
        evolved=False,
        reason="Customer is frozen by the registered ablation",
        discovery_scores=((strategy_id, incumbent_evaluation.fitness),),
    )
    return CustomerRound(
        incumbent=incumbent_evaluation,
        proposals=(),
        candidates=(),
        selection=selection,
        verified_failures=incumbent_evaluation.verified_failures,
        confirmation_evaluations=confirmations,
    )


def _choose_failure_replay_probe(
    incumbent: CandidateEvaluation,
    candidates: tuple[CandidateEvaluation, ...],
) -> CandidateEvaluation | None:
    """Use audited signals only to ration one fresh-seed reproduction probe."""
    base_score = incumbent.provisional_failure_count
    base_events = incumbent.provisional_failure_events
    strict = [item for item in candidates if item.provisional_failure_count > base_score]
    if strict:
        return min(strict, key=lambda item: (
            -item.provisional_failure_count,
            -len(item.provisional_failure_events - base_events),
            item.strategy_id,
        ))
    novel = [item for item in candidates if item.provisional_failure_events - base_events]
    if novel:
        return min(novel, key=lambda item: (
            -len(item.provisional_failure_events - base_events),
            -item.provisional_failure_count,
            item.strategy_id,
        ))
    return None


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
    """Injectable generation controller, configured as a two-generation smoke by default.

    Pilot callers may supply an explicit multi-generation mapping manifest and
    a larger E task panel. The checked-in Phase 3 ``MechanismManifest`` remains
    frozen to the original two-generation, one-E/one-V smoke.
    """

    def __init__(self, *, manifest: dict | MechanismManifest | PilotManifest, checkpoint_path: str,
                 runner: EpisodeRunner, task_ids: tuple[str, ...], seed: int,
                 request_budget: RequestBudget | None = None,
                 failure_archive: FailureArchive | None = None,
                 manifest_context: Mapping[str, Any] | None = None,
                 evolution_task_ids: tuple[str, ...] | None = None,
                 episode_seed_base: int | None = None,
                 generations: int | None = None,
                 max_episodes: int | None = None):
        if len(task_ids) != 2 or task_ids[0] == task_ids[1]:
            raise ValueError("controller requires distinct primary evolution and validation task IDs")
        self.manifest = manifest
        typed_manifest = isinstance(manifest, (MechanismManifest, PilotManifest))
        manifest_payload = manifest.to_payload() if typed_manifest else manifest
        if evolution_task_ids is None:
            evolution_task_ids = (task_ids[0],)
        if (not evolution_task_ids or len(set(evolution_task_ids)) != len(evolution_task_ids)
                or any(not isinstance(item, str) or not item.strip() for item in evolution_task_ids)
                or task_ids[1] in evolution_task_ids):
            raise ValueError("controller E task IDs must be unique and disjoint from the primary V task")
        configured_generations = (
            manifest.generations if typed_manifest
            else int(manifest.get("generations", 2))
        )
        self.generations = configured_generations if generations is None else generations
        if type(self.generations) is not int or not 2 <= self.generations <= 8:
            raise ValueError("controller supports a frozen generation count in the range 2..8")
        if self.generations != configured_generations:
            raise ValueError("controller generation count must match the frozen manifest")
        if task_ids[0] not in evolution_task_ids:
            raise ValueError("primary E task must belong to the frozen evolution task panel")
        if isinstance(manifest, MechanismManifest) and self.generations != 2:
            raise ValueError("the Phase 3 mechanism smoke remains frozen to exactly two generations")
        if isinstance(manifest, MechanismManifest) and evolution_task_ids != (manifest.evolution_task_id,):
            raise ValueError("the Phase 3 mechanism smoke remains frozen to one E task")
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
        if isinstance(manifest, PilotManifest):
            if (task_ids != (manifest.evolution_task_ids[0], manifest.validation_task_ids[0])
                    or evolution_task_ids != manifest.evolution_task_ids
                    or seed not in manifest.evolution_seeds):
                raise ValueError("Pilot controller seed and task panels must match the frozen manifest")
            if request_budget is not None and request_budget.snapshot().cap != manifest.request_budget_cap:
                raise ValueError("shared request budget cap must match the frozen Pilot manifest")
        self.checkpoint_path = checkpoint_path
        self.runner = runner
        self.task_ids = task_ids
        self.evolution_task_ids = evolution_task_ids
        self.seed = seed
        self.request_budget = request_budget
        self.failure_archive = failure_archive
        self.episode_seed_base = seed if episode_seed_base is None else episode_seed_base
        if type(self.episode_seed_base) is not int or self.episode_seed_base < 0:
            raise ValueError("episode seed base must be a non-negative integer")
        if isinstance(manifest, MechanismManifest) and self.episode_seed_base != seed:
            raise ValueError("Phase 3 episode seed base must equal its frozen seed")
        frozen_episode_cap = (
            manifest.max_episodes if typed_manifest else int(manifest.get("max_episodes", 23))
        )
        self.max_episodes = frozen_episode_cap if max_episodes is None else max_episodes
        if (type(self.max_episodes) is not int or not 0 <= self.max_episodes <= frozen_episode_cap):
            raise ValueError("controller episode limit must fit within the frozen manifest cap")
        configured_concurrency = (
            manifest.max_concurrency if typed_manifest
            else int(manifest.get("max_concurrency", 1))
        )
        if type(configured_concurrency) is not int or not 1 <= configured_concurrency <= 4:
            raise ValueError("controller max_concurrency must be in the range 1..4")
        self.max_concurrency = configured_concurrency
        # Phase 3's 23-episode ceiling includes the Phase 0 integration episode.
        self.episode_attempts = 1 if self.manifest_context is not None else 0
        self._episode_state_lock = Lock()
        self._reserved_episode_keys: set[str] = set()
        self._last_customer: CustomerStrategy | None = None
        self._last_service: ServiceStrategy | None = None
        self._commit_records: list[dict] = []
        self._episode_history: dict[str, EpisodeRecord] = {}
        self._progress: dict | None = None
        if (not isinstance(manifest, PilotManifest) and request_budget is not None
                and request_budget.snapshot().cap > 1800):
            raise ValueError("Phase 3 shared provider-attempt budget cannot exceed 1,800")

    def _run_episode(self, **kwargs) -> EpisodeRecord:
        cache_key = _runner_cache_key(**kwargs)
        with self._episode_state_lock:
            if self._progress is not None and cache_key in self._progress["episodes"]:
                return self._progress["episodes"][cache_key]
            scheduled = cache_key in self._reserved_episode_keys
            if scheduled:
                self._reserved_episode_keys.remove(cache_key)
            else:
                if self.episode_attempts >= self.max_episodes:
                    raise RuntimeError(f"episode cap {self.max_episodes} reached before dispatch")
                self.episode_attempts += 1
        if (not scheduled and self.request_budget is not None
                and self.request_budget.snapshot().remaining <= 0):
            with self._episode_state_lock:
                self.episode_attempts -= 1
            raise ProviderBudgetExceeded("generation stopped before episode: shared request budget exhausted")
        try:
            episode = self.runner(**kwargs)
            with self._episode_state_lock:
                self._episode_history[cache_key] = episode
                if self._progress is not None:
                    self._progress["episodes"][cache_key] = episode
            return episode
        except StopBeforeEpisodeDispatch:
            with self._episode_state_lock:
                self.episode_attempts -= 1
            raise
        finally:
            if not scheduled:
                self._persist_progress()

    def _reserve_episode_spec(self, spec: EpisodeSpec) -> bool:
        kwargs = spec.runner_kwargs()
        cache_key = _runner_cache_key(**kwargs)
        with self._episode_state_lock:
            if self._progress is not None and cache_key in self._progress["episodes"]:
                return True
            if cache_key in self._reserved_episode_keys:
                raise RuntimeError("episode batch repeats a task/seed/strategy/panel key")
            if self.episode_attempts >= self.max_episodes:
                return False
            self.episode_attempts += 1
            self._reserved_episode_keys.add(cache_key)
            return True

    def _release_episode_spec(self, spec: EpisodeSpec) -> None:
        cache_key = _runner_cache_key(**spec.runner_kwargs())
        with self._episode_state_lock:
            if cache_key in self._reserved_episode_keys:
                self._reserved_episode_keys.remove(cache_key)
                self.episode_attempts -= 1

    def _episode_needs_provider_request(self, spec: EpisodeSpec) -> bool:
        cache_key = _runner_cache_key(**spec.runner_kwargs())
        with self._episode_state_lock:
            if self._progress is not None and cache_key in self._progress["episodes"]:
                return False
        has_completed = getattr(self.runner, "has_completed_episode", None)
        return True if has_completed is None else not has_completed(**spec.runner_kwargs())

    def _should_pause(self) -> bool:
        signal = getattr(self.runner, "stop_before_next_episode_file", None)
        if signal is None:
            return False
        if signal.is_symlink():
            raise RuntimeError("pause signal path cannot be a symlink")
        return signal.exists()

    def _panel_batch_finished(self) -> None:
        with self._episode_state_lock:
            if self._reserved_episode_keys:
                raise RuntimeError("episode batch ended with unconsumed episode-cap reservations")
        self._persist_progress()

    def _serial_episode_finished(self, _spec: EpisodeSpec) -> None:
        if self.max_concurrency == 1:
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
            "episode_seed_base": self.episode_seed_base,
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
        if type(generation) is not int or not 0 <= generation < self.generations:
            raise ValueError(f"generation must be in the frozen range 0..{self.generations - 1}")
        # The controller accepts only immutable strategy snapshots after their
        # selection/gate decisions have completed.
        commit = GenerationCommit(generation, customer_strategy_id(customer),
                                  service_strategy_id(service), customer_evolved, service_evolved,
                                  True, note, decision_record)
        state = {
            "customer": customer.to_dict(), "service": service.to_dict(),
            "commits": [_commit_to_dict(commit)],
            "seed": self.seed,
            "episode_seed_base": self.episode_seed_base,
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
            customer_proposal_provider: OperatorSelector | None = None,
            customer_proposal_mode: str = "failure_conditioned",
            allow_frozen_service: bool = False,
            freeze_customer: bool = False,
            allow_strategy_revisit: bool = False,
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
        if customer_proposal_mode not in {"failure_conditioned", "random_mutation"}:
            raise ValueError("Customer proposal mode must be failure_conditioned or random_mutation")
        if type(allow_frozen_service) is not bool:
            raise TypeError("allow_frozen_service must be boolean")
        if type(freeze_customer) is not bool:
            raise TypeError("freeze_customer must be boolean")
        if type(allow_strategy_revisit) is not bool:
            raise TypeError("allow_strategy_revisit must be boolean")
        if isinstance(self.manifest, MechanismManifest):
            if sha256_json(customer.to_dict()) != self.manifest.customer_strategy_sha256:
                raise ValueError("initial Customer strategy does not match the frozen manifest")
            if sha256_json(service.to_dict()) != self.manifest.service_strategy_sha256:
                raise ValueError("initial Service strategy does not match the frozen manifest")
        if isinstance(self.manifest, PilotManifest):
            if sha256_json(customer.to_dict()) != self.manifest.customer_strategy_sha256:
                raise ValueError("initial Customer strategy does not match the frozen Pilot manifest")
            if sha256_json(service.to_dict()) != self.manifest.service_strategy_sha256:
                raise ValueError("initial Service strategy does not match the frozen Pilot manifest")
            expected_mode = (
                "random_mutation" if self.manifest.condition == "random_mutation"
                else "failure_conditioned"
            )
            if customer_proposal_mode != expected_mode:
                raise ValueError("Pilot condition and Customer proposal mode differ")
            expected_frozen_service = self.manifest.condition in {
                "random_mutation", "frozen_service", "adaptive_customer",
            }
            if allow_frozen_service != expected_frozen_service:
                raise ValueError("Pilot frozen-Service condition and Service transition differ")
            if freeze_customer != (self.manifest.condition == "frozen_customer"):
                raise ValueError("Pilot frozen-Customer condition and Customer update differ")
            if self.manifest.condition == "static_customer" and allow_strategy_revisit:
                raise ValueError("static_customer Pilot does not use adaptive proposal revisits")
        commits: list[GenerationCommit] = []
        start_generation = 0
        if self._last_customer is None:
            self._last_customer, self._last_service = customer, service
        path = Path(self.checkpoint_path)
        if path.exists():
            restored = load_checkpoint(path, expected_manifest_hash=self.manifest_hash)
            if restored.state.get("run_context") != self.manifest_context:
                raise ValueError("checkpoint Phase 0 run context does not match the current run")
            if restored.state.get("episode_seed_base", self.seed) != self.episode_seed_base:
                raise ValueError("checkpoint episode seed schedule differs from the current seed block")
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
        for generation in range(start_generation, self.generations):
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
            seed = self.episode_seed_base + generation
            panel_execution = {
                "max_concurrency": self.max_concurrency,
                "reserve_episode": self._reserve_episode_spec,
                "release_episode": self._release_episode_spec,
                "needs_provider_request": self._episode_needs_provider_request,
                "should_pause": self._should_pause,
                "episode_finished": (
                    self._serial_episode_finished if self.max_concurrency == 1 else None
                ),
                "batch_finished": self._panel_batch_finished,
            }
            if freeze_customer:
                round_result = run_frozen_customer_round(
                    self._run_episode, incumbent=customer, service=service,
                    task_ids=self.evolution_task_ids, seeds=(seed,), generation=generation,
                    verification_refs=verification_refs,
                    failure_verifier=failure_verifier,
                    confirmation_task_ids=self.evolution_task_ids,
                    confirmation_seeds=(self.episode_seed_base + 10_000 + generation,),
                    request_budget=self.request_budget,
                    **panel_execution,
                )
            else:
                round_result = run_customer_round(
                    self._run_episode, incumbent=customer, service=service,
                    task_ids=self.evolution_task_ids, seeds=(seed,), generation=generation,
                    proposal_seed=self.seed + generation * 1009, count=candidates_per_generation,
                    verification_refs=verification_refs,
                    failure_verifier=failure_verifier,
                    already_seen=self._progress["seen_strategy_ids"],
                    prior_failures=(
                        () if self.failure_archive is None
                        else self.failure_archive.active_representatives(current_generation=generation)
                    ),
                    confirmation_task_ids=self.evolution_task_ids,
                    confirmation_seeds=(self.episode_seed_base + 10_000 + generation,),
                    request_budget=self.request_budget,
                    proposal_provider=customer_proposal_provider,
                    proposal_mode=customer_proposal_mode,
                    allow_strategy_revisit=allow_strategy_revisit,
                    **panel_execution,
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
                    service_policy_text=getattr(self.runner, "service_policy_text", None),
                    trajectory_loader=getattr(self.runner, "load_trajectory", None),
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
                if not allow_frozen_service:
                    raise ValueError(
                        "verified Service failures require an audited ServiceTransition before generation commit"
                    )
                note = "verified failures are archived; frozen-Service control makes no repair"
                service_note = note
            decision_record = _generation_decision_record(
                generation=generation,
                old_customer=old_customer,
                new_customer=customer,
                old_service=old_service,
                new_service=service,
                round_result=round_result,
                failure_pool=tuple(failure_pool.values()),
                customer_proposal_mode=customer_proposal_mode,
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
    return budget.snapshot().to_dict()


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
    customer_proposal_mode: str,
    failure_pool: tuple[FailureRecord, ...],
    service_transition_ran: bool,
    service_gate: GateReport | None,
    service_note: str,
) -> dict[str, Any]:
    return {
        "generation": generation,
        "customer": {
            "proposal_mode": customer_proposal_mode,
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
                    "proposal_context_sha256": proposal.proposal_context_sha256,
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
