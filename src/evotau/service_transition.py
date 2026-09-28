"""Concrete, budget-gated Service repair transition for the Phase 3 controller.

Proposal and independent policy audit are injected as narrow providers. The
transition owns target selection, deterministic paired-panel execution,
repair-gate evaluation, and fail-closed handling of exhausted budgets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .budget import ProviderBudgetExceeded, RequestBudget
from .lifecycle import ServiceTransitionEpisodeRunner
from .records import EpisodeRecord, FailureRecord, service_strategy_id
from .service_evolution import (
    GateReport,
    GateUnit,
    RepairAudit,
    RepairProposal,
    build_repair_candidate,
    evaluate_repair_gate,
)
from .strategies import CustomerStrategy, ServiceStrategy

ProposalProvider = Callable[[int, FailureRecord, ServiceStrategy], RepairProposal]
RepairAuditProvider = Callable[[int, RepairProposal, FailureRecord, ServiceStrategy], RepairAudit]


@dataclass(frozen=True, slots=True)
class GatedServiceTransition:
    """Run one complete Service candidate gate or leave the incumbent unchanged.

    Providers receive the generation and immutable inputs. Their LLM use is
    counted by the controller's shared ``RequestBudget`` instrumentation.
    Gate panels use E for target/history/clean, V for adversarial validation,
    and a deterministic seed range disjoint from Customer discovery and
    confirmation panels.
    """

    evolution_task_id: str
    validation_task_id: str
    seed: int
    initial_service: ServiceStrategy
    proposal_provider: ProposalProvider
    audit_provider: RepairAuditProvider
    token_counter: Callable[[str], int]

    def __post_init__(self) -> None:
        if not self.evolution_task_id or not self.validation_task_id:
            raise ValueError("Service transition requires frozen E and V task IDs")
        if self.evolution_task_id == self.validation_task_id:
            raise ValueError("Service transition E and V task IDs must differ")
        if self.seed < 0:
            raise ValueError("Service transition seed must be non-negative")
        if not callable(self.proposal_provider) or not callable(self.audit_provider):
            raise TypeError("Service transition requires proposal and independent audit providers")
        if not callable(self.token_counter):
            raise TypeError("Service transition requires the frozen agent-model token counter")

    def __call__(
        self,
        generation: int,
        customer: CustomerStrategy,
        incumbent: ServiceStrategy,
        failures: tuple[FailureRecord, ...],
        episode_runner: ServiceTransitionEpisodeRunner,
        request_budget: RequestBudget | None,
    ) -> tuple[ServiceStrategy, GateReport | None, str]:
        if generation not in (0, 1):
            raise ValueError("Phase 3 Service transition supports generations 0 and 1")
        target, target_customer = self._select_target(failures, episode_runner, incumbent)
        if target is None or target_customer is None:
            return incumbent, None, "inconclusive: no E-task verified failure has a resolvable Customer snapshot"
        historical = self._select_historical(episode_runner, incumbent)
        if historical is None:
            return incumbent, None, "inconclusive: no valid incumbent-passing E-task replay is available"
        history_customer = episode_runner.resolve_customer_strategy(historical.customer_strategy_id)
        if history_customer is None:
            return incumbent, None, "inconclusive: historical replay Customer snapshot is unavailable"

        s0_id = service_strategy_id(self.initial_service)
        requires_separate_s0_anchor = service_strategy_id(incumbent) != s0_id
        required_episodes = 11 if requires_separate_s0_anchor else 10
        if episode_runner.remaining_episodes < required_episodes:
            return incumbent, None, (
                "inconclusive: complete Service gate requires "
                f"{required_episodes} episode slots; {episode_runner.remaining_episodes} remain"
            )
        if request_budget is not None:
            minimum_requests = required_episodes + 2  # one minimum attempt for each panel and provider
            remaining_requests = request_budget.snapshot().remaining
            if remaining_requests < minimum_requests:
                return incumbent, None, (
                    "inconclusive: complete Service gate requires at least "
                    f"{minimum_requests} provider attempts; {remaining_requests} remain"
                )

        try:
            proposal = self.proposal_provider(generation, target, incumbent)
            if not isinstance(proposal, RepairProposal):
                raise TypeError("proposal_provider must return RepairProposal")
            audit = self.audit_provider(generation, proposal, target, incumbent)
            if not isinstance(audit, RepairAudit):
                raise TypeError("audit_provider must return RepairAudit")
        except Exception as exc:
            if not _is_budget_exhaustion(exc):
                raise
            return incumbent, None, f"inconclusive: request budget exhausted before gate execution ({type(exc).__name__})"

        try:
            candidate = build_repair_candidate(
                incumbent, proposal, target, audit, token_counter=self.token_counter,
            )
        except ValueError:
            # Preserve the failed static review in an auditable report without
            # dispatching episodes or exposing an unapproved Service strategy.
            report = evaluate_repair_gate(
                incumbent, incumbent, (), target_failure=target, proposal=proposal,
                audit=audit, initial_service_strategy_id=s0_id,
                token_counter=self.token_counter,
            )
            return incumbent, report, "rejected: static policy audit or repair constraints failed"
        if candidate == incumbent:
            report = evaluate_repair_gate(
                incumbent, candidate, (), target_failure=target, proposal=proposal,
                audit=audit, initial_service_strategy_id=s0_id,
                token_counter=self.token_counter,
            )
            return incumbent, report, "rejected: repair does not change the incumbent Service strategy"

        if request_budget is not None:
            remaining_requests = request_budget.snapshot().remaining
            if remaining_requests < required_episodes:
                report = evaluate_repair_gate(
                    incumbent, candidate, (), target_failure=target, proposal=proposal,
                    audit=audit, initial_service_strategy_id=s0_id,
                    token_counter=self.token_counter, inconclusive=True,
                )
                return incumbent, report, (
                    "inconclusive: proposal and audit left fewer provider attempts than the "
                    f"{required_episodes}-episode gate minimum; no gate episode was dispatched"
                )

        units: list[GateUnit] = []
        stopped_for_budget: Exception | None = None
        base_seed = self.seed + 20_000 + generation * 16

        def run_pair(
            *, key: str, panel: str, task_id: str, seed: int,
            panel_name: str, fixed_customer: CustomerStrategy | None,
            target_failure_id: str | None = None,
        ) -> GateUnit:
            old = episode_runner(
                task_id=task_id, seed=seed, customer=fixed_customer,
                service=incumbent, panel_name=f"service-g{generation}-{panel_name}-incumbent",
            )
            new = episode_runner(
                task_id=task_id, seed=seed, customer=fixed_customer,
                service=candidate, panel_name=f"service-g{generation}-{panel_name}-candidate",
            )
            return GateUnit(key, panel, old, new, target_failure_id=target_failure_id)

        for index, trial_seed in enumerate((base_seed, base_seed + 1), start=1):
            try:
                units.append(run_pair(
                    key=f"target-{index}", panel="target", task_id=target.task_id,
                    seed=trial_seed, panel_name=f"target-{index}",
                    fixed_customer=target_customer, target_failure_id=target.failure_id,
                ))
            except Exception as exc:
                if not _is_budget_exhaustion(exc):
                    raise
                stopped_for_budget = exc
                break
        if stopped_for_budget is None:
            try:
                units.append(run_pair(
                    key="historical-1", panel="historical", task_id=historical.task_id,
                    seed=base_seed + 2, panel_name="historical-1",
                    fixed_customer=history_customer,
                ))
                clean_old = episode_runner(
                    task_id=self.evolution_task_id, seed=base_seed + 3, customer=None,
                    service=incumbent, panel_name=f"service-g{generation}-clean-incumbent",
                )
                clean_new = episode_runner(
                    task_id=self.evolution_task_id, seed=base_seed + 3, customer=None,
                    service=candidate, panel_name=f"service-g{generation}-clean-candidate",
                )
                if requires_separate_s0_anchor:
                    initial_s0 = episode_runner(
                        task_id=self.evolution_task_id, seed=base_seed + 3, customer=None,
                        service=self.initial_service, panel_name=f"service-g{generation}-clean-initial-s0",
                    )
                else:
                    initial_s0 = clean_old
                units.append(GateUnit(
                    "clean-1", "clean", clean_old, clean_new, initial_s0=initial_s0,
                ))
                units.append(run_pair(
                    key="validation-1", panel="validation", task_id=self.validation_task_id,
                    seed=base_seed + 4, panel_name="validation-1",
                    fixed_customer=customer,
                ))
            except Exception as exc:
                if not _is_budget_exhaustion(exc):
                    raise
                stopped_for_budget = exc

        report = evaluate_repair_gate(
            incumbent, candidate, tuple(units), target_failure=target,
            proposal=proposal, audit=audit, initial_service_strategy_id=s0_id,
            token_counter=self.token_counter, inconclusive=stopped_for_budget is not None,
        )
        if report.accepted:
            return candidate, report, "accepted: complete target/history/clean/validation gate passed"
        if stopped_for_budget is not None:
            return incumbent, report, (
                "inconclusive: request or episode budget exhausted during the paired gate; "
                "incumbent retained"
            )
        return incumbent, report, "rejected: candidate failed one or more prespecified Service gates"

    def _select_target(
        self,
        failures: tuple[FailureRecord, ...],
        runner: ServiceTransitionEpisodeRunner,
        incumbent: ServiceStrategy,
    ) -> tuple[FailureRecord | None, CustomerStrategy | None]:
        severity_rank = {"critical": 0, "high": 1, "material": 2, "low": 3}
        eligible = sorted(
            (item for item in failures if item.task_id == self.evolution_task_id),
            key=lambda item: (
                -item.generation, severity_rank.get(item.severity.lower(), 4), item.failure_id,
            ),
        )
        for failure in eligible:
            strategy = runner.resolve_customer_strategy(failure.customer_strategy_id)
            if strategy is None:
                continue
            source = runner.find_failure_episode(failure)
            if source is not None and (
                source.task_id != failure.task_id
                or source.customer_strategy_id != failure.customer_strategy_id
                or source.service_strategy_id != failure.service_strategy_id
                or not source.has_attributable_failure_candidate
            ):
                continue
            return failure, strategy
        return None, None

    def _select_historical(
        self,
        runner: ServiceTransitionEpisodeRunner,
        incumbent: ServiceStrategy,
    ) -> EpisodeRecord | None:
        matches = runner.passing_history(
            task_id=self.evolution_task_id,
            service_id=service_strategy_id(incumbent),
        )
        return next((
            item for item in matches
            if runner.resolve_customer_strategy(item.customer_strategy_id) is not None
        ), None)


def _is_budget_exhaustion(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ProviderBudgetExceeded):
            return True
        if isinstance(current, RuntimeError) and any(
            marker in str(current).lower()
            for marker in ("episode cap", "episode capacity", "request budget", "budget exhausted")
        ):
            return True
        current = current.__cause__
    return False
