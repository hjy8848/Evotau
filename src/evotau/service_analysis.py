"""Descriptive RQ2 metrics for paired Service cross-play panels."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .crossplay import CrossPlayCell, CrossPlayMatrix
from .manifest import sha256_json
from .records import FailureRecord, customer_strategy_id
from .service_evolution import GateReport

SERVICE_PANEL_SCOPES = frozenset({"target", "historical", "clean", "validation", "heldout"})


@dataclass(frozen=True, slots=True)
class CustomerRepairEffect:
    customer_strategy_id: str
    incumbent_valid_episodes: int
    candidate_valid_episodes: int
    incumbent_adherent_episodes: int
    candidate_adherent_episodes: int
    incumbent_success_rate: float | None
    candidate_success_rate: float | None
    success_rate_change: float | None
    incumbent_policy_violation_rate: float | None
    candidate_policy_violation_rate: float | None
    policy_violation_rate_change: float | None
    invalid_repeated_write_call_rate_change: float | None
    incumbent_repeated_write_audit_coverage: float | None
    candidate_repeated_write_audit_coverage: float | None
    incumbent_attributable_failure_rate: float | None
    candidate_attributable_failure_rate: float | None
    attributable_failure_rate_change: float | None
    candidate_historical_recurrence_episodes: int
    candidate_historical_recurrence_rate: float | None
    candidate_repaired_signature_coverage: float | None


@dataclass(frozen=True, slots=True)
class ServiceRepairAnalysis:
    panel_scope: str
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    request_budget_cap: int
    provider_attempts: int
    incumbent_service_strategy_id: str
    candidate_service_strategy_id: str
    incumbent_matrix_sha256: str
    candidate_matrix_sha256: str
    gate_report_sha256: tuple[str, ...]
    target_failure_sha256: tuple[str, ...]
    proposed_repairs: int
    accepted_repairs: int
    rejected_repairs: int
    inconclusive_repairs: int
    repair_acceptance_rate: float | None
    task_success_rate_change: float | None
    policy_violation_rate_change: float | None
    invalid_repeated_write_call_rate_change: float | None
    incumbent_repeated_write_audit_coverage: float | None
    candidate_repeated_write_audit_coverage: float | None
    attributable_failure_rate_change: float | None
    target_failure_rate_reduction: float | None
    accepted_target_signature_keys: tuple[str, ...]
    accepted_target_signature_count: int
    clean_success_rate_change: float | None
    clean_policy_violation_rate_change: float | None
    incumbent_historical_recurrence_rate: float | None
    historical_recurrence_rate: float | None
    historical_recurrence_rate_change: float | None
    incumbent_historical_recurrence_episodes: int
    historical_recurrence_episodes: int
    incumbent_historical_adherent_episodes: int
    historical_adherent_episodes: int
    incumbent_repaired_signature_coverage: float | None
    repaired_signature_coverage: float | None
    repaired_signature_count: int
    repaired_signature_keys: tuple[str, ...]
    customer_effects: tuple[CustomerRepairEffect, ...]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_service_repair_crossplay(
    incumbent: CrossPlayMatrix,
    candidate: CrossPlayMatrix,
    *,
    panel_scope: str,
    gate_reports: Sequence[GateReport],
    target_failures: Sequence[FailureRecord],
    request_budget_cap: int,
    provider_attempts: int,
    repaired_signature_keys: Sequence[str] = (),
) -> ServiceRepairAnalysis:
    """Compare one frozen Service checkpoint with its proposed replacement.

    Both matrices must use the same complete task/seed/Customer panel. The
    candidate matrix must carry the exact set of previously repaired failure
    signature keys used to calculate historical recurrence. Per-cell rates
    retain their valid/adherent denominators. This is a panel-level descriptive
    analysis; Formal inference must aggregate over independent evolution runs.
    """

    if panel_scope not in SERVICE_PANEL_SCOPES:
        raise ValueError(f"panel_scope must be one of {sorted(SERVICE_PANEL_SCOPES)}")
    if type(request_budget_cap) is not int or request_budget_cap <= 0:
        raise ValueError("request_budget_cap must be a positive integer")
    if (type(provider_attempts) is not int
            or not 0 <= provider_attempts <= request_budget_cap):
        raise ValueError("provider_attempts must be within the frozen request budget")
    if incumbent.task_ids != candidate.task_ids or incumbent.seeds != candidate.seeds:
        raise ValueError("incumbent and candidate matrices must use the same task/seed panel")
    if incumbent.customer_strategy_ids != candidate.customer_strategy_ids:
        raise ValueError("incumbent and candidate matrices must use the same frozen Customer panel")
    if len(incumbent.service_strategy_ids) != 1 or len(candidate.service_strategy_ids) != 1:
        raise ValueError("RQ2 repair analysis compares exactly one incumbent and one candidate Service")
    old_service_id = incumbent.service_strategy_ids[0]
    new_service_id = candidate.service_strategy_ids[0]
    if old_service_id == new_service_id:
        raise ValueError("candidate Service strategy must differ from the incumbent")
    repaired_keys = tuple(sorted(repaired_signature_keys))
    if (len(set(repaired_keys)) != len(repaired_keys)
            or any(not re.fullmatch(r"[0-9a-f]{16}", item) for item in repaired_keys)):
        raise ValueError("repaired signature keys must be unique 16-character lowercase hashes")
    if incumbent.repaired_signature_keys != repaired_keys:
        raise ValueError("incumbent matrix historical-signature set differs from the analysis input")
    if candidate.repaired_signature_keys != repaired_keys:
        raise ValueError("candidate matrix historical-signature set differs from the analysis input")

    failures_by_id = {failure.failure_id: failure for failure in target_failures}
    if len(failures_by_id) != len(target_failures):
        raise ValueError("target FailureRecord IDs must be unique")
    if any(not isinstance(report, GateReport) for report in gate_reports):
        raise TypeError("gate_reports must contain GateReport records")
    if any(report.target_failure_id not in failures_by_id for report in gate_reports):
        raise ValueError("every repair gate must reference a supplied verified target FailureRecord")
    for report in gate_reports:
        if report.proposal.target_failure_id != report.target_failure_id:
            raise ValueError("repair gate proposal and target failure references differ")
        if report.accepted and (
            report.candidate_strategy is None
            or report.candidate_strategy != report.evaluated_candidate
        ):
            raise ValueError("accepted repair gate does not identify its exact evaluated candidate")

    old_by_customer = _cells_by_customer(incumbent)
    new_by_customer = _cells_by_customer(candidate)
    clean_id = customer_strategy_id(None)
    if clean_id not in old_by_customer:
        raise ValueError("RQ2 cross-play matrices must include the native clean Customer")
    clean_old = old_by_customer[clean_id]
    clean_new = new_by_customer[clean_id]
    for cell in (clean_old, clean_new):
        if (cell.strategy_opportunities != 0
                or cell.strategy_not_applicable_episodes != cell.attempted_episodes
                or cell.valid_episodes != cell.attempted_episodes
                or cell.invalid_episodes or cell.infrastructure_episodes or cell.uncertain_episodes):
            raise ValueError("clean Customer cells must be fully audited native/no-overlay episodes")

    effects = tuple(
        _customer_effect(
            old_by_customer[customer_id],
            new_by_customer[customer_id],
            repaired_keys=repaired_keys,
            panel_scope=panel_scope,
        )
        for customer_id in incumbent.customer_strategy_ids
    )
    adversarial_ids = tuple(
        customer_id for customer_id in incumbent.customer_strategy_ids if customer_id != clean_id
    )
    adv_old = tuple(old_by_customer[item] for item in adversarial_ids)
    adv_new = tuple(new_by_customer[item] for item in adversarial_ids)
    target_reduction = None
    accepted_target_keys = {
        failures_by_id[report.target_failure_id].signature.key
        for report in gate_reports if report.accepted
    }
    if panel_scope == "target" and accepted_target_keys:
        old_rate = _weighted_rate(
            sum(_signature_episode_count(cell, accepted_target_keys) for cell in adv_old),
            sum(cell.strategy_adherent_episodes for cell in adv_old),
        )
        new_rate = _weighted_rate(
            sum(_signature_episode_count(cell, accepted_target_keys) for cell in adv_new),
            sum(cell.strategy_adherent_episodes for cell in adv_new),
        )
        if old_rate is not None and new_rate is not None:
            target_reduction = old_rate - new_rate

    recurrent_episodes = sum(cell.recurrent_verified_failure_episodes for cell in adv_new)
    historical_denominator = sum(cell.strategy_adherent_episodes for cell in adv_new)
    incumbent_recurrent_episodes = sum(cell.recurrent_verified_failure_episodes for cell in adv_old)
    incumbent_historical_denominator = sum(cell.strategy_adherent_episodes for cell in adv_old)
    recurrence_rate = None
    incumbent_recurrence_rate = None
    coverage = None
    incumbent_coverage = None
    if panel_scope == "historical":
        recurrence_rate = _weighted_rate(recurrent_episodes, historical_denominator)
        incumbent_recurrence_rate = _weighted_rate(
            incumbent_recurrent_episodes, incumbent_historical_denominator,
        )
        observed_repaired_signatures = {
            key for cell in adv_new for key in cell.recurrent_signature_keys
        }
        incumbent_observed_repaired_signatures = {
            key for cell in adv_old for key in cell.recurrent_signature_keys
        }
        coverage = (
            len(observed_repaired_signatures) / len(repaired_keys) if repaired_keys else None
        )
        incumbent_coverage = (
            len(incumbent_observed_repaired_signatures) / len(repaired_keys)
            if repaired_keys else None
        )

    accepted = sum(report.accepted for report in gate_reports)
    proposed = len(gate_reports)
    inconclusive = sum(report.inconclusive for report in gate_reports)
    all_old = tuple(old_by_customer.values())
    all_new = tuple(new_by_customer.values())
    total_old_valid = sum(cell.valid_episodes for cell in all_old)
    total_new_valid = sum(cell.valid_episodes for cell in all_new)
    total_old_adherent = sum(cell.strategy_adherent_episodes for cell in all_old)
    total_new_adherent = sum(cell.strategy_adherent_episodes for cell in all_new)
    old_failure_rate = _weighted_rate(
        sum(cell.verified_failure_episodes for cell in all_old), total_old_adherent,
    )
    new_failure_rate = _weighted_rate(
        sum(cell.verified_failure_episodes for cell in all_new), total_new_adherent,
    )
    old_policy_rate = _weighted_rate(
        sum(cell.policy_violation_episodes for cell in all_old), total_old_valid,
    )
    new_policy_rate = _weighted_rate(
        sum(cell.policy_violation_episodes for cell in all_new), total_new_valid,
    )
    old_repeat_audited = sum(cell.repeated_write_audited_episodes for cell in all_old)
    new_repeat_audited = sum(cell.repeated_write_audited_episodes for cell in all_new)
    old_repeat_coverage = _rate(old_repeat_audited, total_old_valid)
    new_repeat_coverage = _rate(new_repeat_audited, total_new_valid)
    old_repeat_rate = _rate(
        sum(cell.invalid_repeated_write_calls for cell in all_old), old_repeat_audited,
    )
    new_repeat_rate = _rate(
        sum(cell.invalid_repeated_write_calls for cell in all_new), new_repeat_audited,
    )
    repeat_rate_change = (
        _difference(old_repeat_rate, new_repeat_rate)
        if old_repeat_coverage == 1.0 and new_repeat_coverage == 1.0 else None
    )
    old_success_rate = _weighted_rate(sum(cell.successful_episodes for cell in all_old), total_old_valid)
    new_success_rate = _weighted_rate(sum(cell.successful_episodes for cell in all_new), total_new_valid)
    old_clean_success = _rate(clean_old.successful_episodes, clean_old.valid_episodes)
    new_clean_success = _rate(clean_new.successful_episodes, clean_new.valid_episodes)
    old_clean_violation = _rate(clean_old.policy_violation_episodes, clean_old.valid_episodes)
    new_clean_violation = _rate(clean_new.policy_violation_episodes, clean_new.valid_episodes)
    interpretation = (
        "descriptive paired-panel metrics only; formal RQ2 claims require multiple independent evolution runs"
    )

    return ServiceRepairAnalysis(
        panel_scope=panel_scope,
        task_ids=incumbent.task_ids,
        seeds=incumbent.seeds,
        request_budget_cap=request_budget_cap,
        provider_attempts=provider_attempts,
        incumbent_service_strategy_id=old_service_id,
        candidate_service_strategy_id=new_service_id,
        incumbent_matrix_sha256=sha256_json(incumbent.to_dict()),
        candidate_matrix_sha256=sha256_json(candidate.to_dict()),
        gate_report_sha256=tuple(sha256_json(report.to_dict()) for report in gate_reports),
        target_failure_sha256=tuple(
            sha256_json(failure.to_dict()) for failure in target_failures
        ),
        proposed_repairs=proposed,
        accepted_repairs=accepted,
        rejected_repairs=proposed - accepted - inconclusive,
        inconclusive_repairs=inconclusive,
        repair_acceptance_rate=(
            None if inconclusive else (accepted / proposed if proposed else None)
        ),
        task_success_rate_change=_difference(old_success_rate, new_success_rate),
        policy_violation_rate_change=_difference(old_policy_rate, new_policy_rate),
        invalid_repeated_write_call_rate_change=repeat_rate_change,
        incumbent_repeated_write_audit_coverage=old_repeat_coverage,
        candidate_repeated_write_audit_coverage=new_repeat_coverage,
        attributable_failure_rate_change=_difference(old_failure_rate, new_failure_rate),
        target_failure_rate_reduction=target_reduction,
        accepted_target_signature_keys=tuple(sorted(accepted_target_keys)),
        accepted_target_signature_count=len(accepted_target_keys),
        clean_success_rate_change=_difference(old_clean_success, new_clean_success),
        clean_policy_violation_rate_change=_difference(old_clean_violation, new_clean_violation),
        incumbent_historical_recurrence_rate=incumbent_recurrence_rate,
        historical_recurrence_rate=recurrence_rate,
        historical_recurrence_rate_change=_difference(incumbent_recurrence_rate, recurrence_rate),
        incumbent_historical_recurrence_episodes=(
            incumbent_recurrent_episodes if panel_scope == "historical" else 0
        ),
        historical_recurrence_episodes=(recurrent_episodes if panel_scope == "historical" else 0),
        incumbent_historical_adherent_episodes=(
            incumbent_historical_denominator if panel_scope == "historical" else 0
        ),
        historical_adherent_episodes=historical_denominator if panel_scope == "historical" else 0,
        incumbent_repaired_signature_coverage=incumbent_coverage,
        repaired_signature_coverage=coverage,
        repaired_signature_count=len(repaired_keys),
        repaired_signature_keys=repaired_keys,
        customer_effects=effects,
        interpretation=interpretation,
    )


def _cells_by_customer(matrix: CrossPlayMatrix) -> dict[str, CrossPlayCell]:
    service_id = matrix.service_strategy_ids[0]
    selected = {
        cell.customer_strategy_id: cell
        for cell in matrix.cells
        if cell.service_strategy_id == service_id
    }
    if set(selected) != set(matrix.customer_strategy_ids):
        raise ValueError("cross-play matrix is missing a Customer/Service cell")
    return selected


def _customer_effect(
    incumbent: CrossPlayCell,
    candidate: CrossPlayCell,
    *,
    repaired_keys: tuple[str, ...],
    panel_scope: str,
) -> CustomerRepairEffect:
    old_failure_rate = incumbent.verified_failure_rate
    new_failure_rate = candidate.verified_failure_rate
    recurrence = (
        candidate.recurrent_verified_failure_rate if panel_scope == "historical" else None
    )
    signature_coverage = None
    if panel_scope == "historical" and repaired_keys:
        signature_coverage = len(set(candidate.recurrent_signature_keys)) / len(repaired_keys)
    old_repeat_rate = (
        _rate(incumbent.invalid_repeated_write_calls, incumbent.repeated_write_audited_episodes)
        if incumbent.repeated_write_audit_coverage == 1.0 else None
    )
    new_repeat_rate = (
        _rate(candidate.invalid_repeated_write_calls, candidate.repeated_write_audited_episodes)
        if candidate.repeated_write_audit_coverage == 1.0 else None
    )
    return CustomerRepairEffect(
        customer_strategy_id=incumbent.customer_strategy_id,
        incumbent_valid_episodes=incumbent.valid_episodes,
        candidate_valid_episodes=candidate.valid_episodes,
        incumbent_adherent_episodes=incumbent.strategy_adherent_episodes,
        candidate_adherent_episodes=candidate.strategy_adherent_episodes,
        incumbent_success_rate=incumbent.native_success_rate,
        candidate_success_rate=candidate.native_success_rate,
        success_rate_change=_difference(incumbent.native_success_rate, candidate.native_success_rate),
        incumbent_policy_violation_rate=incumbent.policy_violation_rate,
        candidate_policy_violation_rate=candidate.policy_violation_rate,
        policy_violation_rate_change=_difference(
            incumbent.policy_violation_rate, candidate.policy_violation_rate,
        ),
        invalid_repeated_write_call_rate_change=_difference(old_repeat_rate, new_repeat_rate),
        incumbent_repeated_write_audit_coverage=incumbent.repeated_write_audit_coverage,
        candidate_repeated_write_audit_coverage=candidate.repeated_write_audit_coverage,
        incumbent_attributable_failure_rate=old_failure_rate,
        candidate_attributable_failure_rate=new_failure_rate,
        attributable_failure_rate_change=_difference(old_failure_rate, new_failure_rate),
        candidate_historical_recurrence_episodes=(
            candidate.recurrent_verified_failure_episodes if panel_scope == "historical" else 0
        ),
        candidate_historical_recurrence_rate=recurrence,
        candidate_repaired_signature_coverage=signature_coverage,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _weighted_rate(numerator: int, denominator: int) -> float | None:
    return _rate(numerator, denominator)


def _difference(old: float | None, new: float | None) -> float | None:
    return new - old if old is not None and new is not None else None


def _signature_episode_count(cell: CrossPlayCell, signature_keys: set[str]) -> int:
    return sum(count for key, count in cell.verified_signature_episode_counts if key in signature_keys)


@dataclass(frozen=True, slots=True)
class ServiceRobustnessRun:
    """The five paired Service panels for one independent evolution run."""

    run_id: str
    evolution_seed: int
    panels: tuple[ServiceRepairAnalysis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("RQ2 run_id must be a non-empty string")
        if type(self.evolution_seed) is not int or self.evolution_seed < 0:
            raise ValueError("RQ2 evolution seed must be a non-negative integer")
        if any(not isinstance(panel, ServiceRepairAnalysis) for panel in self.panels):
            raise TypeError("RQ2 run panels must be ServiceRepairAnalysis records")
        scopes = [panel.panel_scope for panel in self.panels]
        if len(scopes) != len(set(scopes)) or set(scopes) != SERVICE_PANEL_SCOPES:
            raise ValueError("each RQ2 evolution run must include target/history/clean/validation/heldout once")
        budget_caps = {panel.request_budget_cap for panel in self.panels}
        provider_attempts = {panel.provider_attempts for panel in self.panels}
        if len(budget_caps) != 1 or len(provider_attempts) != 1:
            raise ValueError("all RQ2 panels in one evolution run must share its request budget snapshot")
        cap = next(iter(budget_caps))
        attempts = next(iter(provider_attempts))
        if type(cap) is not int or cap <= 0 or type(attempts) is not int or not 0 <= attempts <= cap:
            raise ValueError("RQ2 run request-attempt snapshot is outside its positive frozen budget")
        service_pairs = {
            (panel.incumbent_service_strategy_id, panel.candidate_service_strategy_id)
            for panel in self.panels
        }
        if len(service_pairs) != 1:
            raise ValueError("all RQ2 panels in a run must compare the same Service checkpoint pair")

    @property
    def panels_by_scope(self) -> dict[str, ServiceRepairAnalysis]:
        return {panel.panel_scope: panel for panel in self.panels}

    @property
    def input_sha256(self) -> str:
        return sha256_json({
            "run_id": self.run_id,
            "evolution_seed": self.evolution_seed,
            "panels": [panel.to_dict() for panel in sorted(self.panels, key=lambda item: item.panel_scope)],
        })


@dataclass(frozen=True, slots=True)
class RQ2MetricEstimate:
    metric: str
    interpretation_direction: str
    status: str
    independent_runs: int
    complete_run_values: int
    observations: tuple[tuple[str, float | None], ...]
    mean: float | None
    median: float | None
    bootstrap_95_percentile_interval: tuple[float, float] | None
    bootstrap_replicates: int
    directionally_favorable_runs: int | None


@dataclass(frozen=True, slots=True)
class RQ2RobustnessReport:
    status: str
    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    request_budget_cap: int
    minimum_independent_runs: int
    bootstrap_seed: int
    run_ids_and_input_sha256: tuple[tuple[str, int, str], ...]
    actual_provider_attempts: tuple[tuple[str, int], ...]
    metrics: tuple[RQ2MetricEstimate, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_rq2_service_robustness(
    runs: Sequence[ServiceRobustnessRun],
    *,
    evolution_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
    heldout_task_ids: Sequence[str],
    minimum_independent_runs: int = 3,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> RQ2RobustnessReport:
    """Aggregate RQ2 panels over independent evolution runs.

    The five panel scopes and their task partitions are frozen across runs.
    Paired changes are first calculated within each run; bootstrap resampling
    then uses only independent evolution runs. Episode-level rows never act
    as independent statistical units. All intervals remain descriptive until
    a pilot-informed, pre-registered formal analysis is supplied.
    """

    e_tasks = _task_tuple(evolution_task_ids, "evolution")
    v_tasks = _task_tuple(validation_task_ids, "validation")
    h_tasks = _task_tuple(heldout_task_ids, "heldout")
    if not e_tasks or not v_tasks or not h_tasks:
        raise ValueError("RQ2 E, V, and H task selections must each be non-empty")
    all_selected = (*e_tasks, *v_tasks, *h_tasks)
    if len(set(all_selected)) != len(all_selected):
        raise ValueError("RQ2 E, V, and H task selections must be unique and disjoint")
    if type(minimum_independent_runs) is not int or minimum_independent_runs < 2:
        raise ValueError("minimum_independent_runs must be at least two")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not runs:
        raise ValueError("RQ2 analysis requires independent evolution runs")

    run_ids: set[str] = set()
    evolution_seeds: set[int] = set()
    for run in runs:
        if run.run_id in run_ids:
            raise ValueError("RQ2 run IDs must be unique")
        if run.evolution_seed in evolution_seeds:
            raise ValueError("RQ2 independent evolution runs must use distinct seeds")
        run_ids.add(run.run_id)
        evolution_seeds.add(run.evolution_seed)
        panels = run.panels_by_scope
        if not set(panels["target"].task_ids) <= set(e_tasks):
            raise ValueError("RQ2 target panel tasks must belong to the frozen E partition")
        if not set(panels["historical"].task_ids) <= set(e_tasks):
            raise ValueError("RQ2 historical replay tasks must belong to the frozen E partition")
        if not set(panels["clean"].task_ids) <= set(v_tasks):
            raise ValueError("RQ2 clean control tasks must belong to the frozen V partition")
        if not set(panels["validation"].task_ids) <= set(v_tasks):
            raise ValueError("RQ2 adversarial validation tasks must belong to the frozen V partition")
        if panels["heldout"].task_ids != h_tasks:
            raise ValueError("RQ2 heldout panel must equal the sealed ordered H partition")

    reference = runs[0].panels_by_scope
    request_budget_cap = reference["target"].request_budget_cap
    for run in runs[1:]:
        panels = run.panels_by_scope
        for scope in SERVICE_PANEL_SCOPES:
            if panels[scope].task_ids != reference[scope].task_ids:
                raise ValueError(f"RQ2 {scope} task panel must be identical across independent runs")
        if panels["target"].request_budget_cap != request_budget_cap:
            raise ValueError("RQ2 independent runs must share one request-attempt budget cap")

    specs = (
        ("target_failure_rate_reduction", "target", "target_failure_rate_reduction", "higher"),
        ("repair_acceptance_rate", "target", "repair_acceptance_rate", "descriptive"),
        ("clean_success_rate_change", "clean", "clean_success_rate_change", "higher"),
        ("clean_policy_violation_rate_change", "clean", "clean_policy_violation_rate_change", "lower"),
        ("validation_success_rate_change", "validation", "task_success_rate_change", "higher"),
        ("validation_policy_violation_rate_change", "validation", "policy_violation_rate_change", "lower"),
        ("clean_invalid_repeated_write_call_rate_change", "clean",
         "invalid_repeated_write_call_rate_change", "lower"),
        ("validation_attributable_failure_rate_change", "validation", "attributable_failure_rate_change", "lower"),
        ("heldout_success_rate_change", "heldout", "task_success_rate_change", "higher"),
        ("heldout_policy_violation_rate_change", "heldout", "policy_violation_rate_change", "lower"),
        ("heldout_attributable_failure_rate_change", "heldout", "attributable_failure_rate_change", "lower"),
        ("historical_recurrence_rate", "historical", "historical_recurrence_rate", "lower"),
        ("historical_recurrence_rate_change", "historical", "historical_recurrence_rate_change", "lower"),
    )
    rng = random.Random(bootstrap_seed)
    estimates = tuple(
        _estimate_rq2_metric(
            name=name,
            scope=scope,
            attribute=attribute,
            direction=direction,
            runs=runs,
            minimum_runs=minimum_independent_runs,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        for name, scope, attribute, direction in specs
    )
    reasons: list[str] = []
    if len(runs) < minimum_independent_runs:
        reasons.append(
            f"only {len(runs)} independent evolution runs; {minimum_independent_runs} required for pilot-level comparison"
        )
    if any(metric.status == "incomplete_denominator" for metric in estimates):
        reasons.append("one or more panel metrics lack a valid denominator in at least one independent run")
    status = (
        "insufficient_independent_runs" if len(runs) < minimum_independent_runs
        else "incomplete_denominators" if reasons
        else "descriptive"
    )
    return RQ2RobustnessReport(
        status=status,
        evolution_task_ids=e_tasks,
        validation_task_ids=v_tasks,
        heldout_task_ids=h_tasks,
        request_budget_cap=request_budget_cap,
        minimum_independent_runs=minimum_independent_runs,
        bootstrap_seed=bootstrap_seed,
        run_ids_and_input_sha256=tuple(
            (run.run_id, run.evolution_seed, run.input_sha256) for run in runs
        ),
        actual_provider_attempts=tuple(
            (run.run_id, run.panels_by_scope["target"].provider_attempts) for run in runs
        ),
        metrics=estimates,
        reasons=tuple(reasons),
    )


def _estimate_rq2_metric(
    *,
    name: str,
    scope: str,
    attribute: str,
    direction: str,
    runs: Sequence[ServiceRobustnessRun],
    minimum_runs: int,
    bootstrap_replicates: int,
    rng: random.Random,
) -> RQ2MetricEstimate:
    observations = tuple(
        (run.run_id, getattr(run.panels_by_scope[scope], attribute)) for run in runs
    )
    values = [value for _, value in observations if value is not None]
    if len(runs) < minimum_runs:
        status = "insufficient_independent_runs"
    elif len(values) != len(runs):
        status = "incomplete_denominator"
    else:
        status = "descriptive"
    interval = None
    metric_mean = metric_median = None
    favorable_runs = None
    used_replicates = 0
    if len(values) == len(runs) and values:
        metric_mean = mean(values)
        metric_median = _percentile(values, 0.5)
        favorable_runs = (
            sum(value >= 0 for value in values) if direction == "higher"
            else sum(value <= 0 for value in values) if direction == "lower"
            else None
        )
        if len(runs) >= minimum_runs:
            bootstrap_means = [
                mean(rng.choice(values) for _ in values)
                for _ in range(bootstrap_replicates)
            ]
            interval = (_percentile(bootstrap_means, 0.025), _percentile(bootstrap_means, 0.975))
            used_replicates = bootstrap_replicates
    return RQ2MetricEstimate(
        metric=name,
        interpretation_direction=direction,
        status=status,
        independent_runs=len(runs),
        complete_run_values=len(values),
        observations=observations,
        mean=metric_mean,
        median=metric_median,
        bootstrap_95_percentile_interval=interval,
        bootstrap_replicates=used_replicates,
        directionally_favorable_runs=favorable_runs,
    )


def _task_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    tasks = tuple(values)
    if any(not isinstance(item, str) or not item for item in tasks):
        raise ValueError(f"{name} task IDs must be non-empty strings")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"{name} task IDs must be unique")
    return tasks


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
