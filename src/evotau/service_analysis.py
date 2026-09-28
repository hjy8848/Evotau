"""Descriptive RQ2 metrics for paired Service cross-play panels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    incumbent_service_strategy_id: str
    candidate_service_strategy_id: str
    incumbent_matrix_sha256: str
    candidate_matrix_sha256: str
    gate_report_sha256: tuple[str, ...]
    target_failure_sha256: tuple[str, ...]
    proposed_repairs: int
    accepted_repairs: int
    rejected_repairs: int
    repair_acceptance_rate: float | None
    task_success_rate_change: float | None
    policy_violation_rate_change: float | None
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
        rejected_repairs=proposed - accepted,
        repair_acceptance_rate=(accepted / proposed if proposed else None),
        task_success_rate_change=_difference(old_success_rate, new_success_rate),
        policy_violation_rate_change=_difference(old_policy_rate, new_policy_rate),
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
