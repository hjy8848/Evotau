"""Descriptive research checks over independently seeded cross-play matrices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from .crossplay import CrossPlayMatrix
from .manifest import sha256_json
from .records import customer_strategy_id, service_strategy_id
from .strategies import CustomerStrategy, ServiceStrategy


class AdaptationOutcome(StrEnum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class AdaptationResponseReport:
    outcome: AdaptationOutcome
    discovery_matrix_sha256: str
    confirmation_matrix_sha256: str
    discovery_seed_count: int
    confirmation_seed_count: int
    old_customer_old_service_rate: float | None
    old_customer_new_service_rate: float | None
    new_customer_new_service_rate: float | None
    new_customer_old_service_rate: float | None
    repair_reduction: float | None
    counter_adaptation_increase: float | None
    repair_specific_interaction: float | None
    repair_reduced_failure_rate: bool | None
    customer_raised_failure_rate_after_repair: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["outcome"] = self.outcome.value
        result["reasons"] = list(self.reasons)
        return result


def analyze_adaptation_response(
    discovery: CrossPlayMatrix,
    confirmation: CrossPlayMatrix,
    *,
    old_customer: CustomerStrategy,
    new_customer: CustomerStrategy,
    old_service: ServiceStrategy,
    new_service: ServiceStrategy,
) -> AdaptationResponseReport:
    """Check the four-cell response chain on a fresh, disjoint seed panel.

    The confirmation matrix must contain the same tasks and strategy IDs as
    discovery, but use different episode seeds. Rates come from verified
    failures divided by valid, applicable, adherent episodes in each cell.
    This is a descriptive observation check, not a significance test; formal
    conclusions still require analysis over independent evolution runs.
    """

    if discovery.task_ids != confirmation.task_ids:
        raise ValueError("discovery and confirmation matrices must use the same frozen task panel")
    if not discovery.seeds or not confirmation.seeds:
        raise ValueError("discovery and confirmation matrices must each contain episode seeds")
    if discovery.customer_strategy_ids != confirmation.customer_strategy_ids:
        raise ValueError("discovery and confirmation matrices must use the same Customer strategies")
    if discovery.service_strategy_ids != confirmation.service_strategy_ids:
        raise ValueError("discovery and confirmation matrices must use the same Service strategies")
    if not set(discovery.seeds).isdisjoint(confirmation.seeds):
        raise ValueError("confirmation must use episode seeds not used for discovery")

    c_old = customer_strategy_id(old_customer)
    c_new = customer_strategy_id(new_customer)
    s_old = service_strategy_id(old_service)
    s_new = service_strategy_id(new_service)
    _require_matrix_strategy_ids(discovery, c_old, c_new, s_old, s_new)

    rates = {
        "old_customer_old_service_rate": _failure_rate(confirmation, c_old, s_old),
        "old_customer_new_service_rate": _failure_rate(confirmation, c_old, s_new),
        "new_customer_new_service_rate": _failure_rate(confirmation, c_new, s_new),
        "new_customer_old_service_rate": _failure_rate(confirmation, c_new, s_old),
    }
    if any(value is None for value in rates.values()):
        return AdaptationResponseReport(
            outcome=AdaptationOutcome.INCONCLUSIVE,
            discovery_matrix_sha256=sha256_json(discovery.to_dict()),
            confirmation_matrix_sha256=sha256_json(confirmation.to_dict()),
            discovery_seed_count=len(discovery.seeds),
            confirmation_seed_count=len(confirmation.seeds),
            **rates,
            repair_reduction=None,
            counter_adaptation_increase=None,
            repair_specific_interaction=None,
            repair_reduced_failure_rate=None,
            customer_raised_failure_rate_after_repair=None,
            reasons=("one or more confirmation cells have no valid adherent denominator",),
        )

    p_old_old = rates["old_customer_old_service_rate"]
    p_old_new = rates["old_customer_new_service_rate"]
    p_new_new = rates["new_customer_new_service_rate"]
    p_new_old = rates["new_customer_old_service_rate"]
    assert p_old_old is not None and p_old_new is not None
    assert p_new_new is not None and p_new_old is not None
    repair_reduction = p_old_old - p_old_new
    counter_increase = p_new_new - p_old_new
    interaction = counter_increase - (p_new_old - p_old_old)
    repair_reduced = p_old_new < p_old_old
    customer_raised = p_new_new > p_old_new
    outcome = (
        AdaptationOutcome.OBSERVED if repair_reduced and customer_raised
        else AdaptationOutcome.NOT_OBSERVED
    )
    reasons = () if outcome == AdaptationOutcome.OBSERVED else (
        "the fresh-seed confirmation panel did not satisfy both directional response inequalities",
    )
    return AdaptationResponseReport(
        outcome=outcome,
        discovery_matrix_sha256=sha256_json(discovery.to_dict()),
        confirmation_matrix_sha256=sha256_json(confirmation.to_dict()),
        discovery_seed_count=len(discovery.seeds),
        confirmation_seed_count=len(confirmation.seeds),
        **rates,
        repair_reduction=repair_reduction,
        counter_adaptation_increase=counter_increase,
        repair_specific_interaction=interaction,
        repair_reduced_failure_rate=repair_reduced,
        customer_raised_failure_rate_after_repair=customer_raised,
        reasons=reasons,
    )


def _require_matrix_strategy_ids(
    matrix: CrossPlayMatrix,
    old_customer_id: str,
    new_customer_id: str,
    old_service_id: str,
    new_service_id: str,
) -> None:
    if not {old_customer_id, new_customer_id} <= set(matrix.customer_strategy_ids):
        raise ValueError("cross-play matrix is missing an old or new Customer strategy")
    if not {old_service_id, new_service_id} <= set(matrix.service_strategy_ids):
        raise ValueError("cross-play matrix is missing an old or new Service strategy")


def _failure_rate(matrix: CrossPlayMatrix, customer_id: str, service_id: str) -> float | None:
    cell = next(
        (item for item in matrix.cells
         if item.customer_strategy_id == customer_id and item.service_strategy_id == service_id),
        None,
    )
    if cell is None:
        raise ValueError("cross-play matrix is missing a required Customer/Service pair")
    return cell.verified_failure_rate
