"""Descriptive research checks over independently seeded cross-play matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from statistics import mean
from typing import Any

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
    discovery_matrix: CrossPlayMatrix
    confirmation_matrix: CrossPlayMatrix
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
    generation: int
    task_ids: tuple[str, ...]
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    old_customer_strategy_id: str
    new_customer_strategy_id: str
    old_service_strategy_id: str
    new_service_strategy_id: str

    def __post_init__(self) -> None:
        if (not isinstance(self.discovery_matrix, CrossPlayMatrix)
                or not isinstance(self.confirmation_matrix, CrossPlayMatrix)):
            raise TypeError("adaptation response must retain both cross-play source matrices")
        if self.discovery_matrix_sha256 != sha256_json(self.discovery_matrix.to_dict()):
            raise ValueError("discovery matrix hash does not match its retained source matrix")
        if self.confirmation_matrix_sha256 != sha256_json(self.confirmation_matrix.to_dict()):
            raise ValueError("confirmation matrix hash does not match its retained source matrix")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("adaptation response generation must be a non-negative integer")
        if (not self.task_ids or len(set(self.task_ids)) != len(self.task_ids)
                or any(not isinstance(item, str) or not item for item in self.task_ids)):
            raise ValueError("adaptation response requires a unique frozen task panel")
        for name, values in (
            ("discovery", self.discovery_seeds),
            ("confirmation", self.confirmation_seeds),
        ):
            if (not values or len(set(values)) != len(values)
                    or any(type(seed) is not int or seed < 0 for seed in values)):
                raise ValueError(f"adaptation response {name} seeds must be unique non-negative integers")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("adaptation response confirmation must use fresh seeds")
        if not all((
            self.old_customer_strategy_id, self.new_customer_strategy_id,
            self.old_service_strategy_id, self.new_service_strategy_id,
        )):
            raise ValueError("adaptation response must identify all four strategy checkpoints")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["outcome"] = self.outcome.value
        result["reasons"] = list(self.reasons)
        result["task_ids"] = list(self.task_ids)
        result["discovery_seeds"] = list(self.discovery_seeds)
        result["confirmation_seeds"] = list(self.confirmation_seeds)
        result["discovery_matrix"] = self.discovery_matrix.to_dict()
        result["confirmation_matrix"] = self.confirmation_matrix.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Any) -> AdaptationResponseReport:
        required = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("RQ3 transition report has missing or unknown fields")
        arrays = ("reasons", "task_ids", "discovery_seeds", "confirmation_seeds")
        if any(not isinstance(value[name], list) for name in arrays):
            raise TypeError("RQ3 transition arrays must be JSON arrays")
        try:
            discovery_matrix = CrossPlayMatrix.from_dict(value["discovery_matrix"])
            confirmation_matrix = CrossPlayMatrix.from_dict(value["confirmation_matrix"])
            expected = _analyze_adaptation_response_with_ids(
                discovery_matrix,
                confirmation_matrix,
                generation=value["generation"],
                old_customer_id=value["old_customer_strategy_id"],
                new_customer_id=value["new_customer_strategy_id"],
                old_service_id=value["old_service_strategy_id"],
                new_service_id=value["new_service_strategy_id"],
            )
            if expected.to_dict() != value:
                raise ValueError("RQ3 transition summary does not match its source matrices")
            return expected
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid RQ3 transition report: {exc}") from exc


def analyze_adaptation_response(
    discovery: CrossPlayMatrix,
    confirmation: CrossPlayMatrix,
    *,
    old_customer: CustomerStrategy,
    new_customer: CustomerStrategy,
    old_service: ServiceStrategy,
    new_service: ServiceStrategy,
    generation: int = 0,
) -> AdaptationResponseReport:
    """Check the four-cell response chain on a fresh, disjoint seed panel.

    The confirmation matrix must contain the same tasks and strategy IDs as
    discovery, but use different episode seeds. Rates come from verified
    failures divided by valid, applicable, adherent episodes in each cell.
    This is a descriptive observation check, not a significance test; formal
    conclusions still require analysis over independent evolution runs.
    """

    c_old = customer_strategy_id(old_customer)
    c_new = customer_strategy_id(new_customer)
    s_old = service_strategy_id(old_service)
    s_new = service_strategy_id(new_service)
    return _analyze_adaptation_response_with_ids(
        discovery,
        confirmation,
        generation=generation,
        old_customer_id=c_old,
        new_customer_id=c_new,
        old_service_id=s_old,
        new_service_id=s_new,
    )


def _analyze_adaptation_response_with_ids(
    discovery: CrossPlayMatrix,
    confirmation: CrossPlayMatrix,
    *,
    generation: int,
    old_customer_id: str,
    new_customer_id: str,
    old_service_id: str,
    new_service_id: str,
) -> AdaptationResponseReport:
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
    _require_matrix_strategy_ids(
        discovery, old_customer_id, new_customer_id, old_service_id, new_service_id,
    )

    rates = {
        "old_customer_old_service_rate": _failure_rate(
            confirmation, old_customer_id, old_service_id,
        ),
        "old_customer_new_service_rate": _failure_rate(
            confirmation, old_customer_id, new_service_id,
        ),
        "new_customer_new_service_rate": _failure_rate(
            confirmation, new_customer_id, new_service_id,
        ),
        "new_customer_old_service_rate": _failure_rate(
            confirmation, new_customer_id, old_service_id,
        ),
    }
    if any(value is None for value in rates.values()):
        return AdaptationResponseReport(
            outcome=AdaptationOutcome.INCONCLUSIVE,
            discovery_matrix_sha256=sha256_json(discovery.to_dict()),
            confirmation_matrix_sha256=sha256_json(confirmation.to_dict()),
            discovery_matrix=discovery,
            confirmation_matrix=confirmation,
            discovery_seed_count=len(discovery.seeds),
            confirmation_seed_count=len(confirmation.seeds),
            **rates,
            repair_reduction=None,
            counter_adaptation_increase=None,
            repair_specific_interaction=None,
            repair_reduced_failure_rate=None,
            customer_raised_failure_rate_after_repair=None,
            reasons=("one or more confirmation cells have no valid adherent denominator",),
            generation=generation,
            task_ids=discovery.task_ids,
            discovery_seeds=discovery.seeds,
            confirmation_seeds=confirmation.seeds,
            old_customer_strategy_id=old_customer_id,
            new_customer_strategy_id=new_customer_id,
            old_service_strategy_id=old_service_id,
            new_service_strategy_id=new_service_id,
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
        discovery_matrix=discovery,
        confirmation_matrix=confirmation,
        discovery_seed_count=len(discovery.seeds),
        confirmation_seed_count=len(confirmation.seeds),
        **rates,
        repair_reduction=repair_reduction,
        counter_adaptation_increase=counter_increase,
        repair_specific_interaction=interaction,
        repair_reduced_failure_rate=repair_reduced,
        customer_raised_failure_rate_after_repair=customer_raised,
        reasons=reasons,
        generation=generation,
        task_ids=discovery.task_ids,
        discovery_seeds=discovery.seeds,
        confirmation_seeds=confirmation.seeds,
        old_customer_strategy_id=old_customer_id,
        new_customer_strategy_id=new_customer_id,
        old_service_strategy_id=old_service_id,
        new_service_strategy_id=new_service_id,
    )


RQ3_CONDITIONS = ("adaptive_coevolution", "frozen_service", "random_mutation")


@dataclass(frozen=True, slots=True)
class EvolutionResponseRun:
    """Longitudinal cross-play evidence for one condition and seed block."""

    run_id: str
    seed_block_id: str
    evolution_seed: int
    condition: str
    request_budget_cap: int
    provider_attempts: int
    transitions: tuple[AdaptationResponseReport, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.run_id, str) or not self.run_id.strip()
                or not isinstance(self.seed_block_id, str) or not self.seed_block_id.strip()):
            raise ValueError("RQ3 run and paired seed-block IDs must be non-empty")
        if type(self.evolution_seed) is not int or self.evolution_seed < 0:
            raise ValueError("RQ3 evolution seed must be a non-negative integer")
        if self.condition not in RQ3_CONDITIONS:
            raise ValueError(f"RQ3 condition must be one of {RQ3_CONDITIONS}")
        if type(self.request_budget_cap) is not int or self.request_budget_cap <= 0:
            raise ValueError("RQ3 request budget cap must be positive")
        if (type(self.provider_attempts) is not int
                or not 0 <= self.provider_attempts <= self.request_budget_cap):
            raise ValueError("RQ3 provider attempts must fit within the frozen budget cap")
        if len(self.transitions) < 2:
            raise ValueError("RQ3 longitudinal runs need at least two consecutive response transitions")
        if any(not isinstance(item, AdaptationResponseReport) for item in self.transitions):
            raise TypeError("RQ3 transitions must be AdaptationResponseReport values")
        generations = tuple(item.generation for item in self.transitions)
        if generations != tuple(range(generations[0], generations[0] + len(generations))):
            raise ValueError("RQ3 response transitions must be ordered and generation-contiguous")
        task_panel = self.transitions[0].task_ids
        seen_seed_units: set[tuple[int, str]] = set()
        for index, report in enumerate(self.transitions):
            if report.task_ids != task_panel:
                raise ValueError("RQ3 run must use one frozen task panel across transitions")
            if set(report.discovery_seeds) & set(report.confirmation_seeds):
                raise ValueError("each RQ3 response must use fresh confirmation seeds")
            for task_id in report.task_ids:
                for seed in (*report.discovery_seeds, *report.confirmation_seeds):
                    key = (seed, task_id)
                    if key in seen_seed_units:
                        raise ValueError("RQ3 longitudinal transitions must use fresh task/seed panels")
                    seen_seed_units.add(key)
            if index:
                previous = self.transitions[index - 1]
                if (previous.new_customer_strategy_id != report.old_customer_strategy_id
                        or previous.new_service_strategy_id != report.old_service_strategy_id):
                    raise ValueError("RQ3 transitions must follow the prior generation's committed strategies")
        if self.condition == "frozen_service" and any(
            item.old_service_strategy_id != item.new_service_strategy_id for item in self.transitions
        ):
            raise ValueError("frozen_service control cannot change Service strategy checkpoints")

    @property
    def input_sha256(self) -> str:
        return sha256_json({
            "run_id": self.run_id,
            "seed_block_id": self.seed_block_id,
            "evolution_seed": self.evolution_seed,
            "condition": self.condition,
            "request_budget_cap": self.request_budget_cap,
            "provider_attempts": self.provider_attempts,
            "transitions": [item.to_dict() for item in self.transitions],
        })

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "seed_block_id": self.seed_block_id,
            "evolution_seed": self.evolution_seed,
            "condition": self.condition,
            "request_budget_cap": self.request_budget_cap,
            "provider_attempts": self.provider_attempts,
            "transitions": [item.to_dict() for item in self.transitions],
        }

    @classmethod
    def from_dict(cls, value: Any) -> EvolutionResponseRun:
        required = {
            "run_id", "seed_block_id", "evolution_seed", "condition",
            "request_budget_cap", "provider_attempts", "transitions",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("RQ3 study run has missing or unknown fields")
        if not isinstance(value["transitions"], list):
            raise TypeError("RQ3 study run transitions must be a JSON array")
        try:
            return cls(
                run_id=value["run_id"],
                seed_block_id=value["seed_block_id"],
                evolution_seed=value["evolution_seed"],
                condition=value["condition"],
                request_budget_cap=value["request_budget_cap"],
                provider_attempts=value["provider_attempts"],
                transitions=tuple(AdaptationResponseReport.from_dict(item)
                                  for item in value["transitions"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid RQ3 study run: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RQ3RunMetrics:
    run_id: str
    seed_block_id: str
    evolution_seed: int
    condition: str
    input_sha256: str
    transitions: int
    confirmed_response_chains: int
    sustained_two_chain_response: float | None
    mean_repair_reduction: float | None
    mean_counter_adaptation_increase: float | None
    mean_repair_specific_interaction: float | None


@dataclass(frozen=True, slots=True)
class RQ3PairedComparison:
    baseline_condition: str
    metric: str
    independent_seed_blocks: int
    adaptive_mean: float | None
    baseline_mean: float | None
    mean_paired_difference: float | None
    bootstrap_95_percentile_interval: tuple[float, float] | None
    adaptive_wins: int | None
    ties: int | None
    adaptive_losses: int | None
    status: str


@dataclass(frozen=True, slots=True)
class RQ3ResponseReport:
    status: str
    independent_seed_blocks: int
    minimum_independent_seed_blocks: int
    bootstrap_seed: int
    bootstrap_replicates: int
    runs: tuple[RQ3RunMetrics, ...]
    paired_comparisons: tuple[RQ3PairedComparison, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_rq3_longitudinal_response(
    runs: Sequence[EvolutionResponseRun],
    *,
    minimum_independent_seed_blocks: int = 3,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> RQ3ResponseReport:
    """Aggregate sustained response chains over paired independent runs.

    Response transitions are first reduced within each evolution run. Paired
    bootstrap resampling then uses only independent seed blocks; generations
    and episodes are never treated as independent observations. All estimates
    are descriptive until the formal analysis and sample size are frozen.
    """

    if type(minimum_independent_seed_blocks) is not int or minimum_independent_seed_blocks < 3:
        raise ValueError("RQ3 requires at least three independent seed blocks")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not runs:
        raise ValueError("RQ3 analysis requires longitudinal run data")

    by_block: dict[str, dict[str, EvolutionResponseRun]] = {}
    run_ids: set[str] = set()
    for run in runs:
        if run.run_id in run_ids:
            raise ValueError("RQ3 run IDs must be unique")
        run_ids.add(run.run_id)
        block = by_block.setdefault(run.seed_block_id, {})
        if run.condition in block:
            raise ValueError("each RQ3 seed block may contain only one run per condition")
        block[run.condition] = run

    block_ids = tuple(sorted(by_block))
    for block_id, conditions in by_block.items():
        if set(conditions) != set(RQ3_CONDITIONS):
            raise ValueError(f"RQ3 seed block {block_id!r} must contain all conditions")
        reference = conditions["adaptive_coevolution"]
        if len({item.evolution_seed for item in conditions.values()}) != 1:
            raise ValueError("paired RQ3 conditions must share their evolution seed block")
        for candidate in conditions.values():
            if candidate.request_budget_cap != reference.request_budget_cap:
                raise ValueError("paired RQ3 conditions must use the same request budget cap")
            if _transition_schedule(candidate) != _transition_schedule(reference):
                raise ValueError("paired RQ3 conditions must use the same task/seed schedule")

    if len({by_block[block]["adaptive_coevolution"].evolution_seed for block in block_ids}) != len(block_ids):
        raise ValueError("independent RQ3 seed blocks must use distinct evolution seeds")
    first = by_block[block_ids[0]]["adaptive_coevolution"]
    seen_task_seed_units: set[tuple[str, int]] = set()
    for conditions in by_block.values():
        if conditions["adaptive_coevolution"].request_budget_cap != first.request_budget_cap:
            raise ValueError("all RQ3 seed blocks must use one frozen request budget cap")
        reference_panel = tuple(item.task_ids for item in first.transitions)
        adaptive_run = conditions["adaptive_coevolution"]
        if tuple(item.task_ids for item in adaptive_run.transitions) != reference_panel:
            raise ValueError("RQ3 independent runs must share the same ordered task panels")
        for transition in adaptive_run.transitions:
            for task_id in transition.task_ids:
                for seed in (*transition.discovery_seeds, *transition.confirmation_seeds):
                    key = (task_id, seed)
                    if key in seen_task_seed_units:
                        raise ValueError(
                            "independent RQ3 seed blocks must use disjoint task/seed schedules"
                        )
                    seen_task_seed_units.add(key)

    metrics = tuple(
        _summarize_response_run(run)
        for block in block_ids
        for run in (by_block[block][condition] for condition in RQ3_CONDITIONS)
    )
    rng = random.Random(bootstrap_seed)
    comparisons = tuple(
        _compare_rq3_metric(
            by_block=by_block,
            block_ids=block_ids,
            baseline=baseline,
            metric=metric,
            minimum_blocks=minimum_independent_seed_blocks,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        for baseline in ("frozen_service", "random_mutation")
        for metric in (
            "sustained_two_chain_response",
            "mean_repair_reduction",
            "mean_counter_adaptation_increase",
            "mean_repair_specific_interaction",
        )
    )
    reasons = () if len(block_ids) >= minimum_independent_seed_blocks else (
        (f"only {len(block_ids)} independent evolution seed blocks; "
         f"{minimum_independent_seed_blocks} required for the planned RQ3 comparison"),
    )
    if any(item.status == "incomplete_denominator" for item in comparisons):
        reasons += ("one or more paired RQ3 metrics have incomplete transition denominators",)
    status = (
        "insufficient_independent_runs" if len(block_ids) < minimum_independent_seed_blocks
        else "incomplete_denominators" if reasons
        else "descriptive"
    )
    return RQ3ResponseReport(
        status=status,
        independent_seed_blocks=len(block_ids),
        minimum_independent_seed_blocks=minimum_independent_seed_blocks,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        runs=metrics,
        paired_comparisons=comparisons,
        reasons=reasons,
    )


def _transition_schedule(run: EvolutionResponseRun) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (item.generation, item.task_ids, item.discovery_seeds, item.confirmation_seeds)
        for item in run.transitions
    )


def _summarize_response_run(run: EvolutionResponseRun) -> RQ3RunMetrics:
    chains = sum(item.outcome == AdaptationOutcome.OBSERVED for item in run.transitions)
    has_two_chain_response = any(
        left.outcome == right.outcome == AdaptationOutcome.OBSERVED
        for left, right in zip(run.transitions, run.transitions[1:])
    )
    has_inconclusive = any(item.outcome == AdaptationOutcome.INCONCLUSIVE for item in run.transitions)
    sustained = 1.0 if has_two_chain_response else None if has_inconclusive else 0.0
    values = {
        "mean_repair_reduction": [item.repair_reduction for item in run.transitions],
        "mean_counter_adaptation_increase": [item.counter_adaptation_increase for item in run.transitions],
        "mean_repair_specific_interaction": [item.repair_specific_interaction for item in run.transitions],
    }
    means = {
        name: (mean(value for value in items if value is not None)
               if all(value is not None for value in items) else None)
        for name, items in values.items()
    }
    return RQ3RunMetrics(
        run_id=run.run_id,
        seed_block_id=run.seed_block_id,
        evolution_seed=run.evolution_seed,
        condition=run.condition,
        input_sha256=run.input_sha256,
        transitions=len(run.transitions),
        confirmed_response_chains=chains,
        sustained_two_chain_response=sustained,
        **means,
    )


def _compare_rq3_metric(
    *,
    by_block: dict[str, dict[str, EvolutionResponseRun]],
    block_ids: tuple[str, ...],
    baseline: str,
    metric: str,
    minimum_blocks: int,
    bootstrap_replicates: int,
    rng: random.Random,
) -> RQ3PairedComparison:
    adaptive_values = [
        getattr(_summarize_response_run(by_block[block]["adaptive_coevolution"]), metric)
        for block in block_ids
    ]
    baseline_values = [
        getattr(_summarize_response_run(by_block[block][baseline]), metric)
        for block in block_ids
    ]
    if any(value is None for value in (*adaptive_values, *baseline_values)):
        return RQ3PairedComparison(
            baseline, metric, len(block_ids), None, None, None, None,
            None, None, None, "incomplete_denominator",
        )
    adaptive = [float(value) for value in adaptive_values]
    control = [float(value) for value in baseline_values]
    differences = [left - right for left, right in zip(adaptive, control)]
    status = "descriptive" if len(block_ids) >= minimum_blocks else "insufficient_independent_runs"
    interval = None
    if status == "descriptive":
        bootstrap = [
            mean(rng.choice(differences) for _ in differences)
            for _ in range(bootstrap_replicates)
        ]
        interval = (_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975))
    return RQ3PairedComparison(
        baseline_condition=baseline,
        metric=metric,
        independent_seed_blocks=len(block_ids),
        adaptive_mean=mean(adaptive),
        baseline_mean=mean(control),
        mean_paired_difference=mean(differences),
        bootstrap_95_percentile_interval=interval,
        adaptive_wins=sum(value > 0 for value in differences),
        ties=sum(value == 0 for value in differences),
        adaptive_losses=sum(value < 0 for value in differences),
        status=status,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_rq3_document(value: Any) -> tuple[EvolutionResponseRun, ...]:
    """Load the strict JSON interchange format for longitudinal RQ3 runs."""

    if not isinstance(value, dict) or set(value) != {"schema_version", "runs"}:
        raise ValueError("RQ3 input must contain only schema_version and runs")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise ValueError("unsupported RQ3 input schema_version")
    if not isinstance(value["runs"], list) or not value["runs"]:
        raise ValueError("RQ3 runs must be a non-empty JSON array")
    return tuple(EvolutionResponseRun.from_dict(item) for item in value["runs"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze longitudinal co-evolution response chains over paired seed blocks."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 2 RQ3 run-level JSON")
    parser.add_argument("--output", type=Path, help="write a new report without overwriting an existing file")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        runs = load_rq3_document(json.loads(raw.decode("utf-8")))
        report = analyze_rq3_longitudinal_response(
            runs,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    payload = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "analysis": report.to_dict(),
    }
    rendered = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
        except OSError as exc:
            parser.error(str(exc))
    else:
        sys.stdout.write(rendered)
    return 0


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
