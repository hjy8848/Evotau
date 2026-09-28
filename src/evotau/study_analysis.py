"""Run-level analysis for budget-matched Customer search conditions.

The unit of comparison is an independent evolution-seed block. Episodes are
used to form run-level outcomes and denominators; they are never resampled as
independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean
from typing import Any

from .manifest import sha256_json
from .records import EpisodeRecord, EpisodeStatus, FailureRecord

RQ1_CONDITIONS = ("adaptive_customer", "static_customer", "random_mutation")


@dataclass(frozen=True, slots=True)
class StudyRun:
    """Audited outcome bundle for one condition and one independent run seed."""

    run_id: str
    seed_block_id: str
    evolution_seed: int
    condition: str
    task_ids: tuple[str, ...]
    request_budget_cap: int
    provider_attempts: int
    episodes: tuple[EpisodeRecord, ...]
    verified_failures: tuple[FailureRecord, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.run_id, str) or not self.run_id.strip()
                or not isinstance(self.seed_block_id, str) or not self.seed_block_id.strip()):
            raise ValueError("study run and independent seed-block IDs must be non-empty")
        if type(self.evolution_seed) is not int or self.evolution_seed < 0:
            raise ValueError("evolution seed must be non-negative")
        if self.condition not in RQ1_CONDITIONS:
            raise ValueError(f"condition must be one of {RQ1_CONDITIONS}")
        if (not self.task_ids or any(not isinstance(task_id, str) or not task_id for task_id in self.task_ids)
                or len(set(self.task_ids)) != len(self.task_ids)):
            raise ValueError("study run requires a non-empty unique fixed task panel")
        if type(self.request_budget_cap) is not int or self.request_budget_cap <= 0:
            raise ValueError("study run request budget cap must be positive")
        if (type(self.provider_attempts) is not int
                or not 0 <= self.provider_attempts <= self.request_budget_cap):
            raise ValueError("actual provider attempts must be within the fixed request budget")
        if not self.episodes:
            raise ValueError("study run must contain executed episode records")
        if any(not isinstance(item, EpisodeRecord) for item in self.episodes):
            raise TypeError("study run episodes must be EpisodeRecord values")
        if any(not isinstance(item, FailureRecord) for item in self.verified_failures):
            raise TypeError("study run failures must be verified FailureRecord values")

        episode_ids = [episode.episode_id for episode in self.episodes]
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("study run episode IDs must be unique")
        if any(episode.task_id not in self.task_ids for episode in self.episodes):
            raise ValueError("study run episode falls outside its frozen task panel")
        observed_tasks = {episode.task_id for episode in self.episodes}
        if observed_tasks != set(self.task_ids):
            raise ValueError("study run must include every task in the frozen task panel")

        by_episode = {episode.episode_id: episode for episode in self.episodes}
        failure_episode_ids = [failure.episode_id for failure in self.verified_failures]
        if len(failure_episode_ids) != len(set(failure_episode_ids)):
            raise ValueError("study run permits only the earliest verified failure per episode")
        for failure in self.verified_failures:
            episode = by_episode.get(failure.episode_id)
            if episode is None:
                raise ValueError("verified failure refers to an episode outside this study run")
            if not episode.has_attributable_failure_candidate:
                raise ValueError("verified failure is not backed by a complete valid adherent episode")
            if (
                failure.task_id != episode.task_id
                or failure.customer_strategy_id != episode.customer_strategy_id
                or failure.service_strategy_id != episode.service_strategy_id
                or failure.policy_ref != episode.policy_rule_id
                or failure.signature.domain != "retail"
                or failure.signature.workflow_stage != episode.workflow_stage
                or failure.signature.mistake_type != episode.mistake_type
                or failure.evidence != episode.evidence
            ):
                raise ValueError("verified failure fields do not match its episode evidence")

    @property
    def input_sha256(self) -> str:
        """Fingerprint the exact non-sensitive records used by the analysis."""

        episode_rows = []
        for episode in self.episodes:
            episode_rows.append({
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "seed": episode.seed,
                "customer_strategy_id": episode.customer_strategy_id,
                "service_strategy_id": episode.service_strategy_id,
                "status": episode.status.value,
                "task_success": episode.task_success,
                "native_reward": episode.native_reward,
                "customer_valid": episode.customer_valid,
                "strategy_applicable": episode.strategy_applicable,
                "customer_strategy_adherent": episode.customer_strategy_adherent,
                "policy_violation": episode.policy_violation,
                "policy_rule_id": episode.policy_rule_id,
                "mistake_type": episode.mistake_type,
                "workflow_stage": episode.workflow_stage,
                "evidence": [asdict(item) for item in episode.evidence],
                "trajectory_ref": episode.trajectory_ref,
                "audit_ref": episode.audit_ref,
                "tool_calls": episode.tool_calls,
            })
        return sha256_json({
            "run_id": self.run_id,
            "seed_block_id": self.seed_block_id,
            "evolution_seed": self.evolution_seed,
            "condition": self.condition,
            "task_ids": list(self.task_ids),
            "request_budget_cap": self.request_budget_cap,
            "provider_attempts": self.provider_attempts,
            "episodes": episode_rows,
            "verified_failures": [failure.to_dict() for failure in self.verified_failures],
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed_block_id": self.seed_block_id,
            "evolution_seed": self.evolution_seed,
            "condition": self.condition,
            "task_ids": list(self.task_ids),
            "request_budget_cap": self.request_budget_cap,
            "provider_attempts": self.provider_attempts,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "verified_failures": [failure.to_dict() for failure in self.verified_failures],
        }

    @classmethod
    def from_dict(cls, value: Any) -> StudyRun:
        required = {
            "run_id", "seed_block_id", "evolution_seed", "condition", "task_ids",
            "request_budget_cap", "provider_attempts", "episodes", "verified_failures",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("study run has missing or unknown fields")
        if not isinstance(value["task_ids"], list) or not all(
            isinstance(item, str) for item in value["task_ids"]
        ):
            raise ValueError("study run task_ids must be a JSON array of strings")
        if not isinstance(value["episodes"], list) or not isinstance(value["verified_failures"], list):
            raise TypeError("study run episodes and verified_failures must be JSON arrays")
        episode_fields = {item.name for item in fields(EpisodeRecord)}
        failure_fields = {item.name for item in fields(FailureRecord)}
        if any(not isinstance(item, dict) or set(item) != episode_fields for item in value["episodes"]):
            raise ValueError("study episode row has missing or unknown fields")
        if any(not isinstance(item, dict) or set(item) != failure_fields
               for item in value["verified_failures"]):
            raise ValueError("verified failure row has missing or unknown fields")
        try:
            return cls(
                run_id=value["run_id"],
                seed_block_id=value["seed_block_id"],
                evolution_seed=value["evolution_seed"],
                condition=value["condition"],
                task_ids=tuple(value["task_ids"]),
                request_budget_cap=value["request_budget_cap"],
                provider_attempts=value["provider_attempts"],
                episodes=tuple(EpisodeRecord.from_dict(item) for item in value["episodes"]),
                verified_failures=tuple(FailureRecord.from_dict(item)
                                        for item in value["verified_failures"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid study run: {exc}") from exc

    def metrics(self) -> RunMetrics:
        valid = tuple(
            episode for episode in self.episodes
            if episode.status in {EpisodeStatus.COMPLETE, EpisodeStatus.INVALID_STRATEGY}
            and episode.customer_valid is True
        )
        invalid = tuple(
            episode for episode in self.episodes
            if episode.status == EpisodeStatus.INVALID_CUSTOMER or episode.customer_valid is False
        )
        infrastructure = tuple(
            episode for episode in self.episodes
            if episode.status == EpisodeStatus.INFRASTRUCTURE_ERROR
        )
        classified_ids = {
            episode.episode_id for episode in (*valid, *invalid, *infrastructure)
        }
        opportunities = tuple(episode for episode in valid if episode.strategy_opportunity)
        adherent = tuple(
            episode for episode in opportunities if episode.customer_strategy_adherent is True
        )
        failure_episode_ids = {failure.episode_id for failure in self.verified_failures}
        task_signature_keys = {
            (failure.task_id, failure.signature.key) for failure in self.verified_failures
        }
        signature_keys = {failure.signature.key for failure in self.verified_failures}
        valid_failure_count = len(failure_episode_ids & {episode.episode_id for episode in valid})
        return RunMetrics(
            run_id=self.run_id,
            seed_block_id=self.seed_block_id,
            condition=self.condition,
            input_sha256=self.input_sha256,
            request_budget_cap=self.request_budget_cap,
            provider_attempts=self.provider_attempts,
            attempted_episodes=len(self.episodes),
            valid_episodes=len(valid),
            invalid_episodes=len(invalid),
            infrastructure_episodes=len(infrastructure),
            uncertain_episodes=len(self.episodes) - len(classified_ids),
            strategy_opportunities=len(opportunities),
            strategy_adherent_episodes=len(adherent),
            verified_failure_episodes=valid_failure_count,
            verified_task_signature_yield=len(task_signature_keys),
            verified_signature_count=len(signature_keys),
            attributable_failure_rate=(valid_failure_count / len(valid) if valid else None),
            adherent_failure_rate=(valid_failure_count / len(adherent) if adherent else None),
            discovery_yield_per_100_requests=(
                100 * len(task_signature_keys) / self.provider_attempts
                if self.provider_attempts else None
            ),
        )


@dataclass(frozen=True, slots=True)
class RunMetrics:
    run_id: str
    seed_block_id: str
    condition: str
    input_sha256: str
    request_budget_cap: int
    provider_attempts: int
    attempted_episodes: int
    valid_episodes: int
    invalid_episodes: int
    infrastructure_episodes: int
    uncertain_episodes: int
    strategy_opportunities: int
    strategy_adherent_episodes: int
    verified_failure_episodes: int
    verified_task_signature_yield: int
    verified_signature_count: int
    attributable_failure_rate: float | None
    adherent_failure_rate: float | None
    discovery_yield_per_100_requests: float | None


@dataclass(frozen=True, slots=True)
class RQ1Comparison:
    baseline_condition: str
    independent_seed_blocks: int
    adaptive_mean_yield: float
    baseline_mean_yield: float
    mean_paired_yield_difference: float
    median_paired_yield_difference: float
    adaptive_wins: int
    ties: int
    adaptive_losses: int
    bootstrap_95_percentile_interval: tuple[float, float] | None
    bootstrap_replicates: int
    attributable_failure_rate_difference: float | None
    outcome: str
    interpretation: str


@dataclass(frozen=True, slots=True)
class RQ1Report:
    status: str
    task_ids: tuple[str, ...]
    request_budget_cap: int
    minimum_independent_seed_blocks: int
    bootstrap_seed: int
    adaptive_runs: tuple[RunMetrics, ...]
    static_runs: tuple[RunMetrics, ...]
    random_runs: tuple[RunMetrics, ...]
    comparisons: tuple[RQ1Comparison, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_rq1_customer_advantage(
    runs: Sequence[StudyRun],
    *,
    minimum_independent_seed_blocks: int = 3,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> RQ1Report:
    """Compare adaptive Customer search to static and random controls.

    Each seed block must contain all three conditions on the same task panel
    and with the same declared provider-attempt cap. Pairing and resampling
    happen only over independent seed blocks. The percentile interval is
    descriptive; a formal claim still requires a pre-registered statistical
    plan and pilot-informed sample size.
    """

    if (type(minimum_independent_seed_blocks) is not int
            or minimum_independent_seed_blocks < 2):
        raise ValueError("at least two independent seed blocks are required")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not runs:
        raise ValueError("RQ1 analysis requires run-level data")

    by_block: dict[str, dict[str, StudyRun]] = {}
    run_ids: set[str] = set()
    all_episode_ids: set[str] = set()
    for run in runs:
        if run.run_id in run_ids:
            raise ValueError("RQ1 run IDs must be unique")
        run_ids.add(run.run_id)
        if all_episode_ids & {episode.episode_id for episode in run.episodes}:
            raise ValueError("an episode record cannot be reused across study runs")
        all_episode_ids.update(episode.episode_id for episode in run.episodes)
        block = by_block.setdefault(run.seed_block_id, {})
        if run.condition in block:
            raise ValueError("each seed block must contain at most one run per condition")
        block[run.condition] = run

    expected_conditions = set(RQ1_CONDITIONS)
    block_ids = tuple(sorted(by_block))
    for block_id, conditions in by_block.items():
        if set(conditions) != expected_conditions:
            raise ValueError(f"seed block {block_id!r} must contain all RQ1 conditions")
        reference = conditions["adaptive_customer"]
        for condition in RQ1_CONDITIONS[1:]:
            candidate = conditions[condition]
            if candidate.evolution_seed != reference.evolution_seed:
                raise ValueError("paired conditions in a seed block must share its evolution seed")
            if candidate.task_ids != reference.task_ids:
                raise ValueError("all conditions in a seed block must use the same ordered task panel")
            if candidate.request_budget_cap != reference.request_budget_cap:
                raise ValueError("all conditions in a seed block must use the same request budget cap")
            if _episode_schedule(candidate) != _episode_schedule(reference):
                raise ValueError("all conditions in a seed block must use the same task/seed schedule")

    if len({by_block[block_id]["adaptive_customer"].evolution_seed
            for block_id in block_ids}) != len(block_ids):
        raise ValueError("independent seed blocks must use distinct evolution seeds")

    task_ids = by_block[block_ids[0]]["adaptive_customer"].task_ids
    budget_cap = by_block[block_ids[0]]["adaptive_customer"].request_budget_cap
    for block in by_block.values():
        if block["adaptive_customer"].task_ids != task_ids:
            raise ValueError("independent seed blocks must use one frozen ordered task panel")
        if block["adaptive_customer"].request_budget_cap != budget_cap:
            raise ValueError("all independent seed blocks must use one fixed request budget")

    metrics = {
        (run.seed_block_id, run.condition): run.metrics()
        for run in runs
    }
    rng = random.Random(bootstrap_seed)
    comparisons = tuple(
        _compare_rq1_arm(
            metrics=metrics,
            block_ids=block_ids,
            baseline=baseline,
            minimum_blocks=minimum_independent_seed_blocks,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        for baseline in RQ1_CONDITIONS[1:]
    )
    reasons = () if len(block_ids) >= minimum_independent_seed_blocks else (
        f"only {len(block_ids)} independent seed blocks; {minimum_independent_seed_blocks} required for pilot-level comparison",
    )
    return RQ1Report(
        status="descriptive" if not reasons else "insufficient_independent_runs",
        task_ids=task_ids,
        request_budget_cap=budget_cap,
        minimum_independent_seed_blocks=minimum_independent_seed_blocks,
        bootstrap_seed=bootstrap_seed,
        adaptive_runs=tuple(metrics[(block, "adaptive_customer")] for block in block_ids),
        static_runs=tuple(metrics[(block, "static_customer")] for block in block_ids),
        random_runs=tuple(metrics[(block, "random_mutation")] for block in block_ids),
        comparisons=comparisons,
        reasons=reasons,
    )


def load_rq1_document(value: Any) -> tuple[StudyRun, ...]:
    """Load the strict, versioned JSON interchange format for run-level RQ1."""

    if not isinstance(value, dict) or set(value) != {"schema_version", "runs"}:
        raise ValueError("RQ1 input must contain only schema_version and runs")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported RQ1 input schema_version")
    if not isinstance(value["runs"], list):
        raise TypeError("RQ1 runs must be a JSON array")
    runs = tuple(StudyRun.from_dict(row) for row in value["runs"])
    if not runs:
        raise ValueError("RQ1 input must include at least one study run")
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze budget-matched adaptive, static, and random Customer study runs."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 1 run-level study JSON")
    parser.add_argument("--output", type=Path, help="write a new immutable report file; stdout by default")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        runs = load_rq1_document(document)
        report = analyze_rq1_customer_advantage(
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


def _compare_rq1_arm(
    *,
    metrics: dict[tuple[str, str], RunMetrics],
    block_ids: tuple[str, ...],
    baseline: str,
    minimum_blocks: int,
    bootstrap_replicates: int,
    rng: random.Random,
) -> RQ1Comparison:
    adaptive_yield = [
        metrics[(block, "adaptive_customer")].verified_task_signature_yield
        for block in block_ids
    ]
    baseline_yield = [metrics[(block, baseline)].verified_task_signature_yield for block in block_ids]
    differences = [adaptive - control for adaptive, control in zip(adaptive_yield, baseline_yield, strict=True)]
    interval = None
    if len(block_ids) >= minimum_blocks:
        sampled_means = [
            mean(rng.choice(differences) for _ in differences)
            for _ in range(bootstrap_replicates)
        ]
        interval = (_percentile(sampled_means, 0.025), _percentile(sampled_means, 0.975))

    failure_rate_differences = []
    for block in block_ids:
        adaptive_rate = metrics[(block, "adaptive_customer")].attributable_failure_rate
        baseline_rate = metrics[(block, baseline)].attributable_failure_rate
        if adaptive_rate is None or baseline_rate is None:
            failure_rate_differences = []
            break
        failure_rate_differences.append(adaptive_rate - baseline_rate)
    failure_rate_difference = mean(failure_rate_differences) if failure_rate_differences else None

    if len(block_ids) < minimum_blocks:
        outcome = "insufficient_independent_runs"
        interpretation = "run-level point estimates are descriptive only; the minimum independent-seed count was not met"
    elif mean(differences) > 0:
        outcome = "higher_adaptive_yield_observed"
        interpretation = "adaptive Customer had higher mean verified task-signature yield at the fixed budget; this is not a formal significance claim"
    else:
        outcome = "no_higher_adaptive_yield_observed"
        interpretation = "adaptive Customer did not have higher mean verified task-signature yield at the fixed budget"
    return RQ1Comparison(
        baseline_condition=baseline,
        independent_seed_blocks=len(block_ids),
        adaptive_mean_yield=mean(adaptive_yield),
        baseline_mean_yield=mean(baseline_yield),
        mean_paired_yield_difference=mean(differences),
        median_paired_yield_difference=_percentile(differences, 0.5),
        adaptive_wins=sum(value > 0 for value in differences),
        ties=sum(value == 0 for value in differences),
        adaptive_losses=sum(value < 0 for value in differences),
        bootstrap_95_percentile_interval=interval,
        bootstrap_replicates=bootstrap_replicates if interval is not None else 0,
        attributable_failure_rate_difference=failure_rate_difference,
        outcome=outcome,
        interpretation=interpretation,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _episode_schedule(run: StudyRun) -> tuple[tuple[str, int, int], ...]:
    counts = Counter((episode.task_id, episode.seed) for episode in run.episodes)
    return tuple(sorted((task_id, seed, count) for (task_id, seed), count in counts.items()))
