"""Pilot-informed, conservative seed-block planning for paired tests."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .formal_analysis import _student_t_cdf

_POWER_INPUT_FIELDS = {
    "schema_version", "study_id", "hypothesis_id", "pilot_artifact_sha256",
    "pilot_seed_blocks", "alternative", "noninferiority_margin",
    "minimum_relevant_effect", "familywise_alpha", "primary_family_size",
    "target_power", "maximum_seed_blocks", "simulation_replicates", "simulation_seed",
    "statistical_method", "permutation_seed", "permutation_replicates",
}


def calculate_paired_power(value: Any, *, input_sha256: str) -> dict[str, Any]:
    """Estimate paired-test power by resampling centered Pilot differences.

    The smallest relevant effect and available maximum sample size are inputs,
    not inferred from the observed Pilot outcome. Holm's first-step threshold
    (familywise alpha divided by the primary family size) is used as a
    conservative per-hypothesis bound. For paired sign-flip, the registered
    exact/Monte Carlo test is repeated inside each power simulation. This does
    not estimate joint power.
    """

    _require_sha256(input_sha256, "input_sha256")
    if not isinstance(value, dict) or set(value) != _POWER_INPUT_FIELDS:
        raise ValueError("power input has missing or unknown fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported power input schema_version")
    study_id = _nonempty_string(value["study_id"], "study_id")
    hypothesis_id = _nonempty_string(value["hypothesis_id"], "hypothesis_id")
    pilot_digest = _require_sha256(value["pilot_artifact_sha256"], "pilot_artifact_sha256")
    method = value["statistical_method"]
    if method not in {"paired_t", "paired_sign_flip"}:
        raise ValueError("statistical_method must be paired_t or paired_sign_flip")
    permutation_seed = value["permutation_seed"]
    permutation_replicates = value["permutation_replicates"]
    if method == "paired_t":
        if permutation_seed is not None or permutation_replicates is not None:
            raise ValueError("paired_t power inputs cannot include permutation settings")
    else:
        if type(permutation_seed) is not int or permutation_seed < 0:
            raise ValueError("paired_sign_flip requires a non-negative permutation_seed")
        if type(permutation_replicates) is not int or permutation_replicates < 9_999:
            raise ValueError("paired_sign_flip requires at least 9,999 permutation_replicates")

    rows = value["pilot_seed_blocks"]
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError("power input requires at least three Pilot seed-block differences")
    seed_blocks: list[tuple[int, float]] = []
    seen_seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"evolution_seed", "difference"}:
            raise ValueError("each Pilot seed-block row must contain evolution_seed and difference")
        seed = row["evolution_seed"]
        if type(seed) is not int or seed < 0 or seed in seen_seeds:
            raise ValueError("Pilot evolution seeds must be unique non-negative integers")
        seen_seeds.add(seed)
        seed_blocks.append((seed, _finite_number(row["difference"], "Pilot difference")))

    alternative = value["alternative"]
    if alternative not in {"greater", "less", "non_inferior"}:
        raise ValueError("alternative must be greater, less, or non_inferior")
    margin_value = value["noninferiority_margin"]
    if alternative == "non_inferior":
        margin = _finite_number(margin_value, "noninferiority_margin")
        if margin <= 0:
            raise ValueError("noninferiority_margin must be positive")
    else:
        if margin_value is not None:
            raise ValueError("only non-inferiority inputs may specify a margin")
        margin = None
    directed = [difference for _, difference in seed_blocks]
    if alternative == "less":
        directed = [-difference for difference in directed]
    elif alternative == "non_inferior":
        directed = [difference + margin for difference in directed]
    pilot_mean = mean(directed)
    pilot_sd = stdev(directed)
    if pilot_sd <= 0:
        raise ValueError("Pilot seed-block differences need non-zero variance for power planning")

    minimum_effect = _finite_number(value["minimum_relevant_effect"], "minimum_relevant_effect")
    if minimum_effect <= 0:
        raise ValueError("minimum_relevant_effect must be positive on the null-adjusted scale")
    alpha = _finite_number(value["familywise_alpha"], "familywise_alpha")
    if not 0 < alpha < 1:
        raise ValueError("familywise_alpha must be between zero and one")
    family_size = value["primary_family_size"]
    if type(family_size) is not int or family_size < 1:
        raise ValueError("primary_family_size must be a positive integer")
    target_power = _finite_number(value["target_power"], "target_power")
    if not 0.5 < target_power < 1:
        raise ValueError("target_power must be between 0.5 and one")
    maximum_blocks = value["maximum_seed_blocks"]
    if type(maximum_blocks) is not int or maximum_blocks < max(3, len(rows)):
        raise ValueError("maximum_seed_blocks must cover the Pilot sample and be at least three")
    replicates = value["simulation_replicates"]
    if type(replicates) is not int or replicates < 1_000:
        raise ValueError("simulation_replicates must be at least 1,000")
    seed = value["simulation_seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("simulation_seed must be a non-negative integer")

    test_alpha = alpha / family_size
    residuals = [item - pilot_mean for item in directed]
    generator = random.Random(seed)
    curve: list[dict[str, float | int]] = []
    planned_blocks: int | None = None
    planned_power: float | None = None
    planned_interval: tuple[float, float] | None = None
    for blocks in range(3, maximum_blocks + 1):
        critical = (
            _one_sided_t_critical(test_alpha, blocks - 1)
            if method == "paired_t" else None
        )
        sign_patterns = (
            _monte_carlo_sign_patterns(
                blocks, seed=permutation_seed, replicates=permutation_replicates,
            )
            if method == "paired_sign_flip" and blocks > 20 else None
        )
        rejected = 0
        for _ in range(replicates):
            observations = [minimum_effect + generator.choice(residuals) for _ in range(blocks)]
            if method == "paired_t":
                sample_mean = mean(observations)
                sample_variance = stdev(observations) ** 2
                if sample_variance == 0.0:
                    rejected += sample_mean > 0
                    continue
                statistic = sample_mean / math.sqrt(sample_variance / blocks)
                rejected += statistic >= critical
            elif blocks <= 20:
                rejected += _exact_sign_flip_rejects(observations, test_alpha)
            else:
                rejected += _monte_carlo_sign_flip_rejects(
                    observations, test_alpha, sign_patterns,
                )
        probability = rejected / replicates
        interval = _wilson_interval(rejected, replicates)
        curve.append({
            "seed_blocks": blocks,
            "estimated_power": probability,
            "monte_carlo_95_interval": list(interval),
        })
        if interval[0] >= target_power and planned_blocks is None:
            planned_blocks = blocks
            planned_power = probability
            planned_interval = interval

    return {
        "schema_version": 1,
        "status": "target_power_reached" if planned_blocks is not None else "target_power_not_reached",
        "study_id": study_id,
        "hypothesis_id": hypothesis_id,
        "pilot_artifact_sha256": pilot_digest,
        "power_input_sha256": input_sha256,
        "calculator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calculator": f"centered_empirical_residual_bootstrap_{method}_v1",
        "statistical_method": method,
        "permutation_seed": permutation_seed,
        "permutation_replicates": permutation_replicates,
        "alternative": alternative,
        "noninferiority_margin": margin,
        "pilot_independent_seed_blocks": len(rows),
        "pilot_mean_null_adjusted_difference": pilot_mean,
        "pilot_paired_difference_sd": pilot_sd,
        "minimum_relevant_effect_null_adjusted": minimum_effect,
        "familywise_alpha": alpha,
        "primary_family_size": family_size,
        "conservative_per_hypothesis_alpha": test_alpha,
        "target_power": target_power,
        "maximum_seed_blocks": maximum_blocks,
        "simulation_replicates": replicates,
        "simulation_seed": seed,
        "planned_seed_blocks": planned_blocks,
        "estimated_power_at_planned_seed_blocks": planned_power,
        "planned_power_lower_bound": planned_interval[0] if planned_interval is not None else None,
        "planned_power_monte_carlo_95_interval": (
            list(planned_interval) if planned_interval is not None else None
        ),
        "power_curve": curve,
    }


def calculate_paired_t_power(value: Any, *, input_sha256: str) -> dict[str, Any]:
    """Compatibility wrapper for callers that explicitly require paired t."""
    if not isinstance(value, dict) or value.get("statistical_method") != "paired_t":
        raise ValueError("calculate_paired_t_power requires statistical_method paired_t")
    return calculate_paired_power(value, input_sha256=input_sha256)


def _exact_sign_flip_rejects(values: list[float], alpha: float) -> bool:
    """Match Formal's inclusive one-sided exact sign-flip test in O(2^(n/2))."""
    observed = mean(values)
    tolerance = 1e-12 * max(1.0, abs(observed))
    target = sum(values) - tolerance * len(values)
    midpoint = len(values) // 2
    left = _signed_sums([abs(value) for value in values[:midpoint]])
    right = sorted(_signed_sums([abs(value) for value in values[midpoint:]]))
    extreme = sum(len(right) - bisect.bisect_left(right, target - item) for item in left)
    exact_p = extreme / (1 << len(values))
    return exact_p <= alpha


def _signed_sums(values: list[float]) -> list[float]:
    sums = [0.0]
    for value in values:
        previous = sums
        sums = [item - value for item in previous] + [item + value for item in previous]
    return sums


def _monte_carlo_sign_patterns(
    blocks: int, *, seed: int, replicates: int,
) -> list[tuple[int, ...]]:
    generator = random.Random(seed)
    return [
        tuple(1 if generator.getrandbits(1) else -1 for _ in range(blocks))
        for _ in range(replicates)
    ]


def _monte_carlo_sign_flip_rejects(
    values: list[float], alpha: float, sign_patterns: list[tuple[int, ...]],
) -> bool:
    observed = mean(values)
    tolerance = 1e-12 * max(1.0, abs(observed))
    extreme = sum(
        sum(value * sign for value, sign in zip(values, signs, strict=True)) / len(values)
        >= observed - tolerance
        for signs in sign_patterns
    )
    p_value = (extreme + 1) / (len(sign_patterns) + 1)
    return p_value <= alpha


def _one_sided_t_critical(alpha: float, degrees_freedom: int) -> float:
    target_cdf = 1 - alpha
    low, high = 0.0, 1.0
    while _student_t_cdf(high, degrees_freedom) < target_cdf:
        high *= 2
    for _ in range(70):
        midpoint = (low + high) / 2
        if _student_t_cdf(midpoint, degrees_freedom) < target_cdf:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate paired-test seed-block sample size from Pilot differences."
    )
    parser.add_argument("--input", required=True, type=Path, help="version 1 Pilot power-input JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        result = calculate_paired_power(
            document, input_sha256=hashlib.sha256(raw).hexdigest(),
        )
        rendered = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError,
            ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
