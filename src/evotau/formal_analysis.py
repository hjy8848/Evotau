"""Preregistered paired inference over independent evolution seed blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from .preregistration import validate_formal_preregistration

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONDITIONS_RQ3 = {"adaptive_coevolution", "frozen_service", "random_mutation"}


@dataclass(frozen=True, slots=True)
class FormalHypothesisResult:
    hypothesis_id: str
    research_question: str
    endpoint: str
    contrast: str
    alternative: str
    method: str
    multiplicity_family: str
    independent_seed_blocks: int
    mean_paired_difference: float
    median_paired_difference: float
    paired_difference_sd: float
    noninferiority_margin: float | None
    test_statistic: float | None
    raw_p_value: float
    holm_adjusted_p_value: float
    reject_at_familywise_alpha: bool
    permutation_mode: str | None
    permutation_seed: int | None
    permutation_replicates_registered: int | None
    permutation_replicates_used: int
    paired_differences: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class FormalInferenceReport:
    status: str
    study_id: str
    preregistration_plan_sha256: str
    registry_reference_unverified: str
    report_artifact_sha256: tuple[tuple[str, str], ...]
    report_input_sha256: tuple[tuple[str, str], ...]
    familywise_alpha: float
    multiple_comparison_method: str
    tests: tuple[FormalHypothesisResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_formal_reports(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    rq1_report: Mapping[str, Any],
    rq1_report_sha256: str,
    rq2_report: Mapping[str, Any],
    rq2_report_sha256: str,
    rq3_report: Mapping[str, Any],
    rq3_report_sha256: str,
) -> FormalInferenceReport:
    """Calculate only the hypotheses and methods frozen in a valid plan.

    RQ1/RQ2/RQ3 report files must be outputs of the existing run-level
    analyzers. The test unit is the independent evolution seed, paired within
    that seed. Missing run-level values stop the analysis instead of shrinking
    the denominator.
    """

    _require_sha256(plan_sha256, "plan_sha256")
    report_docs = {"rq1": rq1_report, "rq2": rq2_report, "rq3": rq3_report}
    report_hashes = {
        "rq1": rq1_report_sha256, "rq2": rq2_report_sha256, "rq3": rq3_report_sha256,
    }
    for name, digest in report_hashes.items():
        _require_sha256(digest, f"{name}_report_sha256")
    plan_summary = validate_formal_preregistration(
        plan, exact_input_sha256=plan_sha256,
    )
    registered_seeds = tuple(plan["independent_evolution_seeds"])
    expected_seed_set = set(registered_seeds)
    analyses: dict[str, Mapping[str, Any]] = {}
    source_input_hashes: dict[str, str] = {}
    for name, document in report_docs.items():
        if not isinstance(document, Mapping) or set(document) != {"input_sha256", "analysis"}:
            raise ValueError(f"{name.upper()} report must contain input_sha256 and analysis")
        source_input_hashes[name] = _sha_value(document["input_sha256"], f"{name}.input_sha256")
        if not isinstance(document["analysis"], Mapping):
            raise TypeError(f"{name.upper()} report analysis must be a JSON object")
        analyses[name] = document["analysis"]

    rq1 = _require_complete_rq1(analyses["rq1"], plan, expected_seed_set)
    rq2 = _require_complete_rq2(analyses["rq2"], plan, expected_seed_set)
    rq3 = _require_complete_rq3(analyses["rq3"], plan, expected_seed_set)

    primary = [item for item in plan["hypotheses"] if item["primary"]]
    intermediate: list[tuple[dict[str, Any], list[tuple[int, float]], float | None]] = []
    for hypothesis in primary:
        rq = hypothesis["research_question"]
        endpoint = hypothesis["endpoint"]
        contrast = hypothesis["contrast"]
        if rq == "RQ1":
            adaptive = rq1["adaptive_customer"]
            baseline = rq1[contrast]
            differences = [
                (seed, adaptive[seed][endpoint] - baseline[seed][endpoint])
                for seed in registered_seeds
            ]
            margin = None
        elif rq == "RQ2":
            rq2_endpoint = (
                f"{endpoint}_adaptive_minus_one_shot"
                if contrast == "one_shot_repair" else endpoint
            )
            differences = [(seed, rq2[rq2_endpoint][seed]) for seed in registered_seeds]
            margin = float(plan["noninferiority_margins"].get(endpoint)) if hypothesis["alternative"] == "non_inferior" else None
        else:
            adaptive = rq3["adaptive_coevolution"]
            baseline = rq3[contrast]
            differences = [
                (seed, adaptive[seed][endpoint] - baseline[seed][endpoint])
                for seed in registered_seeds
            ]
            margin = None
        intermediate.append((hypothesis, differences, margin))

    raw_results = [
        _test_hypothesis(hypothesis, differences, margin)
        for hypothesis, differences, margin in intermediate
    ]
    adjusted = _holm_adjust(intermediate, raw_results)
    alpha = float(plan["familywise_alpha"])
    tests = tuple(
        FormalHypothesisResult(
            hypothesis_id=hypothesis["hypothesis_id"],
            research_question=hypothesis["research_question"],
            endpoint=hypothesis["endpoint"],
            contrast=hypothesis["contrast"],
            alternative=hypothesis["alternative"],
            method=hypothesis["statistical_method"],
            multiplicity_family=hypothesis["multiplicity_family"],
            independent_seed_blocks=len(differences),
            mean_paired_difference=mean(value for _, value in differences),
            median_paired_difference=median(value for _, value in differences),
            paired_difference_sd=stdev(value for _, value in differences),
            noninferiority_margin=margin,
            test_statistic=statistic,
            raw_p_value=raw_p,
            holm_adjusted_p_value=adjusted[hypothesis["hypothesis_id"]],
            reject_at_familywise_alpha=adjusted[hypothesis["hypothesis_id"]] <= alpha,
            permutation_mode=mode,
            permutation_seed=hypothesis["permutation_seed"],
            permutation_replicates_registered=hypothesis["permutation_replicates"],
            permutation_replicates_used=replicates,
            paired_differences=tuple(differences),
        )
        for (hypothesis, differences, margin), (statistic, raw_p, mode, replicates) in zip(
            intermediate, raw_results, strict=True,
        )
    )
    return FormalInferenceReport(
        status="formal_inference_computed",
        study_id=plan_summary.study_id,
        preregistration_plan_sha256=plan_sha256,
        registry_reference_unverified=plan_summary.registry_url,
        report_artifact_sha256=tuple(sorted(report_hashes.items())),
        report_input_sha256=tuple(sorted(source_input_hashes.items())),
        familywise_alpha=alpha,
        multiple_comparison_method=plan["multiple_comparison_method"],
        tests=tests,
    )


def _require_complete_rq1(
    analysis: Mapping[str, Any], plan: Mapping[str, Any], expected_seeds: set[int],
) -> dict[str, dict[int, dict[str, float]]]:
    _require_descriptive_status(analysis, "RQ1")
    if tuple(analysis.get("task_ids", ())) != tuple(plan["task_panels"]["E"]):
        raise ValueError("RQ1 report task panel differs from the preregistered E panel")
    report_cap = analysis.get("request_budget_cap")
    if (type(report_cap) is not int
            or report_cap != plan["condition_budget_caps"]["adaptive_customer"]):
        raise ValueError("RQ1 report request cap differs from the preregistered budget")
    out: dict[str, dict[int, dict[str, float]]] = {}
    block_ids_by_condition: dict[str, dict[int, str]] = {}
    expected_conditions = {"adaptive_customer", "static_customer", "random_mutation"}
    for condition in sorted(expected_conditions):
        rows = analysis.get({
            "adaptive_customer": "adaptive_runs",
            "static_customer": "static_runs",
            "random_mutation": "random_runs",
        }[condition])
        out[condition], block_ids_by_condition[condition] = _run_metric_map(
            rows, condition, expected_seeds, plan,
        )
    if set(out["adaptive_customer"]) != expected_seeds:
        raise ValueError("RQ1 report does not contain every preregistered evolution seed")
    for condition in ("static_customer", "random_mutation"):
        if set(out[condition]) != expected_seeds:
            raise ValueError("RQ1 paired control is missing a preregistered evolution seed")
        if block_ids_by_condition[condition] != block_ids_by_condition["adaptive_customer"]:
            raise ValueError("RQ1 conditions disagree on paired seed-block IDs")
    return out


def _run_metric_map(
    rows: Any,
    condition: str,
    expected_seeds: set[int],
    plan: Mapping[str, Any],
) -> tuple[dict[int, dict[str, float]], dict[int, str]]:
    if not isinstance(rows, list):
        raise TypeError(f"RQ1 {condition} runs must be an array")
    result: dict[int, dict[str, float]] = {}
    condition_blocks: dict[int, str] = {}
    block_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("RQ1 run metrics must be JSON objects")
        seed = row.get("evolution_seed")
        block_id = row.get("seed_block_id")
        if type(seed) is not int or seed < 0 or seed not in expected_seeds:
            raise ValueError("RQ1 run has an evolution seed outside the preregistered schedule")
        if not isinstance(block_id, str) or not block_id or block_id in block_ids:
            raise ValueError("RQ1 run seed-block IDs must be unique per condition")
        block_ids.add(block_id)
        condition_blocks[seed] = block_id
        if seed in result:
            raise ValueError(f"RQ1 {condition} repeats an evolution seed")
        if row.get("condition") != condition:
            raise ValueError("RQ1 run condition does not match its report arm")
        _check_run_budget(row, condition, plan)
        _sha_value(row.get("input_sha256"), "RQ1 run input_sha256")
        yield_value = row.get("verified_task_signature_yield")
        if type(yield_value) is not int or yield_value < 0:
            raise ValueError("RQ1 verified discovery yield must be a non-negative integer")
        result[seed] = {"verified_task_signature_yield": float(yield_value)}
    return result, condition_blocks


def _check_run_budget(row: Mapping[str, Any], condition: str, plan: Mapping[str, Any]) -> None:
    cap = row.get("request_budget_cap")
    attempts = row.get("provider_attempts")
    if type(cap) is not int or cap != plan["condition_budget_caps"].get(condition):
        raise ValueError(f"{condition} report budget cap differs from preregistration")
    if type(attempts) is not int or not 0 <= attempts <= cap:
        raise ValueError(f"{condition} provider attempts are outside its frozen budget")


def _require_complete_rq2(
    analysis: Mapping[str, Any], plan: Mapping[str, Any], expected_seeds: set[int],
) -> dict[str, dict[int, float]]:
    _require_descriptive_status(analysis, "RQ2")
    if (tuple(analysis.get("evolution_task_ids", ())) != tuple(plan["task_panels"]["E"])
            or tuple(analysis.get("validation_task_ids", ())) != tuple(plan["task_panels"]["V"])
            or tuple(analysis.get("heldout_task_ids", ())) != tuple(plan["task_panels"]["H"])):
        raise ValueError("RQ2 report E/V/H panels differ from preregistration")
    cap = analysis.get("request_budget_cap")
    caps = set(plan["condition_budget_caps"].values())
    if type(cap) is not int or len(caps) != 1 or cap != next(iter(caps)):
        raise ValueError("RQ2 report request cap differs from the preregistered budget")
    run_rows = analysis.get("run_ids_and_input_sha256")
    if not isinstance(run_rows, list):
        raise TypeError("RQ2 run provenance must be a JSON array")
    run_seed: dict[str, int] = {}
    for item in run_rows:
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError("RQ2 run provenance row must contain run ID, evolution seed, and input hash")
        run_id, seed, digest = item
        if not isinstance(run_id, str) or not run_id or type(seed) is not int or seed < 0:
            raise ValueError("RQ2 run provenance has invalid ID or seed")
        _sha_value(digest, "RQ2 run input_sha256")
        if run_id in run_seed or seed in run_seed.values():
            raise ValueError("RQ2 report repeats a run ID or evolution seed")
        run_seed[run_id] = seed
    if set(run_seed.values()) != expected_seeds:
        raise ValueError("RQ2 report seeds differ from the preregistered schedule")
    attempts_rows = analysis.get("actual_provider_attempts")
    if not isinstance(attempts_rows, list):
        raise TypeError("RQ2 actual provider attempts must be a JSON array")
    attempts = {}
    for item in attempts_rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("RQ2 provider-attempt row must contain run ID and count")
        run_id, count = item
        if run_id not in run_seed or type(count) is not int or not 0 <= count <= cap:
            raise ValueError("RQ2 provider attempts exceed or leave the frozen design")
        if run_id in attempts:
            raise ValueError("RQ2 repeats a provider-attempt row")
        attempts[run_id] = count
    if set(attempts) != set(run_seed):
        raise ValueError("RQ2 provider-attempt rows do not cover every run")

    condition_rows = analysis.get("condition_run_ids_and_input_sha256")
    if not isinstance(condition_rows, list):
        raise TypeError("RQ2 condition run provenance must be a JSON array")
    condition_runs: dict[str, tuple[int, str, str]] = {}
    condition_seed_sets: dict[str, set[int]] = {
        "adaptive_coevolution": set(), "one_shot_repair": set(),
    }
    for item in condition_rows:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError("RQ2 condition provenance row must contain ID, seed, condition, and hash")
        run_id, seed, condition, digest = item
        if (not isinstance(run_id, str) or not run_id or type(seed) is not int
                or seed < 0 or condition not in condition_seed_sets):
            raise ValueError("RQ2 condition provenance has an invalid run identity")
        _sha_value(digest, "RQ2 condition run input_sha256")
        if run_id in condition_runs or seed in condition_seed_sets[condition]:
            raise ValueError("RQ2 repeats a condition run ID or seed")
        condition_runs[run_id] = (seed, condition, digest)
        condition_seed_sets[condition].add(seed)
    if any(seeds != expected_seeds for seeds in condition_seed_sets.values()):
        raise ValueError("RQ2 condition runs must cover every preregistered seed block")
    adaptive_provenance = {
        run_id: (seed, digest)
        for run_id, (seed, condition, digest) in condition_runs.items()
        if condition == "adaptive_coevolution"
    }
    if adaptive_provenance != {
        run_id: (seed, next(item[2] for item in run_rows if item[0] == run_id))
        for run_id, seed in run_seed.items()
    }:
        raise ValueError("RQ2 adaptive run provenance differs from its condition provenance")

    condition_attempt_rows = analysis.get("condition_actual_provider_attempts")
    if not isinstance(condition_attempt_rows, list):
        raise TypeError("RQ2 condition provider-attempt rows must be a JSON array")
    condition_attempts = {}
    for item in condition_attempt_rows:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("RQ2 condition provider-attempt row must contain run ID and count")
        run_id, count = item
        if (run_id not in condition_runs or type(count) is not int
                or not 0 <= count <= cap or run_id in condition_attempts):
            raise ValueError("RQ2 condition provider attempts are outside the frozen design")
        condition_attempts[run_id] = count
    if set(condition_attempts) != set(condition_runs):
        raise ValueError("RQ2 condition provider attempts do not cover every run")
    if {
        run_id: count for run_id, count in condition_attempts.items()
        if condition_runs[run_id][1] == "adaptive_coevolution"
    } != attempts:
        raise ValueError("RQ2 adaptive provider attempts differ from condition run provenance")

    metrics = analysis.get("metrics")
    if not isinstance(metrics, list):
        raise TypeError("RQ2 metrics must be a JSON array")
    result: dict[str, dict[int, float]] = {}
    for row in metrics:
        if not isinstance(row, Mapping) or row.get("status") != "descriptive":
            raise ValueError("RQ2 metric is incomplete or not ready for Formal inference")
        name = row.get("metric")
        if not isinstance(name, str) or not name:
            raise ValueError("RQ2 metric must have a non-empty name")
        if name in result:
            raise ValueError("RQ2 report repeats a metric")
        if row.get("independent_runs") != len(expected_seeds):
            raise ValueError(f"RQ2 metric {name!r} does not use every preregistered seed block")
        if row.get("complete_run_values") != len(expected_seeds):
            raise ValueError(f"RQ2 metric {name!r} has an incomplete Formal denominator")
        observations = row.get("observations")
        if not isinstance(observations, list):
            raise TypeError("RQ2 metric observations must be an array")
        values: dict[int, float] = {}
        for item in observations:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("RQ2 observation must contain run ID and value")
            run_id, value = item
            if run_id not in run_seed or run_seed[run_id] in values:
                raise ValueError("RQ2 metric observations have unknown or duplicate runs")
            values[run_seed[run_id]] = _finite_number(value, f"RQ2 {name} observation")
        if set(values) != expected_seeds:
            raise ValueError(f"RQ2 metric {name!r} has incomplete seed-block denominators")
        result[name] = values
    expected_metrics = {
        item["endpoint"] for item in plan["hypotheses"]
        if item["primary"] and item["research_question"] == "RQ2"
    }
    if not expected_metrics <= set(result):
        raise ValueError("RQ2 report is missing one or more preregistered primary endpoints")
    condition_contrasts = analysis.get("condition_contrasts")
    if not isinstance(condition_contrasts, list):
        raise TypeError("RQ2 condition contrasts must be a JSON array")
    seen_contrasts: set[tuple[str, str]] = set()
    required_control = ("target_failure_rate_reduction", "one_shot_repair")
    for row in condition_contrasts:
        if not isinstance(row, Mapping):
            raise TypeError("RQ2 condition contrast must be a JSON object")
        endpoint = row.get("endpoint")
        baseline_condition = row.get("baseline_condition")
        key = (endpoint, baseline_condition)
        if (not isinstance(endpoint, str) or baseline_condition != "one_shot_repair"
                or key in seen_contrasts):
            raise ValueError("RQ2 condition contrast has an invalid or duplicate endpoint")
        seen_contrasts.add(key)
        if row.get("status") != "descriptive":
            raise ValueError("RQ2 adaptive-versus-one-shot contrast is incomplete for Formal inference")
        if (row.get("independent_seed_blocks") != len(expected_seeds)
                or row.get("complete_seed_blocks") != len(expected_seeds)):
            raise ValueError("RQ2 condition contrast does not cover every preregistered seed")
        observations = row.get("observations")
        if not isinstance(observations, list):
            raise TypeError("RQ2 condition contrast observations must be an array")
        values: dict[int, float] = {}
        for observation in observations:
            required_fields = {
                "evolution_seed", "adaptive_run_id", "adaptive_input_sha256",
                "one_shot_run_id", "one_shot_input_sha256", "adaptive_value",
                "one_shot_value", "paired_difference",
            }
            if not isinstance(observation, Mapping) or set(observation) != required_fields:
                raise ValueError("RQ2 condition observation has missing or unknown fields")
            seed = observation["evolution_seed"]
            if type(seed) is not int or seed not in expected_seeds or seed in values:
                raise ValueError("RQ2 condition observation has a duplicate or unexpected seed")
            adaptive = condition_runs.get(observation["adaptive_run_id"])
            one_shot = condition_runs.get(observation["one_shot_run_id"])
            if adaptive is None or one_shot is None or (
                adaptive[0], adaptive[1], adaptive[2]
            ) != (
                seed, "adaptive_coevolution", observation["adaptive_input_sha256"],
            ) or (
                one_shot[0], one_shot[1], one_shot[2]
            ) != (
                seed, "one_shot_repair", observation["one_shot_input_sha256"],
            ):
                raise ValueError("RQ2 condition observation source runs do not match its seed and hashes")
            adaptive_value = _finite_number(observation["adaptive_value"], "RQ2 adaptive condition value")
            one_shot_value = _finite_number(observation["one_shot_value"], "RQ2 one-shot condition value")
            difference = _finite_number(observation["paired_difference"], "RQ2 paired condition difference")
            if not math.isclose(difference, adaptive_value - one_shot_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("RQ2 paired condition difference does not match its source values")
            values[seed] = difference
        if set(values) != expected_seeds:
            raise ValueError("RQ2 condition contrast has incomplete seed-block observations")
        result[f"{endpoint}_adaptive_minus_one_shot"] = values
    if required_control not in seen_contrasts:
        raise ValueError("RQ2 report is missing the preregistered one-shot repair contrast")
    return result


def _require_complete_rq3(
    analysis: Mapping[str, Any], plan: Mapping[str, Any], expected_seeds: set[int],
) -> dict[str, dict[int, dict[str, float]]]:
    _require_descriptive_status(analysis, "RQ3")
    if tuple(analysis.get("task_ids", ())) != tuple(plan["task_panels"]["E"]):
        raise ValueError("RQ3 report task panel differs from the preregistered E panel")
    rows = analysis.get("runs")
    if not isinstance(rows, list):
        raise TypeError("RQ3 run metrics must be an array")
    result: dict[str, dict[int, dict[str, float]]] = {condition: {} for condition in _CONDITIONS_RQ3}
    seen_blocks: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("RQ3 run metrics must be JSON objects")
        condition = row.get("condition")
        if condition not in _CONDITIONS_RQ3:
            raise ValueError("RQ3 run has an unknown condition")
        seed = row.get("evolution_seed")
        block_id = row.get("seed_block_id")
        if type(seed) is not int or seed not in expected_seeds or not isinstance(block_id, str) or not block_id:
            raise ValueError("RQ3 run seed or seed-block ID is outside the frozen schedule")
        if seed in seen_blocks and seen_blocks[seed] != block_id:
            raise ValueError("paired RQ3 conditions disagree on their seed-block ID")
        seen_blocks[seed] = block_id
        if seed in result[condition]:
            raise ValueError("RQ3 report repeats a condition/evolution seed")
        _check_run_budget(row, condition, plan)
        _sha_value(row.get("input_sha256"), "RQ3 run input_sha256")
        metric_value = row.get("sustained_two_chain_response")
        metric_value = _finite_number(metric_value, "RQ3 sustained response")
        if metric_value not in {0.0, 1.0}:
            raise ValueError("RQ3 sustained response endpoint must be binary")
        result[condition][seed] = {"sustained_two_chain_response": metric_value}
    if set(seen_blocks) != expected_seeds or any(
        set(values) != expected_seeds for values in result.values()
    ):
        raise ValueError("RQ3 report lacks a complete paired condition/seed panel")
    return result


def paired_differences_from_report(
    plan: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    document: Mapping[str, Any],
) -> tuple[tuple[int, float], ...]:
    """Derive one hypothesis's ordered Pilot differences from its report."""

    if not isinstance(document, Mapping) or set(document) != {"input_sha256", "analysis"}:
        raise ValueError("Pilot report must contain input_sha256 and analysis")
    _sha_value(document["input_sha256"], "Pilot report input_sha256")
    analysis = document["analysis"]
    if not isinstance(analysis, Mapping):
        raise TypeError("Pilot report analysis must be a JSON object")
    seeds = tuple(plan["independent_evolution_seeds"])
    expected_seeds = set(seeds)
    rq = hypothesis["research_question"]
    endpoint = hypothesis["endpoint"]
    if rq == "RQ1":
        values = _require_complete_rq1(analysis, plan, expected_seeds)
        adaptive = values["adaptive_customer"]
        baseline = values[hypothesis["contrast"]]
        return tuple(
            (seed, adaptive[seed][endpoint] - baseline[seed][endpoint])
            for seed in seeds
        )
    if rq == "RQ2":
        values = _require_complete_rq2(analysis, plan, expected_seeds)
        return tuple((seed, values[endpoint][seed]) for seed in seeds)
    if rq == "RQ3":
        values = _require_complete_rq3(analysis, plan, expected_seeds)
        adaptive = values["adaptive_coevolution"]
        baseline = values[hypothesis["contrast"]]
        return tuple(
            (seed, adaptive[seed][endpoint] - baseline[seed][endpoint])
            for seed in seeds
        )
    raise ValueError("Pilot report hypothesis must refer to RQ1, RQ2, or RQ3")


def _require_descriptive_status(analysis: Mapping[str, Any], name: str) -> None:
    if analysis.get("status") != "descriptive":
        raise ValueError(f"{name} report is not complete and descriptive")


def _test_hypothesis(
    hypothesis: Mapping[str, Any],
    differences: Sequence[tuple[int, float]],
    margin: float | None,
) -> tuple[float | None, float, str | None, int]:
    if len(differences) < 3 or len({seed for seed, _ in differences}) != len(differences):
        raise ValueError("Formal paired inference requires at least three unique seed blocks")
    values = [value for _, value in differences]
    if margin is not None:
        values = [value + margin for value in values]
    if hypothesis["alternative"] == "less":
        values = [-value for value in values]
    elif hypothesis["alternative"] not in {"greater", "non_inferior"}:
        raise ValueError("unsupported preregistered alternative")
    method = hypothesis["statistical_method"]
    if method == "paired_t":
        statistic = _paired_t_greater(values)
        return statistic, _paired_t_pvalue(values, statistic), None, 0
    if method == "paired_sign_flip":
        return _sign_flip_pvalue(
            values,
            seed=hypothesis["permutation_seed"],
            replicates=hypothesis["permutation_replicates"],
        )
    raise ValueError("unsupported preregistered statistical method")


def _paired_t_greater(values: Sequence[float]) -> float | None:
    deviation = stdev(values)
    if deviation == 0:
        return None
    return mean(values) / (deviation / math.sqrt(len(values)))


def _paired_t_pvalue(values: Sequence[float], statistic: float | None = None) -> float:
    if statistic is None:
        return 0.0 if mean(values) > 0 else 1.0
    if statistic == 0.0:
        return 1.0
    return min(1.0, max(0.0, 1.0 - _student_t_cdf(statistic, len(values) - 1)))


def _student_t_cdf(value: float, degrees_freedom: int) -> float:
    if degrees_freedom <= 0:
        raise ValueError("Student t degrees of freedom must be positive")
    if value == 0:
        return 0.5
    x = degrees_freedom / (degrees_freedom + value * value)
    beta = _regularized_incomplete_beta(x, degrees_freedom / 2, 0.5)
    return 1 - beta / 2 if value > 0 else beta / 2


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1 - factor * _beta_continued_fraction(b, a, 1 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    tiny = 1e-300
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    result = d
    for iteration in range(1, 201):
        m2 = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        change = d * c
        result *= change
        if abs(change - 1) < 3e-14:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _sign_flip_pvalue(
    values: Sequence[float], *, seed: int, replicates: int,
) -> tuple[float, float, str, int]:
    observed = mean(values)
    tolerance = 1e-12 * max(1.0, abs(observed))
    if len(values) <= 20:
        extreme = 0
        total = 1 << len(values)
        for mask in range(total):
            statistic = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            ) / len(values)
            extreme += statistic >= observed - tolerance
        return observed, extreme / total, "exact", total
    generator = random.Random(seed)
    extreme = 0
    for _ in range(replicates):
        statistic = mean(value if generator.getrandbits(1) else -value for value in values)
        extreme += statistic >= observed - tolerance
    return observed, (extreme + 1) / (replicates + 1), "monte_carlo", replicates


def _holm_adjust(
    intermediate: Sequence[tuple[dict[str, Any], list[tuple[int, float]], float | None]],
    results: Sequence[tuple[float | None, float, str | None, int]],
) -> dict[str, float]:
    if len(intermediate) != len(results):
        raise ValueError("Holm adjustment inputs have different hypothesis counts")
    families: dict[str, list[tuple[str, float]]] = {}
    seen_ids: set[str] = set()
    for (hypothesis, _, _), (_, p_value, _, _) in zip(intermediate, results, strict=True):
        identifier = hypothesis["hypothesis_id"]
        family = hypothesis["multiplicity_family"]
        if identifier in seen_ids:
            raise ValueError("Formal inference repeats a hypothesis ID")
        if not math.isfinite(p_value) or not 0 <= p_value <= 1:
            raise ValueError("hypothesis p-values must be finite and between zero and one")
        seen_ids.add(identifier)
        families.setdefault(family, []).append((identifier, p_value))

    adjusted: dict[str, float] = {}
    for family_results in families.values():
        ordered = sorted(family_results, key=lambda item: item[1])
        running_max = 0.0
        count = len(ordered)
        for index, (identifier, p_value) in enumerate(ordered):
            running_max = max(running_max, (count - index) * p_value)
            adjusted[identifier] = min(1.0, running_max)
    return adjusted


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number with a complete denominator")
    return float(value)


def _sha_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the hypotheses and paired seed-block tests frozen in a Formal plan."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--rq1-report", required=True, type=Path)
    parser.add_argument("--rq2-report", required=True, type=Path)
    parser.add_argument("--rq3-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plan_raw = args.plan.read_bytes()
        plan = json.loads(plan_raw.decode("utf-8"))
        validate_formal_preregistration(
            plan,
            exact_input_sha256=hashlib.sha256(plan_raw).hexdigest(),
            artifact_root=args.plan.resolve().parent,
        )
        documents: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        for name in ("rq1", "rq2", "rq3"):
            path = getattr(args, f"{name}_report")
            raw = path.read_bytes()
            documents[name] = json.loads(raw.decode("utf-8"))
            hashes[name] = hashlib.sha256(raw).hexdigest()
        report = analyze_formal_reports(
            plan=plan,
            plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
            rq1_report=documents["rq1"], rq1_report_sha256=hashes["rq1"],
            rq2_report=documents["rq2"], rq2_report_sha256=hashes["rq2"],
            rq3_report=documents["rq3"], rq3_report_sha256=hashes["rq3"],
        )
        rendered = json.dumps(report.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
            TypeError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
