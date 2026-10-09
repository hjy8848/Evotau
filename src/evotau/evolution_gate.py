"""Task-block paired bootstrap, conservative risk bounds and frozen look correction."""

import random
from statistics import NormalDist, mean

from .evolution_failures import episode_metrics, is_stuck, observable_outcome


def _interval(values, alpha):
    values = sorted(values)
    return [
        values[int((alpha / 2) * (len(values) - 1))],
        values[int((1 - alpha / 2) * (len(values) - 1))],
    ]


def _wilson_upper(events, trials, alpha):
    if not trials:
        return 1.0
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = events / trials
    return (
        p
        + z * z / (2 * trials)
        + z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5)
    ) / (1 + z * z / trials)


def evaluate_gate(
    old, new, policy, *, looks=1, smoke=False, seed=1, objective="superiority"
):
    if objective not in ("superiority", "preservation"):
        raise ValueError("unknown opponent gate objective")
    a = {(r.task_id, r.seed): r for r in old}
    b = {(r.task_id, r.seed): r for r in new}
    if len(a) != len(old) or len(b) != len(new) or a.keys() != b.keys() or not a:
        raise ValueError("statistical gate requires aligned nonempty paired cells")
    seed_coverage = {}
    for task_id, episode_seed in a:
        seed_coverage.setdefault(task_id, set()).add(episode_seed)
    expected_seeds = {r.seed for r in old}
    complete_seed_coverage = len(expected_seeds) >= 2 and all(
        seeds == expected_seeds for seeds in seed_coverage.values()
    )
    result = {
        "policy": dict(policy),
        "gate_looks": looks,
        "candidate_count": looks,
        "multiplicity": "bonferroni",
        "panel": "E" if smoke else "V",
        "seeds": sorted({r.seed for r in old}),
        "seed_coverage_by_task": {
            task_id: sorted(seeds) for task_id, seeds in seed_coverage.items()
        },
        "complete_paired_seed_coverage": complete_seed_coverage,
        "task_count": len({r.task_id for r in old}),
        "old_metrics": episode_metrics(old),
        "new_metrics": episode_metrics(new),
        "objective": objective,
        "method": policy.get("method", "task_block_bootstrap"),
    }
    if any(observable_outcome(r) is None for r in (*old, *new)):
        return {
            **result,
            "verdict": "INCONCLUSIVE",
            "reason": "unknown native outcomes; no fitness imputation",
        }
    blocks = {}
    for key, x in a.items():
        y = b[key]
        blocks.setdefault(key[0], []).append(
            (
                int(y.task_success) - int(x.task_success),
                int(x.task_success and not y.task_success),
                int(x.task_success),
                int(is_stuck(y)) - int(is_stuck(x)),
            )
        )
    values = list(blocks.values())
    paired_delta = mean(cell[0] for block in values for cell in block)
    hard_regressions = sum(
        max(
            0,
            b[k].hard_policy_protocol_violations - a[k].hard_policy_protocol_violations,
        )
        for k in a
    )
    alpha = (1 - policy["confidence"]) / max(1, looks)
    rng = random.Random(seed)
    draws, harms, stuck = [], [], []
    for _ in range(policy["bootstrap_samples"]):
        sample = [values[rng.randrange(len(values))] for _ in values]
        cells = [c for block in sample for c in block]
        draws.append(mean(c[0] for c in cells))
        denominator = sum(c[2] for c in cells)
        harms.append(sum(c[1] for c in cells) / denominator if denominator else 1.0)
        stuck.append(mean(c[3] for c in cells))
    risk_tasks = [block for block in values if any(c[2] for c in block)]
    harmfulness_ci = _interval(harms, alpha)
    # Bootstrap alone gives false certainty when zero regressions were observed.
    harmfulness_ci[1] = max(
        harmfulness_ci[1],
        _wilson_upper(
            sum(any(c[1] for c in block) for block in risk_tasks),
            len(risk_tasks),
            alpha,
        ),
    )
    result.update(
        paired_delta=paired_delta,
        fail_to_pass_count=sum(c[0] == 1 for block in values for c in block),
        pass_to_fail_count=sum(c[1] for block in values for c in block),
        success_ci=_interval(draws, alpha),
        harmfulness_ci=harmfulness_ci,
        stuck_delta_ci=_interval(stuck, alpha),
        adjusted_confidence=1 - alpha,
        hard_regressions=hard_regressions,
        regression_cells=[
            {"task_id": k[0], "seed": k[1]}
            for k, x in a.items()
            if x.task_success and not b[k].task_success
        ],
        new_stuck_cells=[
            {"task_id": k[0], "seed": k[1]}
            for k, x in a.items()
            if not is_stuck(x) and is_stuck(b[k])
        ],
    )
    seed_deltas = {str(s): mean(int(b[k].task_success) - int(a[k].task_success)
                               for k in a if k[1] == s) for s in sorted(expected_seeds)}
    result["paired_delta_by_seed"] = seed_deltas
    result["positive_seed_fraction"] = mean(v > 0 for v in seed_deltas.values())
    stuck_delta = mean(c[3] for block in values for c in block)
    finite_panel = policy.get("method") == "finite_panel_paired"
    observed_risk = policy.get("risk_scope") == "observed_panel"
    observed_harm = sum(c[1] for block in values for c in block) / max(1, sum(c[2] for block in values for c in block))
    result["observed_harmfulness"] = observed_harm
    result["risk_scope"] = policy.get("risk_scope", "population_bound")
    result.update(
        inference_scope="frozen_tasks_observed_seeds_only"
        if finite_panel or smoke or not policy["enabled"]
        else "task_block_population_inference",
        population_risk_certified=False,
    )
    if (hard_regressions or stuck_delta > policy["max_stuck_delta"]
        or result["new_metrics"]["stuck_rate"] > policy.get("max_stuck_rate", 1.0)
        or (policy.get("require_zero_hard_violations", False) and result["new_metrics"]["hard_violations"])):
        verdict, reason = (
            "REJECTED",
            "hard policy/protocol regression or increased stuck rate",
        )
    elif finite_panel:
        # This certifies observed cells only. It never claims V3 proves a 5% population risk bound.
        if result["regression_cells"] or result["new_stuck_cells"]:
            verdict, reason = "REJECTED", "observed paired-cell preservation violated"
        elif not complete_seed_coverage:
            verdict, reason = (
                "INCONCLUSIVE",
                "finite-panel check requires the same repeated paired seeds for every task",
            )
        elif objective == "preservation" or paired_delta > 0:
            verdict, reason = (
                "ACCEPTED",
                "observed paired-cell preservation"
                if objective == "preservation"
                else "observed paired-cell superiority with preservation",
            )
        else:
            verdict, reason = (
                "REJECTED",
                "current Customer native success did not strictly improve",
            )
    elif smoke or not policy["enabled"]:
        verdict, reason = (
            (
                "ACCEPTED",
                "strict native success improvement"
                if objective == "superiority"
                else "observed opponent preservation",
            )
            if (
                paired_delta > 0
                if objective == "superiority"
                else paired_delta >= 0
                and not result["regression_cells"]
                and not result["new_stuck_cells"]
            )
            else ("REJECTED", "native success did not satisfy opponent objective")
        )
    elif paired_delta < 0 or (
        observed_harm > policy["max_harmfulness"] if observed_risk
        else risk_tasks and harmfulness_ci[0] > policy["max_harmfulness"]
    ):
        verdict, reason = "REJECTED", "native regression or harmfulness above policy"
    elif len(values) < policy["min_tasks"] or not complete_seed_coverage:
        verdict, reason = (
            "INCONCLUSIVE",
            "insufficient tasks or stochastic seed coverage",
        )
    elif (
        (
            (
                objective == "preservation"
                and result["success_ci"][0] >= -policy.get("preservation_margin", 0.0)
            )
            or (
                objective == "superiority"
                and (
                    not policy["require_positive_success_lower_bound"]
                    or result["success_ci"][0] > policy.get("min_success_gain", 0.0)
                    and result["positive_seed_fraction"] >= policy.get("min_positive_seed_fraction", 0.0)
                )
            )
        )
        and (observed_harm <= policy["max_harmfulness"] if observed_risk
             else harmfulness_ci[1] <= policy["max_harmfulness"])
        and (stuck_delta <= policy["max_stuck_delta"] if observed_risk
             else result["stuck_delta_ci"][1] <= policy["max_stuck_delta"])
    ):
        verdict, reason = "ACCEPTED", "task-block confidence and risk gates passed"
        result["population_risk_certified"] = not policy.get("adaptive_validation", False)
    else:
        verdict, reason = (
            "INCONCLUSIVE",
            "improvement or risk bound insufficient after look correction",
        )
    if policy.get("adaptive_validation", False):
        result["inference_scope"] = "adaptive_validation_panel_task_block_evidence"
        result["population_risk_certified"] = False
    return {**result, "verdict": verdict, "reason": reason}


def cheap_screen(old, new, target_ids, protected_ids, *, max_regression_rate=0.0, max_stuck_delta=0.0, regression_allowance=0):
    a = {(r.task_id, r.seed): r for r in old}
    b = {(r.task_id, r.seed): r for r in new}
    if a.keys() != b.keys() or len(a) != len(old) or len(b) != len(new):
        raise ValueError("screen must be paired")
    if any(observable_outcome(r) is None for r in (*old, *new)):
        return {"passed": False, "reason": "uncertain_screen"}
    fixed = sum(
        k[0] in target_ids and not x.task_success and b[k].task_success
        for k, x in a.items()
    )
    # Every incumbent passing cell is protected, even when another seed of the same task is a target failure.
    protected_cells = [k for k, x in a.items() if x.task_success]
    broken_cells = [k for k in protected_cells if not b[k].task_success]
    broken = len(broken_cells)
    hard = sum(
        max(0, b[k].hard_policy_protocol_violations - x.hard_policy_protocol_violations)
        for k, x in a.items()
    )
    stuck = sum(is_stuck(b[k]) - is_stuck(x) for k, x in a.items())
    passed = (fixed > 0 and broken <= max(regression_allowance, max_regression_rate * len(protected_cells))
              and hard == 0 and stuck / max(1, len(a)) <= max_stuck_delta)
    return {
        "passed": passed,
        "max_regression_rate": max_regression_rate,
        "regression_allowance": regression_allowance,
        "max_stuck_delta": max_stuck_delta,
        "reason": "screen_passed"
        if passed
        else "screen_regression_or_no_target_fix",
        "fixed_cells": fixed,
        "protected_regressions": broken,
        "protected_cells": [{"task_id": t, "seed": s} for t, s in protected_cells],
        "broken_cells": [{"task_id": t, "seed": s} for t, s in broken_cells],
        "target_cells": [
            {"task_id": k[0], "seed": k[1]}
            for k, x in a.items()
            if k[0] in target_ids and not x.task_success
        ],
        "declared_protected_task_ids": sorted(protected_ids),
        "hard_regressions": hard,
        "stuck_count_delta": stuck,
        "old_metrics": episode_metrics(old),
        "new_metrics": episode_metrics(new),
    }


def pass_power_k(records, k):
    """Unbiased pass^k estimate per task (all k trials succeed, without replacement)."""
    import math

    groups = {}
    for record in records:
        if observable_outcome(record) is None:
            return None
        groups.setdefault(record.task_id, []).append(record.task_success)
    if not groups or any(len(v) < k for v in groups.values()):
        return None
    return mean(math.comb(sum(v), k) / math.comb(len(v), k) for v in groups.values())
