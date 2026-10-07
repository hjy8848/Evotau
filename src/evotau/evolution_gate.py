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


def evaluate_gate(old, new, policy, *, looks=1, smoke=False, seed=1):
    a = {(r.task_id, r.seed): r for r in old}
    b = {(r.task_id, r.seed): r for r in new}
    if len(a) != len(old) or len(b) != len(new) or a.keys() != b.keys() or not a:
        raise ValueError("statistical gate requires aligned nonempty paired cells")
    result = {
        "policy": dict(policy),
        "gate_looks": looks,
        "candidate_count": looks,
        "multiplicity": "bonferroni",
        "panel": "E" if smoke else "V",
        "seeds": sorted({r.seed for r in old}),
        "task_count": len({r.task_id for r in old}),
        "old_metrics": episode_metrics(old),
        "new_metrics": episode_metrics(new),
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
        success_ci=_interval(draws, alpha),
        harmfulness_ci=harmfulness_ci,
        stuck_delta_ci=_interval(stuck, alpha),
        adjusted_confidence=1 - alpha,
        hard_regressions=hard_regressions,
    )
    stuck_delta = mean(c[3] for block in values for c in block)
    if hard_regressions or stuck_delta > policy["max_stuck_delta"]:
        verdict, reason = (
            "REJECTED",
            "hard policy/protocol regression or increased stuck rate",
        )
    elif smoke or not policy["enabled"]:
        verdict, reason = (
            ("ACCEPTED", "strict native success improvement")
            if paired_delta > 0
            else ("REJECTED", "native success did not strictly improve")
        )
    elif paired_delta < 0 or harmfulness_ci[0] > policy["max_harmfulness"]:
        verdict, reason = "REJECTED", "native regression or harmfulness above policy"
    elif len(values) < policy["min_tasks"] or len(result["seeds"]) < 2:
        verdict, reason = (
            "INCONCLUSIVE",
            "insufficient tasks or stochastic seed coverage",
        )
    elif (
        (
            not policy["require_positive_success_lower_bound"]
            or result["success_ci"][0] > 0
        )
        and harmfulness_ci[1] <= policy["max_harmfulness"]
        and result["stuck_delta_ci"][1] <= policy["max_stuck_delta"]
    ):
        verdict, reason = "ACCEPTED", "task-block confidence and risk gates passed"
    else:
        verdict, reason = (
            "INCONCLUSIVE",
            "improvement or risk bound insufficient after look correction",
        )
    return {**result, "verdict": verdict, "reason": reason}


def cheap_screen(old, new, target_ids, protected_ids):
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
    broken = sum(
        k[0] in protected_ids and x.task_success and not b[k].task_success
        for k, x in a.items()
    )
    hard = sum(
        max(0, b[k].hard_policy_protocol_violations - x.hard_policy_protocol_violations)
        for k, x in a.items()
    )
    stuck = sum(is_stuck(b[k]) - is_stuck(x) for k, x in a.items())
    return {
        "passed": fixed > 0 and broken == 0 and hard == 0 and stuck <= 0,
        "reason": "screen_passed"
        if fixed > 0 and broken == 0 and hard == 0 and stuck <= 0
        else "screen_regression_or_no_target_fix",
        "fixed_cells": fixed,
        "protected_regressions": broken,
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
