"""Opt-in frozen-E target improvement plus existing independent V preservation."""

from copy import deepcopy

from .evolution_failures import is_stuck, observable_outcome
from .records import customer_strategy_id, service_strategy_id

PROTOCOL = "discovery_targeted_repair_v1"
DEFAULT = {
    "protocol_version": PROTOCOL,
    "max_targets_per_generation": 1,
    "min_fixed_cells": 2,
    "max_target_regressions": 0,
    "require_all_seed_improvement": True,
    "require_zero_hard_violations": True,
}


def validate_policy(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT):
        raise ValueError("unknown targeted repair policy")
    p = {**deepcopy(DEFAULT), **deepcopy(raw)}
    if p["protocol_version"] != PROTOCOL:
        raise ValueError("unknown targeted protocol")
    for k in (
        "max_targets_per_generation",
        "min_fixed_cells",
        "max_target_regressions",
    ):
        if type(p[k]) is not int or p[k] < (0 if k == "max_target_regressions" else 1):
            raise ValueError("invalid targeted budget")
    for k in ("require_all_seed_improvement", "require_zero_hard_violations"):
        if type(p[k]) is not bool:
            raise ValueError("invalid targeted protection")
    return p


def evaluate_target(old, new, discovery, current_service, proposed_service, policy):
    p = validate_policy(policy)
    if discovery["after_service_id"] != service_strategy_id(current_service):
        raise ValueError("target repair frontier differs from current Service")
    challenge = discovery["compiled_customer"]
    from .strategies import PromptStrategy

    cid = customer_strategy_id(PromptStrategy(challenge["text"]))
    expected = {
        (discovery["discovery"]["task_id"], s) for s in discovery["discovery"]["seeds"]
    }
    a, b = {}, {}
    for records, index, svc in ((old, a, current_service), (new, b, proposed_service)):
        for r in records:
            key = (r.task_id, r.seed)
            if (
                key in index
                or r.customer_strategy_id != cid
                or r.service_strategy_id != service_strategy_id(svc)
            ):
                raise ValueError("targeted task/seed/strategy identity mismatch")
            index[key] = r
        if set(index) != expected:
            raise ValueError("targeted seed coverage mismatch")
    result = {
        "protocol_version": PROTOCOL,
        "discovery_id": discovery["discovery_id"],
        "panel": "E",
        "inference_scope": "observed_training_challenge_only",
        "causal_service_failure_confirmed": False,
        "policy": p,
    }
    if len(expected) < 2 or any(observable_outcome(r) is None for r in (*old, *new)):
        return {
            **result,
            "verdict": "INCONCLUSIVE",
            "reason": "unknown or unreplicated target outcomes",
        }
    fixed = sum(not a[k].task_success and b[k].task_success for k in expected)
    broken = sum(a[k].task_success and not b[k].task_success for k in expected)
    hard = sum(r.hard_policy_protocol_violations for r in new)
    stuck = sum(not is_stuck(a[k]) and is_stuck(b[k]) for k in expected)
    all_improve = all(not a[k].task_success and b[k].task_success for k in expected)
    accepted = (
        fixed >= p["min_fixed_cells"]
        and fixed > broken
        and broken <= p["max_target_regressions"]
        and not stuck
        and (not hard or not p["require_zero_hard_violations"])
        and (all_improve or not p["require_all_seed_improvement"])
    )
    return {
        **result,
        "verdict": "ACCEPTED" if accepted else "REJECTED",
        "reason": "replicated native target improvement"
        if accepted
        else "native target improvement/protection not satisfied",
        "fixed_cells": fixed,
        "regression_cells": broken,
        "new_stuck_cells": stuck,
        "hard_violations": hard,
        "old": [r.to_dict() for r in old],
        "new": [r.to_dict() for r in new],
    }
