"""Pure, deterministic, version-local Customer search allocation (no provider I/O)."""

from copy import deepcopy
from math import isfinite, log, sqrt

from .tau_provenance import sha256_json

PROTOCOL = "repair_conditioned_bandit_v1"
ARMS = (
    "disclosure_timing",
    "request_decomposition",
    "authorization_boundary",
    "correction_recovery",
    "open_exploration",
)
DEFAULT_POLICY = {
    "protocol_version": PROTOCOL,
    "beta": 1.0,
    "window_trials": 50,
    "open_exploration_interval": 5,
    "max_trials_per_generation": 2,
    "feedback_seeds": [1, 2],
    "min_replications": 2,
    "max_context_tokens": 30000,
}


def validate_policy(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_POLICY):
        raise ValueError("unknown RC-Bandit configuration")
    policy = {**deepcopy(DEFAULT_POLICY), **deepcopy(raw)}
    if policy["protocol_version"] != PROTOCOL:
        raise ValueError("unsupported RC-Bandit protocol")
    if (
        type(policy["beta"]) not in (float, int)
        or not isfinite(policy["beta"])
        or policy["beta"] < 0
    ):
        raise ValueError("beta must be finite and nonnegative")
    for key in (
        "window_trials",
        "open_exploration_interval",
        "max_trials_per_generation",
        "min_replications",
        "max_context_tokens",
    ):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if policy["open_exploration_interval"] > len(ARMS):
        raise ValueError("open exploration quota must be at least one in five pulls")
    seeds = policy["feedback_seeds"]
    if (
        not isinstance(seeds, list)
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
        or not 2 <= policy["min_replications"] <= len(seeds)
    ):
        raise ValueError("repair discovery requires at least two predeclared seeds")
    return policy


def new_state(policy):
    return {
        "protocol_version": PROTOCOL,
        "policy_sha256": sha256_json(validate_policy(policy)),
        "trials": [],
    }


def restore(value, policy):
    policy = validate_policy(policy)
    if (
        not isinstance(value, dict)
        or set(value) != {"protocol_version", "policy_sha256", "trials"}
        or value["protocol_version"] != PROTOCOL
        or value["policy_sha256"] != sha256_json(policy)
    ):
        raise ValueError("RC-Bandit state/config mismatch")
    state = new_state(policy)
    for trial in value["trials"]:
        if len(state["trials"]) != len(record_trial(state, trial)["trials"]) - 1:
            raise ValueError("duplicate trial in restored state")
        state = record_trial(state, trial)
    return state


def serialize(state):
    return deepcopy(state)


def choose_arm(state, service_pair_id, policy):
    """Fixed arm-order cold start/ties; quota precedes UCB. No stale version reward."""
    policy = validate_policy(policy)
    state = restore(state, policy)
    trials = [t for t in state["trials"] if t["service_pair_id"] == service_pair_id]
    recent = trials[-policy["window_trials"] :]
    counts = {arm: sum(t["arm"] == arm for t in recent) for arm in ARMS}
    scores = {
        arm: (
            sum(t["reward"] for t in recent if t["arm"] == arm) / counts[arm]
            if counts[arm]
            else 0.0
        )
        + policy["beta"] * sqrt(log(1 + len(recent)) / (1 + counts[arm]))
        for arm in ARMS
    }
    interval = policy["open_exploration_interval"]
    if (
        len(trials) >= interval - 1
        and not any(t["arm"] == "open_exploration" for t in trials[-(interval - 1) :])
        or interval == 1
    ):
        arm, reason = "open_exploration", "minimum open exploration quota"
    elif not service_pair_id:
        arm, reason = (
            ARMS[len(trials) % len(ARMS)],
            "no repair pair: uniform exploration",
        )
    elif any(counts[a] == 0 for a in ARMS):
        arm, reason = (
            next(a for a in ARMS if counts[a] == 0),
            "version-local cold start",
        )
    else:
        arm, reason = max(ARMS, key=lambda a: scores[a]), "version-local window UCB"
    return {
        "arm": arm,
        "service_pair_id": service_pair_id,
        "frontier_status": "repair_pair" if service_pair_id else "no_repair_pair",
        "reason": reason,
        "scores": scores,
        "pull_index": len(trials),
        "state_sha256": sha256_json(state),
    }


def record_trial(state, trial):
    """Idempotent exact publication; conflicts and unsupported positive reward fail closed."""
    if (
        not isinstance(trial, dict)
        or not isinstance(trial.get("trial_id"), str)
        or not trial["trial_id"]
        or trial.get("arm") not in ARMS
        or type(trial.get("reward")) is not int
        or trial["reward"] not in (0, 1)
        or trial.get("protocol_version") != PROTOCOL
        or not isinstance(trial.get("cost"), dict)
    ):
        raise ValueError("invalid Bandit trial")
    if trial["reward"] and (
        not trial.get("service_pair_id")
        or trial.get("feedback", {}).get("status") != "discovery"
        or trial.get("candidate_validity") != "valid"
        or trial.get("feedback", {}).get("service_pair_id") != trial["service_pair_id"]
        or not trial.get("feedback", {}).get("discovery_keys")
        or not trial.get("feedback", {}).get("reviewer_provenance")
        or not trial.get("feedback", {}).get("review")
        or trial.get("feedback", {}).get("reward") != trial["reward"]
    ):
        raise ValueError("unverified discovery cannot receive positive reward")
    for key, value in trial["cost"].items():
        if value is not None and (
            type(value) not in (int, float) or not isfinite(value) or value < 0
        ):
            raise ValueError(f"invalid actual cost: {key}")
    existing = next(
        (t for t in state["trials"] if t["trial_id"] == trial["trial_id"]), None
    )
    if existing is not None:
        if existing != trial:
            raise ValueError("immutable trial conflict")
        return deepcopy(state)
    if trial["reward"] and any(
        set(t.get("feedback", {}).get("discovery_keys", []))
        & set(trial["feedback"]["discovery_keys"])
        for t in state["trials"]
    ):
        raise ValueError("duplicate discovery reward")
    result = deepcopy(state)
    result["trials"].append(deepcopy(trial))
    return result
