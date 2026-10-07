"""Frozen, strict V2 experimental policy; no research settings inferred at runtime."""

from copy import deepcopy

from .service_skills import V2_MUTATION_TYPES

DEFAULT_V2 = {
    "schema_version": 2,
    "service_skill_runtime": "activate_topk_v2",
    "max_active_service_skills": 2,
    "activator": {"model": None, "model_args": {}},
    "service_evolution": {
        "candidates_per_cluster": 3,
        "max_clusters_per_generation": 1,
        "semantic_dedup": True,
        "crossover": True,
        "stagnation_patience": 2,
        "lineage": True,
    },
    "evaluation": {
        "repair_panel": "E",
        "gate_panel": "V",
        "screen_seeds": [1, 2],
        "gate_seeds": [1, 2, 3, 4],
        "screen_clean_tasks": 2,
    },
    "statistical_gate": {
        "enabled": True,
        "confidence": 0.95,
        "max_harmfulness": 0.05,
        "require_positive_success_lower_bound": True,
        "bootstrap_samples": 2000,
        "min_tasks": 8,
        "multiple_look_correction": "bonferroni",
        "max_stuck_delta": 0.0,
    },
    "archive": {"enabled": True, "max_candidates": 5},
    "opponent_replay": {
        "enabled": True,
        "archived_customers": 2,
        "include_native_customer": True,
        "current_weight": 1.0,
    },
    "history": {"summarize": True, "max_families": 8, "representative_cases": 6},
    "skill_budgets": {
        "max_skills": 8,
        "guidance_chars": 2400,
        "guidance_tokens": 800,
        "active_tokens": 2000,
        "max_active_skills": 2,
    },
}


def freeze_v2_policy(raw, models, model_args):
    policy = deepcopy(DEFAULT_V2)
    if not isinstance(raw, dict) or set(raw) - set(policy):
        raise ValueError("unknown Skill Evolution V2 configuration fields")
    for key, value in raw.items():
        if isinstance(policy[key], dict):
            if not isinstance(value, dict) or set(value) - set(policy[key]):
                raise ValueError(f"unknown V2 policy fields in {key}")
            policy[key].update(deepcopy(value))
        else:
            policy[key] = deepcopy(value)
    if policy["schema_version"] != 2:
        raise ValueError("unsupported V2 schema")
    if policy["service_skill_runtime"] not in ("activate_topk_v2", "render_all_v1"):
        raise ValueError("unsupported activation ablation")
    for group in (
        "service_evolution",
        "statistical_gate",
        "archive",
        "opponent_replay",
        "history",
    ):
        for key, default in DEFAULT_V2[group].items():
            value = policy[group][key]
            if isinstance(default, bool) and type(value) is not bool:
                raise ValueError(f"{group}.{key} must be boolean")
            if type(default) is int and (
                type(value) is not int
                or value < (0 if key == "archived_customers" else 1)
            ):
                raise ValueError(f"{group}.{key} must be a valid integer")
    if (
        type(policy["max_active_service_skills"]) is not int
        or not 1 <= policy["max_active_service_skills"] <= 2
    ):
        raise ValueError("V2 activation K must be 1 or 2")
    for key in ("screen_seeds", "gate_seeds"):
        seeds = policy["evaluation"][key]
        if (
            not isinstance(seeds, list)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
        ):
            raise ValueError("seed schedules must be unique nonnegative integer arrays")
    if (
        policy["evaluation"]["repair_panel"] != "E"
        or policy["evaluation"]["gate_panel"] != "V"
    ):
        raise ValueError("V2 uses fixed E repair and disjoint V gate panels")
    gate = policy["statistical_gate"]
    if not 0.5 < gate["confidence"] < 1 or not 0 <= gate["max_harmfulness"] <= 1:
        raise ValueError("invalid confidence/harmfulness policy")
    if gate["multiple_look_correction"] != "bonferroni":
        raise ValueError("unsupported multiple look correction")
    if policy["opponent_replay"]["current_weight"] <= 0:
        raise ValueError("current replay weight must be positive")
    for value in policy["skill_budgets"].values():
        if type(value) is not int or value < 1:
            raise ValueError("skill budgets must be positive integers")
    policy["skill_budgets"]["max_active_skills"] = policy["max_active_service_skills"]
    policy["activator"]["model"] = policy["activator"]["model"] or models["agent"]
    if not policy["activator"]["model_args"]:
        policy["activator"]["model_args"] = deepcopy(model_args["agent"])
    from .tau_provenance import _role_model_args_payload, freeze_role_model_args

    args = freeze_role_model_args(
        {"activator": policy["activator"]["model_args"]}, roles=("activator",)
    )
    policy["activator"]["model_args"] = _role_model_args_payload(args)["activator"]
    if (
        not isinstance(policy["activator"]["model"], str)
        or not policy["activator"]["model"]
    ):
        raise ValueError("activator model must be frozen")
    return policy


MUTATION_TYPES = V2_MUTATION_TYPES
