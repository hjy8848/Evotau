"""Frozen, strict V2 experimental policy; no research settings inferred at runtime."""

import math
from copy import deepcopy

from .service_skills import V2_MUTATION_TYPES

DEFAULT_V2 = {
    "schema_version": 2,
    "algorithm_version": "direct_skill_evolution_v1",
    "service_skill_runtime": "activate_topk_v2",
    "max_active_service_skills": 2,
    "activator": {"model": None, "model_args": {}},
    "service_evolution": {
        "candidates_per_generation": 3,
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
        "v_gate_mode": "fail_fast",
    },
    "mutation_context": {
        "representative_cases": 4,
        "case_chars": 12000,
        "max_proxy_tokens": 40000,
    },
    "statistical_gate": {
        "method": "task_block_bootstrap",
        "preservation_margin": 0.0,
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
    "history": {"summarize": True, "max_families": 8},
    "skill_budgets": {
        "max_skills": 8,
        "guidance_chars": 2400,
        "guidance_tokens": 800,
        "active_tokens": 2000,
        "max_active_skills": 2,
    },
}


def freeze_v2_policy(raw, models, model_args):
    raw = deepcopy(raw)
    if not isinstance(raw, dict):
        raise TypeError("V2 policy must be a mapping")
    # Old manifests can be inspected, but the runtime explicitly refuses legacy algorithms.
    version = raw.get("algorithm_version", "diagnoser_v2")
    if version not in ("direct_skill_evolution_v1", "diagnoser_v2"):
        raise ValueError("unsupported evolution algorithm version")
    raw["algorithm_version"] = version
    evolution = raw.get("service_evolution", {})
    if not isinstance(evolution, dict):
        raise TypeError("service_evolution must be a mapping")
    legacy = {
        k: evolution[k]
        for k in ("candidates_per_cluster", "max_clusters_per_generation")
        if k in evolution
    }
    if legacy:
        if "candidates_per_generation" in evolution:
            raise ValueError("cannot mix legacy and direct candidate budgets")
        count = legacy.get("candidates_per_cluster", 3)
        clusters = legacy.get("max_clusters_per_generation", 1)
        if any(type(v) is not int or v < 1 for v in (count, clusters)):
            raise ValueError("legacy candidate limits must be positive integers")
        evolution = {k: v for k, v in evolution.items() if k not in legacy}
        evolution["candidates_per_generation"] = count * clusters
        raw["service_evolution"] = evolution
    # This old limit only selected Diagnoser cases; direct evidence has its own frozen budget.
    legacy_context_limit = None
    if (
        isinstance(raw.get("history"), dict)
        and "representative_cases" in raw["history"]
    ):
        legacy_context_limit = raw["history"].pop("representative_cases")
        if type(legacy_context_limit) is not int or legacy_context_limit < 1:
            raise ValueError(
                "legacy representative case limit must be a positive integer"
            )
        if "mutation_context" not in raw:
            raw["mutation_context"] = {"representative_cases": legacy_context_limit}
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
    if policy["evaluation"]["v_gate_mode"] not in ("fail_fast", "full_audit"):
        raise ValueError("v_gate_mode must be fail_fast or full_audit")
    if any(type(v) is not int or v < 1 for v in policy["mutation_context"].values()):
        raise ValueError("mutation context limits must be positive integers")
    gate = policy["statistical_gate"]
    for value in (
        gate["confidence"],
        gate["max_harmfulness"],
        gate["max_stuck_delta"],
        gate["preservation_margin"],
        policy["opponent_replay"]["current_weight"],
    ):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("V2 numeric policies must be finite numbers")
    if not 0 <= gate["max_stuck_delta"] <= 1:
        raise ValueError("max_stuck_delta must be between zero and one")
    if (
        type(policy["evaluation"]["screen_clean_tasks"]) is not int
        or policy["evaluation"]["screen_clean_tasks"] < 1
    ):
        raise ValueError("screen_clean_tasks must be a positive integer")
    if not 0.5 < gate["confidence"] < 1 or not 0 <= gate["max_harmfulness"] <= 1:
        raise ValueError("invalid confidence/harmfulness policy")
    if gate["multiple_look_correction"] != "bonferroni":
        raise ValueError("unsupported multiple look correction")
    if gate["method"] not in ("task_block_bootstrap", "finite_panel_paired"):
        raise ValueError("unsupported gate method")
    if gate["preservation_margin"] != 0:
        raise ValueError(
            "opponent preservation cannot permit a native accuracy regression"
        )
    if policy["opponent_replay"]["current_weight"] <= 0:
        raise ValueError("current replay weight must be positive")
    for value in policy["skill_budgets"].values():
        if type(value) is not int or value < 1:
            raise ValueError("skill budgets must be positive integers")
    policy["skill_budgets"]["max_active_skills"] = (
        policy["skill_budgets"]["max_skills"]
        if policy["service_skill_runtime"] == "render_all_v1"
        else policy["max_active_service_skills"]
    )
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
    if legacy:
        policy["legacy_candidate_budget_migration"] = {
            "candidates_per_cluster": count,
            "max_clusters_per_generation": clusters,
            "candidates_per_generation": count * clusters,
        }
    if legacy_context_limit is not None:
        policy["legacy_context_migration"] = {
            "history.representative_cases": legacy_context_limit,
            "mutation_context.representative_cases": policy["mutation_context"][
                "representative_cases"
            ],
        }
    return policy


MUTATION_TYPES = V2_MUTATION_TYPES
