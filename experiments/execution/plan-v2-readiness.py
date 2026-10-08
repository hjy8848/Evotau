"""Offline panel/power/call planning; reads split metadata, never task contents."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

import yaml


def plan(config, split_path):
    raw = yaml.safe_load(Path(config).read_text())
    exp = raw["experiment"]
    split_bytes = Path(split_path).read_bytes()
    splits = json.loads(split_bytes)
    selection = exp["task_selection"]
    panels = {name: list(map(str, selection[name]))
              for name in ("evolution", "validation", "heldout")}
    excluded = set(map(str, selection["excluded"]))
    sets = {name: set(ids) for name, ids in panels.items()}
    for name, ids in panels.items():
        if len(ids) != len(sets[name]) or sets[name] & excluded:
            raise ValueError(f"duplicate/excluded IDs in {name}")
        expected = "test" if name == "heldout" else "train"
        if not sets[name] <= set(map(str, splits[expected])):
            raise ValueError(f"wrong split for {name}")
    names = list(sets)
    if any(sets[a] & sets[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError("E/V/H overlap")
    allowed_v = sorted(set(map(str, splits["train"])) - sets["evolution"] - excluded,
                       key=int)
    selection_seed = raw["readiness_review"]["panel_selection_seed"]
    reproduced = random.Random(selection_seed).sample(allowed_v, len(panels["validation"]))
    if reproduced != panels["validation"]:
        raise ValueError("V selection does not reproduce the frozen seed/sample")

    v2 = exp["skill_evolution_v2"]
    service_candidates = (v2["service_evolution"]["candidates_per_cluster"]
                          * v2["service_evolution"]["max_clusters_per_generation"]
                          + int(v2["service_evolution"]["crossover"]))
    seeds = len(v2["evaluation"]["gate_seeds"])
    screen_seeds = len(v2["evaluation"]["screen_seeds"])
    # Deliberately overcounts old conditions and screen/full-E overlap. Actual cache
    # reuse and semantic/screen/repair rejection usually reduce this substantially.
    opponent_bound = (1 + (v2["opponent_replay"]["archived_customers"]
                          + int(v2["opponent_replay"]["include_native_customer"])
                          if v2["opponent_replay"]["enabled"] else 0))
    ne, nv, nh = (len(panels[name]) for name in names)
    generation_upper = {
        "customer_fitness": (1 + exp["customer_candidates"]) * ne,
        "screen": 2 * service_candidates * ne * screen_seeds,
        "full_E_repair": (1 + service_candidates) * ne * seeds,
        "V_opponent_preservation": ((1 + service_candidates) * nv * seeds
                                    * opponent_bound if exp["run_validation"] else 0),
    }
    endpoint_upper = 4 * nh * seeds if exp["run_heldout"] else 0
    episode_upper = sum(generation_upper.values()) * exp["generations"] + endpoint_upper
    zsum = NormalDist().inv_cdf(.975) + NormalDist().inv_cdf(.8)
    stats = [{
        "independent_tasks": n,
        "zero_regression_one_sided_95_upper": 1 - .05 ** (1 / n),
        "approx_80pct_power_MDE_pp_q_0_2": 100 * zsum * math.sqrt(.2 / n),
    } for n in (3, 5, nv, nh)]
    return {
        "config": str(Path(config).resolve()),
        "split_metadata_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "panels": panels,
        "train_tasks": len(splits["train"]),
        "test_tasks": len(splits["test"]),
        "max_disjoint_validation_tasks_after_E_exclusions": len(allowed_v),
        "V_sampling_seed": selection_seed,
        "V_sampling_population_order": "numerically sorted string IDs",
        "H_is_complete_test_split": sets["heldout"] == set(map(str, splits["test"])),
        "H_task_contents_loaded": False,
        "statistics": stats,
        "statistical_assumptions": {
            "risk_bound": "optimistic independent task Bernoulli risk; every task at risk; no multiplicity correction",
            "MDE": "normal paired planning approximation, two-sided alpha=.05, power=.8, discordance q=.2; not empirical power",
            "task_population": "fixed Retail benchmark, not a random sample of real customers/domains",
            "repeated_seeds": "within-task repeats; do not multiply independent task count",
        },
        "minimum_independent_zero_event_blocks_for_5pct_risk": {
            "exact_one_sided_95": math.ceil(math.log(.05) / math.log(.95)),
            "unadjusted_two_sided_95_Wilson": math.ceil(NormalDist().inv_cdf(.975) ** 2 * 19),
        },
        "planning_episode_upper": {
            "per_generation": generation_upper,
            "opponent_bound": opponent_bound,
            "service_candidates_bound": service_candidates,
            "final_H_native_and_fresh": endpoint_upper,
            "total": episode_upper,
            "interpretation": "conservative allocation bound, not predicted completed or billed episodes",
        },
        "planning_calls_at_30_per_episode": episode_upper * 30,
        "calls_upper_not_certified": "max_steps is orchestrator steps, not total API calls; activator/evaluator/evolver add calls",
        "request_cap_proposal": exp["request_budget_cap"],
        "budget_confirmed": raw["readiness_review"]["budget_confirmed"],
        "real_provider_enabled": exp["real_provider_enabled"],
        "real_requests_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = plan(args.config, args.split_metadata)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
