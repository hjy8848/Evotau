"""Source-path cost allocation and historical usage calibration; no calls/H content."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import yaml


def plan(config, split_path, results_root):
    raw = yaml.safe_load(Path(config).read_text())
    exp = raw["experiment"]
    splits = json.loads(Path(split_path).read_text())
    panels = exp["task_selection"]
    names = ("evolution", "validation", "heldout")
    groups = [set(panels[name]) for name in names]
    excluded = set(panels["excluded"])
    for name, ids in zip(names, groups, strict=True):
        if (
            len(ids) != len(panels[name])
            or ids & excluded
            or not ids <= set(splits["test" if name == "heldout" else "train"])
        ):
            raise ValueError("invalid panel")
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("overlapping panels")
    policy = exp["skill_evolution_v2"]
    cs = exp["customer_candidates"]
    ev = policy["service_evolution"]
    candidate_bound = ev["candidates_per_cluster"] * ev[
        "max_clusters_per_generation"
    ] + int(ev["crossover"])
    seed_union = set(policy["evaluation"]["screen_seeds"]) | set(
        policy["evaluation"]["gate_seeds"]
    )
    replay = policy["opponent_replay"]
    rows = []
    for g in range(exp["generations"]):
        opponents = 1 + (
            min(replay["archived_customers"], (g + 1) * cs)
            + int(replay["include_native_customer"])
            if replay["enabled"]
            else 0
        )
        # Same incumbent and candidate conditions are cached across Screen/full-E;
        # scores are never shared across Customer/Service/task/seed/manifest changes.
        rows.append(
            {
                "generation": g,
                "customer_fitness_episode_allocation": (1 + cs) * len(groups[0]),
                "screen_and_full_E_union_allocation": (1 + candidate_bound)
                * len(groups[0])
                * len(seed_union),
                "V_opponent_allocation": (1 + candidate_bound)
                * len(groups[1])
                * len(policy["evaluation"]["gate_seeds"])
                * opponents,
                "opponent_bound": opponents,
            }
        )
    h = 4 * len(groups[2]) * len(policy["evaluation"]["gate_seeds"])
    upper = (
        sum(
            r["customer_fitness_episode_allocation"]
            + r["screen_and_full_E_union_allocation"]
            + r["V_opponent_allocation"]
            for r in rows
        )
        + h
    )
    # All archived Qwen native episode records, unique by native condition + trajectory digest.
    evidence, observations = [], {}
    for path in sorted(Path(results_root).glob("**/run-telemetry.json")):
        key = path.parent / "episode-record.json"
        if not key.exists():
            continue
        data = json.loads(path.read_text())
        usage = data.get("episode_budget_delta", {})
        models = usage.get("model_usage", [])
        if not models or any(r["model_id"] != "openai/qwen3.7-plus" for r in models):
            continue
        calls = usage.get("attempts", 0)
        if not calls:
            continue
        condition = data.get("episode_key_sha256")
        if condition not in observations:
            observations[condition] = {
                "calls": calls,
                "prompt": usage["prompt_tokens"],
                "completion": usage["completion_tokens"],
            }
            evidence.append(str(path))
    values = list(observations.values())
    stats = {}
    for label, key in [
        ("calls_per_episode", "calls"),
        ("prompt_tokens_per_episode", "prompt"),
        ("completion_tokens_per_episode", "completion"),
    ]:
        ordered = sorted(v[key] for v in values)
        stats[label] = {
            "min": min(ordered),
            "mean": mean(ordered),
            "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
            "max": max(ordered),
        }
    totals = {k: sum(v[k] for v in values) for k in ("calls", "prompt", "completion")}
    # Ordinary legal branches, not outcome-optimized estimates; upper branch is all
    # proposals passing Screen/E and invoking all V replay checks. Extra costs from
    # failed attempts are governed by the approved global cap, never hidden retries.
    scenarios = [
        (
            "no skill proposals; Customer rejected; unchanged Gen1 incumbent reused",
            20,
            2 * len(groups[2]) * 4,
        ),
        (
            "valid Customer candidates; no promotion",
            20 + 20 + 20,
            2 * len(groups[2]) * 4,
        ),
        ("one screened proposal per generation plus replay", 568, h),
        ("all candidate paths allocation bound", upper - h, h),
    ]
    estimates = []
    for label, evolution, endpoint in scenarios:
        episodes = evolution + endpoint
        estimates.append(
            {
                "branch": label,
                "episode_allocation": episodes,
                "calls_at_historical_mean": math.ceil(
                    episodes * stats["calls_per_episode"]["mean"]
                )
                + 40,
                "calls_with_50pct_activation_overhead": math.ceil(
                    episodes * stats["calls_per_episode"]["mean"] * 1.5
                )
                + 40,
                "calls_at_66_per_episode_allowance": episodes * 66 + 40,
                "prompt_at_historical_mean": math.ceil(
                    episodes * stats["prompt_tokens_per_episode"]["mean"]
                )
                + 2000000,
                "completion_at_historical_mean": math.ceil(
                    episodes * stats["completion_tokens_per_episode"]["mean"]
                )
                + 100000,
                "prompt_at_historical_p90": episodes
                * stats["prompt_tokens_per_episode"]["p90"]
                + 2000000,
                "completion_at_historical_p90": episodes
                * stats["completion_tokens_per_episode"]["p90"]
                + 100000,
            }
        )
    return {
        "per_generation": rows,
        "candidate_bound": candidate_bound,
        "max_H_episode_conditions": h,
        "total_episode_allocation_bound": upper,
        "historical_unique_Qwen_episode_count": len(values),
        "historical_episode_statistics": stats,
        "historical_totals": totals,
        "evidence": evidence,
        "scenarios": estimates,
        "proposed_request_cap": exp["request_budget_cap"],
        "budget_confirmed": False,
        "allocation_assumptions": [
            "Shared Screen/full-E seed union and old-condition cache; no repeats for identical endpoints.",
            "66 calls/episode = allowance for32 runtime +32 activator +2 evaluator calls; not a certified native API ceiling.",
            "40 GPT calls +2M input/100k output tokens allocation, variable context and reasoning; no output truncation.",
            "The historical sample has empty-catalog/V1 conditions; nonempty V2 memory adds activation calls. A50pct call overhead is a planning scenario, not measured treatment latency.",
            "Retries/recovery billed attempts included in global cap; malformed output and network errors stop immediately.",
            "Request cap is hard; token estimates are NOT hard money/token caps. No current credential-group price evidence; monetary quote unavailable.",
            "Serial pacing30/60.1s implies at least2.0seconds per Chinese-model call at sustained throughput; latency/GPT reasoning add time.",
        ],
        "H_task_content_loaded": False,
        "real_provider_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-metadata", type=Path, required=True)
    parser.add_argument(
        "--results-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.config, args.split_metadata, args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
