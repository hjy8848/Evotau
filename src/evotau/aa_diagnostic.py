"""Independent native baseline/A-A repetitions; no evolution, V or H content."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .alternating_manifest import AlternatingManifest
from .alternating_run import load_alternating_tasks
from .budget import RequestBudget
from .evolution_failures import episode_metrics
from .phase0 import load_config
from .provider_diagnostics import safe_error
from .service_skills import ServiceSkillMemoryV2
from .strategies import PromptStrategy
from .tau_episode_runner import TauBenchEpisodeRunner
from .tau_provenance import sha256_json


def _message(message):
    return {
        k: message.get(k)
        for k in ("role", "content", "tool_calls", "tool_call_id", "name", "recipient")
        if k in message
    }


def compare_trajectories(left, right):
    """Exact observable-message comparison; excludes timing/cost metadata."""
    a = [_message(m) for m in left["messages"]]
    b = [_message(m) for m in right["messages"]]
    first = lambda rows: next((m for m in rows if m.get("role") == "user"), None)
    divergence = next(
        (
            i
            for i in range(max(len(a), len(b)))
            if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)
        ),
        None,
    )

    def behavior(message):
        value = deepcopy(message)
        value.pop("tool_call_id", None)
        for call in value.get("tool_calls") or []:
            call.pop("id", None)
        return value

    x, y = list(map(behavior, a)), list(map(behavior, b))
    behavioral_divergence = next(
        (
            i
            for i in range(max(len(x), len(y)))
            if (x[i] if i < len(x) else None) != (y[i] if i < len(y) else None)
        ),
        None,
    )
    return {
        "first_behavioral_divergence_native_message_index": behavioral_divergence,
        "first_customer_message_equal": first(a) == first(b),
        "first_divergence_native_message_index": divergence,
        "termination_before": left.get("termination_reason"),
        "termination_after": right.get("termination_reason"),
        "native_reward_info_before": left.get("reward_info"),
        "native_reward_info_after": right.get("reward_info"),
        "causal_skill_effect_identified": False,
    }


def run_independent_repetitions(
    config,
    *,
    data_dir,
    output,
    tasks,
    seeds,
    repetitions,
    approved_cap,
    stop_before_next_episode_file=None,
    runner_factory=TauBenchEpisodeRunner,
    task_loader=load_alternating_tasks,
):
    """Separate manifest/output/cache per replicate; resume only within that replicate.

    One budget spans all repetitions. Completed conditions survive explicit resume;
    no fresh identity is generated on resume, and no historical score is imported.
    """
    original = AlternatingManifest.from_mapping(config)
    if repetitions < 1 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("positive repetitions and unique seeds required")
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or not set(tasks) <= set(original.evolution_task_ids)
    ):
        raise ValueError("baseline/A-A loads E tasks only")
    if (
        original.request_budget_cap is None
        or approved_cap != original.request_budget_cap
    ):
        raise ValueError("explicit finite approved cap must match configuration")
    output = Path(output)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("diagnostic output must be an independent relative directory")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": sha256_json(config),
        "tasks": list(tasks),
        "seeds": list(seeds),
        "repetitions": repetitions,
        "approved_cap": approved_cap,
        "output": output.as_posix(),
    }
    binding = output / "diagnostic-config.json"
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("cannot change frozen A/A conditions on resume")
    binding.write_text(json.dumps(identity, indent=2))
    if runner_factory is TauBenchEpisodeRunner:
        from tau2.utils import llm_utils

        if llm_utils.LLM_CACHE_ENABLED or llm_utils.litellm.cache is not None:
            raise ValueError(
                "independent A/A requires disabled client completion cache"
            )
    budget = RequestBudget(approved_cap)
    budget.enable_live_usage(output / "api-usage-live.json")
    results, simulations = [], []
    for repetition in range(repetitions):
        raw = deepcopy(config)
        exp = raw["experiment"]
        exp.update(
            id=f"{original.experiment_id}-baseline-replicate-{repetition}",
            output_path=str(output / f"replicate-{repetition}"),
            checkpoint_path=str(output / f"replicate-{repetition}-checkpoint.json"),
            run_validation=False,
            run_heldout=False,
        )
        # Keep domain/task splits/models/steps frozen; repetition identity is outside service identity.
        manifest = AlternatingManifest.from_mapping(raw)
        loaded = task_loader(
            manifest, data_dir, include_validation=False, include_heldout=False
        )
        runner = runner_factory(
            manifest=manifest,
            config=raw,
            data_dir=data_dir,
            request_budget=budget,
            output_directory=exp["output_path"],
            task_objects={t: loaded[t] for t in tasks},
            stop_before_next_episode_file=stop_before_next_episode_file,
        )
        rows, native = {}, {}
        try:
            for seed in seeds:
                for task in tasks:
                    record = runner(
                        task_id=task,
                        seed=seed,
                        customer=PromptStrategy(original.initial_customer_strategy),
                        service=ServiceSkillMemoryV2(),
                        panel_name=f"independent-baseline-{repetition}",
                    )
                    key = (task, seed)
                    rows[key] = record
                    file = Path(exp["output_path"]) / record.trajectory_ref
                    native[key] = json.loads(file.read_text())
        except Exception as error:
            (output / "diagnostic-failure.json").write_text(
                json.dumps(
                    {
                        "repetition": repetition,
                        "stage": "native_episode",
                        "error_type": type(error).__name__,
                        "message": safe_error(error),
                        "completed_in_repetition": len(rows),
                    },
                    indent=2,
                )
            )
            raise
        results.append(rows)
        simulations.append(native)
    pairs = []
    for repetition in range(1, repetitions):
        for key, before in results[0].items():
            after = results[repetition][key]
            pairs.append(
                {
                    "task_id": key[0],
                    "seed": key[1],
                    "baseline_repetition": 0,
                    "comparison_repetition": repetition,
                    "success_before": before.task_success,
                    "success_after": after.task_success,
                    "outcome_flip": before.task_success != after.task_success,
                    "trajectory_before": f"replicate-0/{before.trajectory_ref}",
                    "trajectory_after": f"replicate-{repetition}/{after.trajectory_ref}",
                    "activated_skill_ids_before": list(before.activated_skill_ids),
                    "activated_skill_ids_after": list(after.activated_skill_ids),
                    **compare_trajectories(
                        simulations[0][key], simulations[repetition][key]
                    ),
                }
            )
    report = {
        "schema_version": 1,
        "conditions": identity,
        "episode_count": len(tasks) * len(seeds) * repetitions,
        "replicate_metrics": [episode_metrics(list(rows.values())) for rows in results],
        "comparisons": pairs,
        "skill_memory": "empty identical baseline in every replicate",
        "inference_scope": "specified E cells only; paired rollouts do not establish causal skill benefit",
    }
    (output / "aa-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tau2-data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--approved-request-cap", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-before-next-episode-file")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = AlternatingManifest.from_mapping(config)
    tasks = args.tasks or list(manifest.evolution_task_ids)
    if not args.execute:
        load_alternating_tasks(
            manifest,
            args.tau2_data_dir,
            include_validation=False,
            include_heldout=False,
        )
        print(
            json.dumps(
                {
                    "episode_count": len(tasks) * len(args.seeds) * args.repetitions,
                    "domain": manifest.domain,
                    "tasks": tasks,
                    "real_requests_started": False,
                }
            )
        )
        return
    from .release_recovery import frozen_run_lock

    with frozen_run_lock(Path(args.output) / "diagnostic-checkpoint.json"):
        report = run_independent_repetitions(
            config,
            data_dir=args.tau2_data_dir,
            output=args.output,
            tasks=tasks,
            seeds=args.seeds,
            repetitions=args.repetitions,
            approved_cap=args.approved_request_cap,
            stop_before_next_episode_file=args.stop_before_next_episode_file,
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
