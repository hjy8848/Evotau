"""RC-Bandit dry validation/scripted examples ONLY; no credentials or provider calls."""

import argparse
import hashlib
import json
from pathlib import Path

from evotau.alternating import _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.customer_skills import CustomerSkill
from evotau.evolution_archive import BanditTrialLedger
from evotau.phase0 import load_config
from evotau.records import EpisodeRecord, EpisodeStatus, customer_strategy_id
from evotau.repair_conditioned_bandit import (
    ARMS,
    PROTOCOL,
    choose_arm,
    new_state,
    record_trial,
)
from evotau.repair_feedback import (
    discovery_feedback,
    paired_observations,
    review_context,
    service_pair,
)
from evotau.service_skills import ServiceSkillMemoryV2, apply_v2_mutation
from evotau.tau_provenance import sha256_json, write_manifest_once


def scripted_example(manifest, config):
    """Synthetic allocation/transition proof, not τ episodes or effectiveness evidence."""
    settings = json.loads(manifest.skill_evolution_v2_json)["repair_conditioned_bandit"]
    state = new_state(settings)
    root = Path(manifest.output_path)
    ledger = BanditTrialLedger(root, manifest.sha256, manifest.evolution_task_ids)
    customer = CustomerSkill.from_mapping(
        {
            "schema_version": 1,
            "mechanism": "scripted scope clarification",
            "trigger": "An original request has a conditional authorization.",
            "procedure": [
                "State the original condition truthfully.",
                "Confirm only after the condition is explained.",
            ],
            "intensity": "low",
            "stop_conditions": ["Original goals are resolved."],
            "hypothesis": "Scripted evidence fixture only.",
            "evidence_refs": [],
        }
    )
    mutation = {
        "operation": "add",
        "target_skill_id": None,
        "skill": {
            "trigger": "Original authorization is conditional.",
            "guidance": "Check the stated condition before acting.",
            "activation_signature": {
                "positive_conditions": ["User states a condition."],
                "negative_conditions": ["No condition is stated."],
                "interaction_phase": ["before_confirmation"],
            },
        },
    }
    before = ServiceSkillMemoryV2()
    after = apply_v2_mutation(before, mutation, next_skill_id_number=1)
    pair = service_pair(before, after, generation=0, promoted=True)
    a, b, rows, validity = [], [], {"before": [], "after": []}, {}
    classes = {}
    for index, task in enumerate(manifest.evolution_task_ids):
        success = [(False, True), (False, False), (True, False), (True, True)][
            min(index, 3)
        ]
        classes[task] = ["repaired", "residual", "regression", "stable"][min(index, 3)]
        for seed in settings["feedback_seeds"]:
            for side, records, ok in (
                ("before", a, success[0]),
                ("after", b, success[1]),
            ):
                path = f"scripted-trajectories/{side}-{task}-{seed}.json"
                record = EpisodeRecord(
                    f"scripted-{side}-{task}-{seed}",
                    task,
                    seed,
                    customer_strategy_id(customer.compile()),
                    pair[side + "_id"],
                    EpisodeStatus.COMPLETE,
                    ok,
                    native_reward=float(ok),
                    termination_reason="scripted",
                    trajectory_ref=path,
                )
                records.append(record)
                message = {
                    "role": "assistant",
                    "content": "Scripted action evidence; NOT native execution.",
                }
                user_message = {
                    "role": "user",
                    "content": "Please help with my original request (scripted).",
                }
                projected_user = {
                    **user_message,
                    "evidence_ref": {
                        "projected_message_index": 0,
                        "message_sha256": sha256_json(user_message),
                    },
                }
                projected = {
                    **message,
                    "evidence_ref": {
                        "projected_message_index": 1,
                        "message_sha256": sha256_json(message),
                    },
                }
                rows[side].append(
                    {
                        "task": {"task_id": task},
                        "seed": seed,
                        "trajectory_ref": path,
                        "trajectory": {"messages": [projected_user, projected]},
                    }
                )
                _write_json_once(
                    root / path,
                    {
                        "scripted": True,
                        "real_provider_calls": 0,
                        "messages": [user_message, message],
                        "record": record.to_dict(),
                    },
                )
    for side, side_rows in rows.items():
        validity[side] = {
            "status": "valid",
            "cells": [
                {
                    "task_id": row["task"]["task_id"],
                    "seed": row["seed"],
                    "status": "valid",
                    "reason": "scripted mock review",
                    "evidence_message_indices": [0],
                }
                for row in side_rows
            ],
        }
    obs = paired_observations(
        a,
        b,
        pair=pair,
        customer_id=customer_strategy_id(customer.compile()),
        task_ids=manifest.evolution_task_ids,
        seeds=settings["feedback_seeds"],
        validity=validity,
    )
    documents, prior = [], []
    for g in range(2):
        choices = []
        for i in range(settings["max_trials_per_generation"]):
            choice = choose_arm(state, pair["pair_id"], settings)
            ctx = review_context(obs, pair, rows, prior_discoveries=prior)
            task = manifest.evolution_task_ids[1 if choice["arm"] in ARMS[:2] else 2]
            report = {
                "discoveries": [
                    {
                        "mechanism": "scripted condition loss " + classes[task],
                        "task_id": task,
                        "discovery_type": classes[task],
                        "seeds": settings["feedback_seeds"],
                        "evidence_refs": [
                            m["evidence_ref"]
                            for r in ctx["evidence"]
                            if r["task_id"] == task
                            for m in r["messages"]
                        ],
                        "repair_related": True,
                        "novel": True,
                        "reason": "SCRIPTED support, not a real reviewer.",
                    }
                ]
            }
            keys = [
                key
                for t in state["trials"]
                for key in t["feedback"].get("discovery_keys", [])
            ]
            fb = discovery_feedback(
                obs,
                ctx,
                report,
                reviewer_provenance={
                    "model": "scripted/no real provider",
                    "response_ref": "scripted",
                    "input_sha256": sha256_json(ctx),
                    "response_sha256": sha256_json(report),
                },
                prior_keys=keys,
                min_replications=settings["min_replications"],
            )
            trial = {
                "protocol_version": PROTOCOL,
                "trial_id": f"g{g:04d}-c{i:04d}",
                "arm": choice["arm"],
                "service_pair_id": pair["pair_id"],
                "candidate_validity": "valid",
                "reward": fb["reward"],
                "feedback": fb,
                "procedure_id": customer.procedure_id,
                "choice": choice,
                "observations": obs,
                "cost": {
                    "api_calls": 0,
                    "native_episodes": 0,
                    "simulated_episode_units": 2 * len(a),
                    "simulated_reviewer_calls": 1,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                },
            }
            ledger.publish(trial["trial_id"], "final", trial)
            state = record_trial(state, trial)
            choices.append(
                {
                    "arm": choice["arm"],
                    "reward": fb["reward"],
                    "scores_before": choice["scores"],
                }
            )
            prior.extend(report["discoveries"])
        documents.append({"generation": g, "scripted": True, "choices": choices})
    result = {
        "scripted": True,
        "no_real_provider": True,
        "real_provider_calls": 0,
        "real_native_episodes_started": 0,
        "interpretation": "allocation/transition fixture, not alternating effectiveness",
        "manifest_sha256": manifest.sha256,
        "config_sha256": sha256_json(config),
        "pair": pair,
        "raw_observations": obs,
        "generations": documents,
        "state": state,
    }
    _write_json_once(root / "rc-scripted-result.json", result)
    return {
        "output": str(root),
        "scripted": True,
        "trials": len(state["trials"]),
        "real_provider_calls": 0,
        "classifications": classes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/airline-rc-bandit-v1-e10-dryrun.yaml"
    )
    parser.add_argument("--tau2-data-dir", required=True)
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = AlternatingManifest.from_mapping(config)
    if (
        manifest.real_provider_enabled
        or manifest.run_validation
        or manifest.run_heldout
    ):
        raise ValueError("offline RC launcher cannot start real providers or V/H")
    for source, digest in config["launch_readiness"]["execution_source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen offline launcher source changed")
    if args.scripted:
        path = Path(manifest.output_path) / "manifest.json"
        if path.exists():
            manifest = manifest.bind_saved_provenance(json.loads(path.read_text()))
        else:
            write_manifest_once(path, manifest)
    tasks = load_alternating_tasks(
        manifest, args.tau2_data_dir, include_validation=False, include_heldout=False
    )
    print(
        json.dumps(
            {
                "task_ids": list(tasks),
                "protocol": PROTOCOL,
                "real_provider_calls": 0,
                "result": scripted_example(manifest, config)
                if args.scripted
                else "dry validated",
            }
        )
    )


if __name__ == "__main__":
    main()
