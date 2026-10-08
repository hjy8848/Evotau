"""Native paired morphology measurement; never runs alternating search or V/H."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from inferai_transport_pacing import SlidingWindowPacer

from evotau.alternating import (
    LLMAlternatingEvolvers,
    _run_panel,
    _trajectory_context,
    _write_json_atomic,
    _write_json_once,
)
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import _api_usage_by_role, load_alternating_tasks
from evotau.budget import RequestBudget
from evotau.evolution_artifacts import EvolutionJournal
from evotau.evolution_candidates import V2Providers
from evotau.evolution_failures import episode_metrics, paired_failure_matrix
from evotau.evolution_gate import cheap_screen, evaluate_gate
from evotau.provider_diagnostics import response_metadata, safe_error
from evotau.records import EpisodeRecord
from evotau.service_skills import (
    ServiceSkillMemoryV2,
    ServiceSkillV2,
    SkillActivationSignature,
    validate_skill_budgets,
)
from evotau.strategies import PromptStrategy
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import (
    role_model_args_for_runtime,
    sha256_json,
    write_manifest_once,
)


def install_transport(output):
    original = httpx.Client.send
    pacer = SlidingWindowPacer(30, 60.1)
    lock = threading.Lock()

    def record(value):
        value["at"] = datetime.now(UTC).isoformat()
        with lock, (output / "actual-provider-http.jsonl").open("a") as handle:
            handle.write(json.dumps(value) + "\n")

    def send(client, request, *args, **kwargs):
        if request.url.host != "inferaiapi.com" or request.method != "POST":
            return original(client, request, *args, **kwargs)
        body = json.loads(request.content)
        wait = pacer.acquire()
        selected = {
            k: body[k]
            for k in (
                "model",
                "thinking",
                "reasoning_effort",
                "temperature",
                "max_tokens",
                "stream",
            )
            if k in body
        }
        record(
            {
                "event": "request",
                "args": selected,
                "tools_count": len(body.get("tools") or []),
                "dispatch_wait_seconds": wait,
            }
        )
        if (
            body.get("model") == "qwen3.7-plus"
            and body.get("thinking", {}).get("type") != "disabled"
        ):
            raise RuntimeError("Qwen wire thinking-off condition violated")
        started = perf_counter()
        try:
            response = original(client, request, *args, **kwargs)
            if body.get("stream") is True:
                # Let the SDK consume SSE incrementally. The budget observer
                # records final usage/finish metadata after full collection.
                record({"event": "response_headers", "model": body.get("model"),
                        "http_status": response.status_code, "stream": True,
                        "elapsed_seconds": perf_counter() - started})
                return response
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            metadata = response_metadata(payload)
            record(
                {
                    "event": "response",
                    "model": body.get("model"),
                    "http_status": response.status_code,
                    "metadata": metadata,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            if (
                response.is_success
                and body.get("model") == "qwen3.7-plus"
                and (
                    (metadata.get("reasoning_tokens") or 0) > 0
                    or (metadata.get("reasoning_content_chars") or 0) > 0
                )
            ):
                raise RuntimeError(
                    "Qwen response violates frozen thinking-off condition"
                )
            return response
        except BaseException as error:
            record(
                {
                    "event": "failure",
                    "error": safe_error(error),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
            raise

    httpx.Client.send = send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = str(args.tau2_data_dir)
    raw = yaml.safe_load(args.config.read_text())
    protocol = raw["measurement_protocol"]
    source = ROOT / protocol["historical_source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol["historical_sha256"]:
        raise ValueError("historical repair source changed")
    historic = json.loads(source.read_text())
    customer = PromptStrategy(historic["frozen_customer"]["strategy"])
    configs = {}
    manifests = {}
    for condition in protocol["conditions"]:
        config = deepcopy(raw)
        exp = config["experiment"]
        exp["id"] += "-" + condition
        exp["output_path"] = "experiments/runs/" + exp["id"]
        exp["customer_strategy"] = customer.text
        mode = "render_all_v1" if condition == "render-all" else "activate_topk_v2"
        exp["evolution"]["service_skill_runtime"] = mode
        exp["skill_evolution_v2"]["service_skill_runtime"] = mode
        configs[condition] = config
        candidate_manifest = AlternatingManifest.from_mapping(config)
        saved_manifest = ROOT / exp["output_path"] / "manifest.json"
        if saved_manifest.exists():
            candidate_manifest = candidate_manifest.bind_saved_provenance(
                json.loads(saved_manifest.read_text())
            )
        manifests[condition] = candidate_manifest
    manifest = manifests["empty"]
    tasks = load_alternating_tasks(
        manifest, args.tau2_data_dir, include_validation=False, include_heldout=False
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": list(tasks),
                    "seeds": protocol["seeds"],
                    "episodes": protocol["planned_episodes"],
                    "V": False,
                    "H": False,
                    "models": dict(manifest.role_models),
                }
            )
        )
        return
    key = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-s",
            "inferaiapi.com/v1",
            "-a",
            "openai-api-key",
            "-w",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not key:
        raise RuntimeError("InferAI Keychain credential unavailable")
    os.environ["OPENAI_API_KEY"] = key
    del key
    output = ROOT / raw["experiment"]["output_path"]
    output.mkdir(parents=True, exist_ok=True)
    master_sha = sha256_json(
        {
            "config": raw,
            "conditions": {k: v.to_document() for k, v in manifests.items()},
        }
    )
    journal = EvolutionJournal(output, master_sha)
    journal.freeze("protocol", raw, lambda: raw)
    # Freeze every treatment before requests, including not-yet-started conditions.
    for condition, current in manifests.items():
        condition_path = ROOT / current.output_path
        condition_path.mkdir(parents=True, exist_ok=True)
        if not (condition_path / "manifest.json").exists():
            write_manifest_once(condition_path / "manifest.json", current)
        _write_json_once(condition_path / "selected-config.json", configs[condition])
    if (output / "mechanism-smoke-result.json").exists():
        print("Measurement already complete; no requests dispatched.", flush=True)
        return
    install_transport(output)
    budget = RequestBudget(None)
    budget.enable_live_usage(output / "api-usage-live.json")
    role_args = role_model_args_for_runtime(manifest.role_model_args)
    providers = V2Providers(
        LLMAlternatingEvolvers(
            model=dict(manifest.role_models)["evolver"],
            model_args=role_args["evolver"],
            request_budget=budget,
            output_directory=output,
        )
    )
    started = perf_counter()
    records = {}
    budgets = {"evolution": budget}

    def progress(stage, status="running", **fields):
        _write_json_atomic(
            output / "progress.json",
            {
                "stage": stage,
                "status": status,
                "at": datetime.now(UTC).isoformat(),
                **fields,
            },
        )
        print(json.dumps({"stage": stage, "status": status, **fields}), flush=True)

    try:
        progress("provider_preflight")
        for role in ("agent", "evolver"):
            probe = V2Providers(
                LLMAlternatingEvolvers(
                    model=dict(manifest.role_models)[role],
                    model_args=role_args[role],
                    request_budget=budget,
                    output_directory=output,
                )
            )
            response = probe.call(
                'Return JSON only: {"ok":true}.',
                {"purpose": "minimal no-tools provider preflight"},
                "evotau_provider_preflight",
            )
            if response != {"ok": True}:
                raise ValueError("provider preflight did not return the requested JSON")
        empty = ServiceSkillMemoryV2()

        def condition_run(condition, memory):
            current = manifests[condition]
            path = ROOT / current.output_path
            path.mkdir(parents=True, exist_ok=True)
            _write_json_once(path / "selected-config.json", configs[condition])
            run_budget = RequestBudget(None)
            run_budget.enable_live_usage(path / "api-usage-live.json")
            budgets[condition] = run_budget
            runner = TauBenchEpisodeRunner(
                manifest=current,
                config=configs[condition],
                data_dir=args.tau2_data_dir,
                request_budget=run_budget,
                output_directory=path,
                task_objects=tasks,
            )
            rows = []
            for seed in protocol["seeds"]:
                progress(condition, seed=seed)
                frozen = journal.freeze(
                    condition + "-seed-" + str(seed),
                    {
                        "memory": memory.to_dict(),
                        "customer": customer.text,
                        "seed": seed,
                        "manifest_sha": current.sha256,
                    },
                    lambda seed=seed: [
                        r.to_dict()
                        for r in _run_panel(
                            runner,
                            task_ids=current.evolution_task_ids,
                            tasks=tasks,
                            seed=seed,
                            customer=customer,
                            service=memory,
                            panel_name="morphology-" + condition,
                            max_parallel_episodes=current.max_parallel_episodes,
                        )
                    ],
                )
                rows.extend(EpisodeRecord.from_dict(r) for r in frozen)
            usage = run_budget.api_usage_by_call_name()
            _write_json_once(
                path / "alternating-result.json",
                {
                    "schema_version": 3,
                    "status": "complete",
                    "experiment_id": current.experiment_id,
                    "manifest_sha256": current.sha256,
                    "measurement_only": True,
                    "condition": condition,
                    "generations": [],
                    "initial_service": memory.to_dict(),
                    "final_service": memory.to_dict(),
                    "initial_customer": customer.to_dict(),
                    "final_customer": customer.to_dict(),
                    "validation_evaluated": False,
                    "heldout_evaluated": False,
                    "native_success_rate": episode_metrics(rows)["accuracy"],
                    "provider_usage": run_budget.snapshot().to_dict(),
                    "api_usage_by_call_name": usage,
                    "api_usage_by_role": _api_usage_by_role(usage),
                },
            )
            return rows, runner

        records["empty"], runner = condition_run("empty", empty)
        old = historic["mutation"]["skill"]
        progress("narrow_activation_signature")
        context = {
            "native_policy": runner.service_policy_text,
            "fixed_trigger": old["trigger"],
            "fixed_guidance": old["guidance"],
            "previous_repair_observed_fixes": ["22", "80"],
            "previous_repair_observed_breaks": ["98", "4", "21", "35"],
            "observed_E": [
                {
                    "task_id": r.task_id,
                    "seed": r.seed,
                    "task_success": r.task_success,
                    "trajectory": _trajectory_context(runner.load_trajectory(r)),
                }
                for r in records["empty"]
            ],
        }
        prompt = 'You narrow only the activation boundary of a frozen reusable Service repair. Do not edit its trigger or guidance. Inspect observable passing and failing controls and retain the useful mechanism while excluding ordinary successful cases. Do not include task IDs, entities or hidden objectives in the signature. Return JSON only: {"activation_signature":{"positive_conditions":[],"negative_conditions":[],"interaction_phase":[]},"reason":"..."}. Conditions must be specific, observable and procedural; negative conditions override positives.'
        proposal = journal.freeze(
            "narrow-signature",
            context,
            lambda: providers.call(prompt, context, "evotau_service_skill_mutator"),
        )
        if set(proposal) != {"activation_signature", "reason"}:
            raise ValueError("invalid signature annotation")
        memory = ServiceSkillMemoryV2(
            (
                ServiceSkillV2(
                    "skill-0001",
                    old["trigger"],
                    old["guidance"],
                    SkillActivationSignature.from_mapping(
                        proposal["activation_signature"]
                    ),
                ),
            )
        )
        policy = json.loads(manifest.skill_evolution_v2_json)
        for current in manifests.values():
            validate_skill_budgets(
                memory, json.loads(current.skill_evolution_v2_json)["skill_budgets"]
            )
        validation = journal.freeze(
            "skill-semantics",
            {"skills": memory.to_dict(), "native_policy": runner.service_policy_text},
            lambda: providers.validate_skill(
                {
                    "skills": memory.to_dict(),
                    "native_policy": runner.service_policy_text,
                }
            ),
        )
        if not all(
            validation.get(k) is True
            for k in ("reusable", "policy_subordinate", "no_task_entities")
        ):
            raise ValueError("narrowed runtime skill semantic validation failed")
        for condition in ("render-all", "activation"):
            records[condition], _ = condition_run(condition, memory)
        usage = {}
        for current in budgets.values():
            for name, row in current.api_usage_by_call_name().items():
                target = usage.setdefault(name, {})
                for k in (
                    "calls",
                    "successes",
                    "failures",
                    "prompt_tokens",
                    "completion_tokens",
                    "usage_responses",
                    "usage_unavailable",
                    "total_elapsed_seconds",
                ):
                    target[k] = target.get(k, 0) + row.get(k, 0)
        for row in usage.values():
            row["average_elapsed_seconds"] = (
                row["total_elapsed_seconds"] / row["calls"] if row["calls"] else 0
            )
        comparisons = {}
        for condition in ("render-all", "activation"):
            matrix = paired_failure_matrix(
                records["empty"],
                records[condition],
                generation=0,
                candidate_id=condition,
            )
            comparisons[condition] = {
                "metrics": episode_metrics(records[condition]),
                "matrix": matrix,
                "fixed_cells": sum(c["status"] == "FIXED" for c in matrix),
                "broken_cells": sum(c["status"] == "BROKEN" for c in matrix),
                "gate": evaluate_gate(
                    records["empty"],
                    records[condition],
                    policy["statistical_gate"],
                    objective="superiority",
                    smoke=True,
                ),
                "screen": cheap_screen(
                    records["empty"], records[condition], {"22", "80"}, set()
                ),
            }
        result = {
            "status": "complete",
            "measurement_only": True,
            "master_sha256": master_sha,
            "protocol": protocol,
            "baseline_metrics": episode_metrics(records["empty"]),
            "comparisons": comparisons,
            "same_guidance_and_trigger": True,
            "proposed_memory": memory.to_dict(),
            "api_usage_by_call_name": usage,
            "api_usage_by_role": _api_usage_by_role(usage),
            "provider_calls": sum(b.snapshot().attempts for b in budgets.values()),
            "prompt_tokens": sum(b.snapshot().prompt_tokens for b in budgets.values()),
            "completion_tokens": sum(
                b.snapshot().completion_tokens for b in budgets.values()
            ),
            "current_process_wall_seconds": perf_counter() - started,
            "completed_episodes": sum(len(r) for r in records.values()),
            "V_evaluated": False,
            "H_evaluated": False,
            "no_generalization_claim": True,
        }
        _write_json_once(output / "mechanism-smoke-result.json", result)
        progress("complete", status="complete", provider_calls=result["provider_calls"])
    except BaseException as error:
        progress(
            "failed",
            status="failed",
            failure_type=type(error).__name__,
            failure_message=safe_error(error),
        )
        _write_json_atomic(
            output / "failure.json",
            {
                "error": safe_error(error),
                "provider_usage": {
                    k: b.snapshot().to_dict() for k, b in budgets.items()
                },
                "traceback": safe_error(traceback.format_exc()),
            },
        )
        raise


if __name__ == "__main__":
    main()
