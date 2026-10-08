"""Replay representative V2 research requests; no native episodes or retries."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evotau.alternating import LLMAlternatingEvolvers, _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.budget import RequestBudget
from evotau.evolution_candidates import V2Providers, validate_mutation
from evotau.provider_diagnostics import safe_error
from evotau.service_skills import ServiceSkillMemoryV2, apply_v2_mutation
from evotau.tau_provenance import role_model_args_for_runtime, sha256_json


def credential_binding(model_args):
    """Select the existing credential group without a cross-group fallback."""
    if model_args.get("api_protocol") == "responses":
        return "openai-gpt-api-key", model_args.get("api_key_env", "INFERAI_API_KEY")
    return "openai-api-key", "OPENAI_API_KEY"


def fixtures(source, historical):
    inputs = [json.loads(p.read_text()) for p in (source / "evolver-calls").glob("*/input.json")]
    diagnosis = next(x for x in inputs if x["call_name"] == "evotau_service_diagnoser")
    customer = next(x for x in inputs if x["call_name"] == "evotau_customer_semantic_validator")
    context = diagnosis["context"]
    failed = [r["task_id"] for r in context["current_outcomes"] if r["task_success"] is False]
    passed = [r["task_id"] for r in context["current_outcomes"] if r["task_success"] is True]
    observed = {r["task"]["task_id"] for r in context["task_interactions"]}
    # A declared readiness fixture, not a claimed LLM diagnosis or fabricated reward.
    cluster = {
        "cluster_id": "readiness-fixture",
        "root_cause": "Inspect the supplied unresolved native failures and passing controls; determine whether a minimal reusable procedural repair is supported, otherwise NO_OP.",
        "evidence_task_ids": [t for t in failed if t in observed],
        "protected_success_task_ids": [t for t in passed if t in observed],
        "recommended_surface": "skill",
        "recommended_mutation_types": ["add", "no_op"],
        "risk": "A broad repair could break the supplied successful native cases.",
    }
    mutation = {
        **deepcopy(context), "root_cause_cluster": cluster,
        "proposal_bias": "narrow_applicability", "prior_fixed_cases": [],
        "prior_broken_cases": [], "archive_parent": None,
    }
    old = json.loads(historical.read_text())["mutation"]["skill"]
    validator = {
        "skills": {"carrier": "skill_memory_v2", "skills": [{
            "skill_id": "skill-0001", **old,
            "activation_signature": {
                "positive_conditions": [old["trigger"]],
                "negative_conditions": [], "interaction_phase": ["before_confirmation"],
            },
        }]},
        "native_policy": context["service_policy"],
    }
    return context, mutation, validator, customer["context"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    manifest = AlternatingManifest.from_mapping(raw)
    model = dict(manifest.role_models)["evolver"]
    model_args = role_model_args_for_runtime(manifest.role_model_args)["evolver"]
    account, environment = credential_binding(model_args)
    historical = ROOT / "experiments/runs/evotau-retail-skillmemory-v1-qwen3-7-plus-runtime-v4-pro-e20-g2-p4-20261007/generation-0000-service-proposal.json"
    contexts = fixtures(args.source, historical)
    names = ("diagnoser", "mutator", "skill_validator", "customer_validator")
    plan = {
        "model": model, "model_args": model_args, "request_cap": 4,
        "source": str(args.source), "fixture_hashes": dict(zip(names, map(sha256_json, contexts))),
        "context_chars": {name: len(json.dumps(ctx, ensure_ascii=False)) for name, ctx in zip(names, contexts)},
        "native_episodes": 0, "automatic_retries": 0,
        "runtime_provenance": manifest.to_document()["evotau"],
        "credential_binding": {"keychain_account": account, "environment_variable": environment,
                               "secret_in_artifacts": False},
        "interpretation": "Representative frozen native evidence; mutator uses a declared readiness-only cluster fixture, not an accepted diagnosis. No search fitness or runtime skills are imported.",
    }
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json_once(args.output / "plan.json", plan)
    key = subprocess.run(["security", "find-generic-password", "-s", "inferaiapi.com/v1", "-a", account, "-w"], capture_output=True, text=True, check=True).stdout.strip()
    if not key:
        raise RuntimeError("InferAI credential unavailable")
    os.environ[environment] = key
    del key
    spec = importlib.util.spec_from_file_location("morphology_transport", ROOT / "experiments/execution/run-v2-activation-morphology-smoke.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.install_transport(args.output)
    if model_args.get("api_protocol") == "responses":
        spec = importlib.util.spec_from_file_location("live_transport", ROOT / "experiments/execution/run-v2-live-evolution.py")
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        live.install_responses_observer(args.output)
    budget = RequestBudget(4)
    budget.enable_live_usage(args.output / "api-usage-live.json")
    provider = V2Providers(LLMAlternatingEvolvers(model=model, model_args=model_args, request_budget=budget, output_directory=args.output))
    functions = (provider.diagnose, provider.mutate, provider.validate_skill, provider.validate_customer)
    rows = []
    for name, context, function in zip(names, contexts, functions):
        started = perf_counter()
        try:
            result = function(context)
            if name == "mutator":
                validate_mutation(result)
                apply_v2_mutation(ServiceSkillMemoryV2(), result, next_skill_id_number=1)
            if name in ("skill_validator", "customer_validator"):
                flags = ("reusable", "policy_subordinate", "no_task_entities") if name == "skill_validator" else ("preserves_facts", "preserves_objective", "interaction_only", "no_benchmark_leakage")
                if set(result) != {*flags, "reason"} or any(type(result[k]) is not bool for k in flags) or not isinstance(result["reason"], str):
                    raise ValueError("validator returned invalid structured schema")
            row = {"stage": name, "status": "complete", "response": result}
        except Exception as exc:  # noqa: BLE001 - each independent audit probe records failure
            row = {"stage": name, "status": "failed", "error_type": type(exc).__name__, "error": safe_error(exc)}
        row["elapsed_seconds"] = perf_counter() - started
        rows.append(row)
        _write_json_once(args.output / (name + ".json"), row)
        print(json.dumps({k: v for k, v in row.items() if k != "response"}, ensure_ascii=False), flush=True)
    report = {"status": "complete" if all(r["status"] == "complete" for r in rows) else "failed", "stages": rows, "provider_usage": budget.snapshot().to_dict(), "native_episodes": 0}
    _write_json_once(args.output / "report.json", report)
    if report["status"] != "complete":
        raise SystemExit(2)  # A recorded failed probe must also fail a shell/CI readiness gate.


if __name__ == "__main__":
    main()
