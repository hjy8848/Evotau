"""Three connected GPT stage probes on historical native evidence, never fitness or rollout."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evotau.alternating import (
    LLMAlternatingEvolvers,
    _write_json_atomic,
    _write_json_once,
)
from evotau.alternating_manifest import AlternatingManifest
from evotau.budget import RequestBudget
from evotau.evolution_artifacts import EvolutionJournal
from evotau.evolution_candidates import V2Providers
from evotau.provider_diagnostics import safe_error
from evotau.service_skills import ServiceSkillMemoryV2, apply_v2_mutation
from evotau.tau_provenance import role_model_args_for_runtime, sha256_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    manifest = AlternatingManifest.from_mapping(raw)
    model_args = role_model_args_for_runtime(manifest.role_model_args)["evolver"]
    model = dict(manifest.role_models)["evolver"]
    if model != "openai/gpt-6.1-sol" or model_args.get("api_protocol") != "responses":
        raise ValueError("This explicit diagnostic uses only the configured InferAI GPT6 Responses route")
    inputs = [json.loads(p.read_text()) for p in (args.source / "evolver-calls").glob("*/input.json")]
    diagnosis = next(x["context"] for x in inputs if x["call_name"] == "evotau_service_diagnoser")
    mutation = next(x["context"] for x in inputs if x["call_name"] == "evotau_service_skill_mutator")
    plan = {
        "kind": "historical-evidence-connected-provider-diagnostic",
        "source": str(args.source), "config_sha256": sha256_json(raw),
        "runtime_provenance": manifest.to_document()["evotau"],
        "model": model, "model_args": model_args, "request_budget_cap": 3,
        "automatic_retries": 0, "native_episodes": 0, "fitness_computed": False,
        "score_imported": False, "heldout_loaded": False,
        "diagnosis_context_sha256": sha256_json(diagnosis),
        "mutation_template_sha256": sha256_json(mutation),
        "interpretation": "Historical source outcomes are diagnostic evidence only; no old score is imported into a new search condition. This probe cannot certify live Screen/Gate or efficacy.",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError("Choose a new probe directory; no automatic replay or retries")
    args.output.mkdir(parents=True)
    _write_json_once(args.output / "plan.json", plan)
    key = subprocess.run(["security", "find-generic-password", "-s", "inferaiapi.com/v1",
                          "-a", "openai-gpt-api-key", "-w"], capture_output=True, text=True, check=True).stdout.strip()
    if not key:
        raise RuntimeError("InferAI GPT credential unavailable")
    os.environ[model_args["api_key_env"]] = key
    del key
    spec = importlib.util.spec_from_file_location("live_observer", Path(__file__).with_name("run-v2-live-evolution.py"))
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    observer.install_responses_observer(args.output)
    budget = RequestBudget(3)
    budget.enable_live_usage(args.output / "api-usage-live.json")
    provider = V2Providers(LLMAlternatingEvolvers(
        model=model, model_args=model_args, request_budget=budget, output_directory=args.output,
    ))
    journal = EvolutionJournal(args.output, sha256_json(plan))
    stage, started, coverage = "diagnoser", perf_counter(), []
    try:
        result = journal.freeze(stage, diagnosis, lambda: provider.diagnose(diagnosis))
        coverage.append(stage)
        clusters = [c for c in result["clusters"] if c["recommended_surface"] == "skill"]
        if not clusters:
            raise RuntimeError("No live skill diagnosis; mutation coverage not established")
        context = {**mutation, "root_cause_cluster": clusters[0]}
        stage = "mutator"
        proposal = journal.freeze(stage, context, lambda: provider.mutate(context))
        coverage.append(stage)
        memory = apply_v2_mutation(ServiceSkillMemoryV2.from_mapping(context["current_service_memory"]),
                                   proposal, next_skill_id_number=1)
        if not memory.skills:
            raise RuntimeError("Live proposal makes no skill; Validator coverage not established")
        stage = "skill_validator"
        context = {"skills": memory.to_dict(), "native_policy": context["service_policy"]}
        validation = journal.freeze(stage, context, lambda: provider.validate_skill(context))
        coverage.append(stage)
        report = {"status": "complete", "completed_stages": coverage,
                  "semantic_accepted": all(validation[k] for k in ("reusable", "policy_subordinate", "no_task_entities")),
                  "native_screen_gate_verified": False, "fitness_computed": False,
                  "native_episodes": 0, "proposal": proposal, "validation": validation}
    except BaseException as error:
        report = {"status": "failed", "stage": stage, "error_type": type(error).__name__,
                  "error": safe_error(error), "completed_stages": coverage,
                  "native_screen_gate_verified": False, "fitness_computed": False,
                  "native_episodes": 0}
        _write_json_atomic(args.output / "failure.json", report)
        raise
    finally:
        report.update(elapsed_seconds=perf_counter() - started, provider_usage=budget.snapshot().to_dict())
        _write_json_atomic(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
