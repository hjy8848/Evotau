"""Dry-validate by default; execute only an authorized frozen E-only Customer search."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from evotau.alternating import LLMAlternatingEvolvers, _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import (
    _api_usage_by_role,
    _begin_execution_attempt,
    _finish_execution_attempt,
    load_alternating_tasks,
)
from evotau.budget import RequestBudget
from evotau.customer_diagnostic import run_customer_diagnostic
from evotau.episode_execution import StopBeforeEpisodeDispatch
from evotau.evolution_candidates import V2Providers
from evotau.phase0 import load_config
from evotau.provider_diagnostics import safe_error
from evotau.release_recovery import frozen_run_lock
from evotau.service_skills import ServiceSkillMemoryV2
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import role_model_args_for_runtime, write_manifest_once


def execute(args, raw, manifest, tasks, policy, project, root):
    """Same credential/transport and durable request machinery as the Airline launcher."""
    spec = importlib.util.spec_from_file_location(
        "airline_transport", Path(__file__).with_name("run-airline-gateway.py")
    )
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    gateway = transport.credential("litellm", "litellm-api-key")
    evolver = transport.credential("api.deepseek.com/v1", "evotau-evolver-api-key")
    os.environ["OPENAI_API_KEY"] = gateway
    os.environ["TAU2_DATA_DIR"] = args.tau2_data_dir
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = os.environ.get(name, "") + ",10.130.138.46"
    transport.install_transport(root, gateway, evolver)
    budget = RequestBudget(manifest.request_budget_cap)
    budget.enable_live_usage(root / "api-usage-live.json")
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=raw,
        data_dir=args.tau2_data_dir,
        request_budget=budget,
        output_directory=root,
        task_objects=tasks,
        stop_before_next_episode_file=args.stop_before_next_episode_file,
    )
    provider = LLMAlternatingEvolvers(
        model=dict(manifest.role_models)["evolver"],
        model_args=role_model_args_for_runtime(manifest.role_model_args)["evolver"],
        request_budget=budget,
        output_directory=root,
    )
    provider.stop_before_next_episode_file = args.stop_before_next_episode_file
    result = run_customer_diagnostic(
        tasks=tasks,
        task_ids=manifest.evolution_task_ids,
        runner=runner,
        providers=V2Providers(provider),
        service=ServiceSkillMemoryV2(),
        policy=policy,
        generations=manifest.generations,
        count=manifest.customer_candidates,
        seed=manifest.evolution_fitness_seed,
        concurrency=manifest.max_parallel_episodes,
        domain_policy=runner.service_policy_text,
        output=root,
        manifest_sha=manifest.sha256,
        checkpoint_path=project / manifest.checkpoint_path,
    )
    result["provider_usage"] = budget.snapshot().to_dict()
    result["api_usage_by_call_name"] = budget.api_usage_by_call_name()
    result["api_usage_by_role"] = _api_usage_by_role(result["api_usage_by_call_name"])
    _write_json_once(root / "customer-diagnostic-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tau2-data-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-unbounded-requests", action="store_true")
    parser.add_argument("--stop-before-next-episode-file")
    args = parser.parse_args()
    file = Path(args.config).resolve()
    project = file.parent.parent
    raw = load_config(file)
    manifest = AlternatingManifest.from_mapping(raw)
    policy = json.loads(manifest.skill_evolution_v2_json)["customer_evolution"]
    if (
        manifest.run_validation
        or manifest.run_heldout
        or manifest.initial_service_strategy
    ):
        raise ValueError(
            "Customer-only diagnosis freezes empty Service memory; V/H disabled"
        )
    for name, expected in raw["launch_readiness"]["execution_source_sha256"].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen diagnostic launcher differs: " + name)
    tasks = load_alternating_tasks(
        manifest, args.tau2_data_dir, include_validation=False, include_heldout=False
    )
    print(
        json.dumps(
            {
                "id": manifest.experiment_id,
                "E": list(manifest.evolution_task_ids),
                "customer_protocol": policy["protocol_version"],
                "service_frozen": True,
                "real_requests_started": False,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if manifest.request_budget_cap is None and not args.approve_unbounded_requests:
        raise ValueError("Explicit authorization required for unbounded requests")
    root = project / manifest.output_path
    with frozen_run_lock(project / manifest.checkpoint_path):
        root.mkdir(parents=True, exist_ok=True)
        path = root / "manifest.json"
        if path.exists():
            manifest = manifest.bind_saved_provenance(json.loads(path.read_text()))
        elif any(root.iterdir()):
            raise ValueError("Unbound nonempty output directory")
        else:
            write_manifest_once(path, manifest)
        if (root / "customer-diagnostic-result.json").exists():
            print(json.dumps({"status": "already_complete", "output": str(root)}))
            return  # No key reads, additional dispatch or final report overwrite.
        state_path, _ = _begin_execution_attempt(root, manifest.sha256)
        try:
            execute(args, raw, manifest, tasks, policy, project, root)
        except (Exception, KeyboardInterrupt) as error:
            latest = root / "customer-diagnostic-stage.json"
            _finish_execution_attempt(
                state_path,
                status="paused"
                if isinstance(error, (StopBeforeEpisodeDispatch, KeyboardInterrupt))
                else "failed",
                failure={
                    "failure_type": type(error).__name__,
                    "failure_message": safe_error(error),
                    "diagnostics_ref": getattr(error, "diagnostics_ref", None),
                    "latest_stage": json.loads(latest.read_text())
                    if latest.exists()
                    else None,
                },
            )
            raise
        _finish_execution_attempt(state_path)
        print(json.dumps({"status": "complete", "output": str(root)}))


if __name__ == "__main__":
    main()
