"""Authorized real Pair Acquisition, with separately accounted no-tools preflight.

This runs the existing native alternating implementation, not the A/B/C search
comparison. No scores are imported and H must remain closed.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

from evotau.alternating import _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks, run_from_config
from evotau.budget import RequestBudget
from evotau.phase0 import load_config
from evotau.provider_diagnostics import safe_error
from evotau.release_recovery import frozen_run_lock
from evotau.skill_evolution_config import validate_v2_promotion_readiness
from evotau.tau_provenance import role_model_args_for_runtime, write_manifest_once


def check_config(raw):
    manifest = AlternatingManifest.from_mapping(raw)
    policy = json.loads(manifest.skill_evolution_v2_json)
    validate_v2_promotion_readiness(policy, run_validation=manifest.run_validation)
    if (manifest.domain != "airline" or manifest.run_heldout
            or not manifest.run_validation or not manifest.real_provider_enabled
            or manifest.generations != 1 or manifest.customer_candidates != 1
            or manifest.max_parallel_episodes != 2):
        raise ValueError("Pair Acquisition requires frozen Airline E/V G1/C1/P2, no H")
    if any(policy.get(k) for k in
           ("open_repair_search", "repair_conditioned_bandit", "discovery_handoff", "targeted_repair")):
        raise ValueError("Acquisition cannot masquerade as repair-conditioned comparison")
    if raw["launch_readiness"].get("authorization") != "user_explicit_unbounded_20261010":
        raise ValueError("Missing current explicit unbounded authorization")
    return manifest


def recorded_probe(directory, label, model, args, budget, generate, messages):
    """Never replay a request with an unresolved durable intent."""
    start, done = directory / (label + "-intent.json"), directory / (label + "-response.json")
    if done.exists():
        saved = json.loads(done.read_text())
        if saved["model"] != model or saved["args"] != args or saved["status"] != "valid":
            raise ValueError("Preflight identity/validity mismatch")
        return saved
    if start.exists():
        raise RuntimeError("Unresolved preflight request: inspect evidence before authorizing replay")
    _write_json_once(start, {"model": model, "args": args, "num_retries": 0, "tools": None})
    with budget.record_provider_calls(directory / "provider-calls.jsonl"):
        response = generate(model=model, messages=messages, call_name="open_rc_preflight_" + label,
                            num_retries=0, **args)
    # Save the complete visible message BEFORE JSON/schema validation.
    _write_json_once(directory / (label + "-visible.json"), response.model_dump(mode="json"))
    parsed = json.loads(response.content)
    if type(parsed) is not dict or parsed != {"ok": True}:
        raise ValueError("Preflight must return exactly a valid {ok: true} object")
    saved = {"model": model, "args": args, "status": "valid", "parsed": parsed}
    _write_json_once(done, saved)
    return saved


def verify_official_evolver_wire(body):
    if (body.get("model") != "deepseek-flash"
            or body.get("thinking") != {"type": "enabled"}
            or body.get("reasoning_effort") != "high" or body.get("tools")):
        raise ValueError("Official Evolver wire args differ from frozen thinking/high configuration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tau2-data-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-unbounded-requests", action="store_true")
    options = parser.parse_args()
    raw = load_config(options.config)
    manifest = check_config(raw)
    import hashlib

    for path, expected in raw["launch_readiness"]["execution_source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen execution source changed: " + path)
    load_alternating_tasks(manifest, options.tau2_data_dir, include_validation=True, include_heldout=False)
    print(json.dumps({"stage": "dry_validated", "experiment_id": manifest.experiment_id,
                      "H_loaded": False, "real_requests_started": False}), flush=True)
    if not options.execute:
        return
    if not options.approve_unbounded_requests or manifest.request_budget_cap is not None:
        raise ValueError("Execute requires explicit authorization matching this unbounded config")
    root = Path(manifest.output_path)
    with frozen_run_lock(root / "operator-acquisition-checkpoint.json"):
        if (root / "manifest.json").exists():
            manifest = manifest.bind_saved_provenance(json.loads((root / "manifest.json").read_text()))
        elif root.exists() and any(p.name not in {"operator-launch.json", "launch.log", "operator-acquisition-checkpoint.lock"}
                                   for p in root.iterdir()):
            raise ValueError("Cannot bind unrecognized existing artifacts")
        if not (root / "manifest.json").exists():
            write_manifest_once(root / "manifest.json", manifest)
        module_spec = importlib.util.spec_from_file_location(
            "airline_transport", Path(__file__).with_name("run-airline-gateway.py"))
        transport = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(transport)
        try:
            os.environ["TAU2_DATA_DIR"] = options.tau2_data_dir
            for name in ("NO_PROXY", "no_proxy"):
                os.environ[name] = os.environ.get(name, "") + ",10.130.138.46"
            gateway_key = transport.credential("litellm", "litellm-api-key")
            evolver_key = transport.credential("api.deepseek.com/v1", "evotau-evolver-api-key")
            os.environ["OPENAI_API_KEY"] = gateway_key
            transport.install_transport(root, gateway_key, evolver_key)
            import httpx

            guarded_send = httpx.Client.send

            def verified_send(client, request, *args, **kwargs):
                if request.method == "POST" and request.url.host == "api.deepseek.com":
                    verify_official_evolver_wire(json.loads(request.content))
                return guarded_send(client, request, *args, **kwargs)

            httpx.Client.send = verified_send
            from tau2.data_model.message import SystemMessage, UserMessage
            from tau2.utils import llm_utils

            directory = root / "preflight"
            directory.mkdir(exist_ok=True)
            budget = RequestBudget(None)
            budget.enable_live_usage(directory / "api-usage-live.json")
            messages = [SystemMessage(role="system", content="Return only the exact JSON object requested."),
                        UserMessage(role="user", content='Return {"ok":true}.')]
            with budget.instrument_tau_llm_utils(llm_utils):
                for role in ("agent", "evolver"):
                    recorded_probe(directory, role, raw["experiment"]["models"][role],
                                   role_model_args_for_runtime(manifest.role_model_args)[role],
                                   budget, llm_utils.generate, messages)
            print(json.dumps({"stage": "minimal_preflight_complete", "calls": budget.snapshot().attempts,
                              "long_context_verified": False}), flush=True)
            output_directory, result = run_from_config(options.config, tau2_data_dir=options.tau2_data_dir)
            result_path = output_directory / "alternating-result.json"
            before, after = result["initial_service"], result["final_service"]
            changed = before != after
            _write_json_once(root / "pair-acquisition-status.json", {
                "stage": "acquisition_complete", "service_changed": changed,
                "comparison_started": False, "H_loaded": False,
                "pair_status": "promotion_requires_provenance_audit" if changed else "no_promoted_pair",
                "preflight_usage": budget.snapshot().to_dict(),
                "native_evolution_usage": result["provider_usage"],
                "interpretation": "No Pair means H1/H2 comparison must not proceed."})
            print(json.dumps({"stage": "acquisition_complete", "service_changed": changed,
                              "result_path": str(result_path)}), flush=True)
        except Exception as error:
            _write_json_once(root / "operator-acquisition-failure.json", {
                "error_type": type(error).__name__, "message": safe_error(error),
                "automatic_retry": False, "comparison_started": False})
            raise


if __name__ == "__main__":
    main()
