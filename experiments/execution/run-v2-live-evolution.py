"""Run the existing V2 CLI, with real probes and the existing InferAI wire observer."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evotau import inferai_responses
from evotau.alternating import LLMAlternatingEvolvers, _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks, run_from_config
from evotau.budget import RequestBudget
from evotau.evolution_candidates import V2Providers
from evotau.provider_diagnostics import response_metadata, safe_error
from evotau.tau_provenance import role_model_args_for_runtime, sha256_json


class WireOutput:
    """Route the single shared observer from probes to the native run directory."""

    def __init__(self, path):
        self.path = path

    def __truediv__(self, filename):
        return self.path / filename


def install_responses_observer(route):
    original = inferai_responses._build_opener
    lock = threading.Lock()

    def record(row):
        row["at"] = datetime.now(UTC).isoformat()
        with lock, (route / "actual-provider-http.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def factory():
        opener = original()
        original_open = opener.open

        def observed_open(request, *args, **kwargs):
            body = json.loads(request.data)
            selected = {
                k: body[k] for k in ("model", "reasoning", "stream") if k in body
            }
            record(
                {
                    "event": "request",
                    "protocol": "responses",
                    "args": selected,
                    "tools_count": len(body.get("tools") or []),
                }
            )
            started = perf_counter()
            try:
                response = original_open(request, *args, **kwargs)
            except BaseException as error:
                record(
                    {
                        "event": "failure",
                        "protocol": "responses",
                        "http_status": getattr(error, "code", None),
                        "error": safe_error(error),
                        "elapsed_seconds": perf_counter() - started,
                    }
                )
                raise
            original_read = response.read

            def observed_read(*args, **kwargs):
                raw = original_read(*args, **kwargs)
                try:
                    payload = json.loads(raw)
                except ValueError:
                    payload = {}
                record(
                    {
                        "event": "response",
                        "protocol": "responses",
                        "model": body.get("model"),
                        "http_status": response.status,
                        "metadata": response_metadata(payload),
                        "elapsed_seconds": perf_counter() - started,
                    }
                )
                return raw

            response.read = observed_read
            return response

        opener.open = observed_open
        return opener

    inferai_responses._build_opener = factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-before-next-episode-file", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["TAU2_DATA_DIR"] = str(args.tau2_data_dir.resolve())
    raw = yaml.safe_load(args.config.read_text())
    manifest = AlternatingManifest.from_mapping(raw)
    output = ROOT / manifest.output_path
    if (output / "manifest.json").exists():
        manifest = manifest.bind_saved_provenance(
            json.loads((output / "manifest.json").read_text())
        )
    tasks = load_alternating_tasks(
        manifest,
        args.tau2_data_dir,
        include_validation=manifest.run_validation,
        include_heldout=False,
    )
    summary = {
        "id": manifest.experiment_id,
        "E": list(manifest.evolution_task_ids),
        "loaded_tasks": list(tasks),
        "generations": manifest.generations,
        "customer_candidates": manifest.customer_candidates,
        "parallel_episodes": manifest.max_parallel_episodes,
        "fitness_seed": manifest.evolution_fitness_seed,
        "V": manifest.run_validation,
        "H": manifest.run_heldout,
        "role_models": dict(manifest.role_models),
        "paired_gate_seeds": json.loads(manifest.skill_evolution_v2_json)["evaluation"][
            "gate_seeds"
        ],
        "real_requests_started": False,
    }
    print(json.dumps(summary), flush=True)
    if args.dry_run:
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
        raise RuntimeError("InferAI Keychain credential missing")
    os.environ["OPENAI_API_KEY"] = key
    del key
    preflight = (
        ROOT
        / "experiments/diagnostics"
        / (manifest.experiment_id + "-provider-preflight")
    )
    preflight.mkdir(parents=True, exist_ok=True)
    _write_json_once(
        preflight / "binding.json",
        {
            "config_sha256": sha256_json(raw),
            "runtime_source_sha256": manifest.evotau_source_sha256,
        },
    )
    module_path = Path(__file__).with_name("run-v2-activation-morphology-smoke.py")
    spec = importlib.util.spec_from_file_location("wire_observer", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    route = WireOutput(preflight)
    module.install_transport(route)
    install_responses_observer(route)
    budget = RequestBudget(None)
    budget.enable_live_usage(preflight / "api-usage-live.json")
    role_args = role_model_args_for_runtime(manifest.role_model_args)
    for role in ("agent", "evolver"):
        probe = V2Providers(
            LLMAlternatingEvolvers(
                model=dict(manifest.role_models)[role],
                model_args=role_args[role],
                request_budget=budget,
                output_directory=preflight,
            )
        )
        result = probe.call(
            'Return JSON only: {"ok":true}.',
            {"purpose": "minimal no-tools provider preflight"},
            "evotau_provider_preflight",
        )
        if result != {"ok": True}:
            raise RuntimeError("Minimal JSON provider preflight failed")
        print(
            json.dumps({"stage": "preflight", "role": role, "result": result}),
            flush=True,
        )
    route.path = output
    Path("/tmp/evotau-v2-live-process.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "config": str(args.config.resolve()),
                "output": str(output),
                "id": manifest.experiment_id,
            }
        )
    )
    result_path, result = run_from_config(
        args.config,
        tau2_data_dir=args.tau2_data_dir,
        stop_before_next_episode_file=args.stop_before_next_episode_file,
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "result_path": str(result_path),
                "generations": len(result["generations"]),
                "service_accepted": [
                    g["service_phase"]["accepted"] for g in result["generations"]
                ],
                "provider_calls": result["provider_usage"]["attempts"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
