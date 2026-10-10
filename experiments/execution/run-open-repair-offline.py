"""Dry configuration validation and explicitly Synthetic G2; no provider/Keychain path."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from evotau.alternating import _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.phase0 import load_config
from evotau.tau_provenance import sha256_json, write_manifest_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/airline-open-rc-v1-e10-offline.yaml"
    )
    parser.add_argument("--tau2-data-dir", required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for source, expected in config["launch_readiness"][
        "execution_source_sha256"
    ].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen offline execution source changed")
    manifest = AlternatingManifest.from_mapping(config)
    if manifest.real_provider_enabled or manifest.run_heldout:
        raise ValueError("offline launcher forbids real provider/H execution")
    if args.synthetic:
        saved = Path(manifest.output_path) / "parent-template-manifest.json"
        if saved.exists():
            manifest = manifest.bind_saved_provenance(json.loads(saved.read_text()))
        else:
            write_manifest_once(saved, manifest)
    tasks = load_alternating_tasks(
        manifest, args.tau2_data_dir, include_validation=False, include_heldout=False
    )
    report = {
        "kind": "dry_validation",
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.sha256,
        "E_task_ids": list(tasks),
        "real_provider_calls": 0,
        "H_loaded": False,
        "V_loaded": False,
    }
    if args.synthetic:
        # Explicit test fixture entry; cannot dispatch a real model or native rollout.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        from test_open_repair_evolution import DiscoveryRunner, OpenProviders, settings
        from test_skill_evolution_v2 import run

        source_files = [
            *Path("src/evotau").glob("*.py"),
            Path(__file__),
            Path("tests/test_open_repair_evolution.py"),
            Path("tests/test_skill_evolution_v2.py"),
            Path("tests/test_customer_skill_protocol.py"),
        ]
        identity = {
            "kind": "SYNTHETIC_ONLY",
            "parent_template_sha256": manifest.sha256,
            "actual_E": ["1", "2", "3"],
            "actual_V": ["4"],
            "actual_seeds": [1, 2],
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "policy": settings(True),
            "provider": "Scripted",
            "native_runner": "FakeRunner",
        }
        root = Path(manifest.output_path) / "synthetic-g2-E3-V1"
        _write_json_once(root / "synthetic-manifest.json", identity)
        providers, runner = OpenProviders(), DiscoveryRunner()
        result, _, _ = run(
            root,
            provider=providers,
            runner=runner,
            policy=settings(True),
            validation=True,
            generations=2,
            manifest_sha=sha256_json(identity),
        )
        output = {
            "kind": "SYNTHETIC_ONLY_NOT_A_RESEARCH_RESULT",
            "identity_sha256": sha256_json(identity),
            "generations": result.generations,
            "real_provider_calls": 0,
            "model_tokens": None,
            "native_online_episodes": 0,
        }
        _write_json_once(root / "synthetic-result.json", output)
        report.update(
            synthetic_output=str(root), completed_generations=len(result.generations)
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
