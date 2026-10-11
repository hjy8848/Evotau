"""Frozen Retail G5 gateway launch; existing alternating runtime, no algorithm changes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks, run_from_config
from evotau.phase0 import load_config
from evotau.skill_evolution_config import validate_v2_promotion_readiness


def prepare(config_path):
    path = Path(config_path).resolve()
    project = path.parent.parent
    raw = load_config(path)
    manifest = AlternatingManifest.from_mapping(raw)
    policy = json.loads(manifest.skill_evolution_v2_json)
    if (manifest.domain != 'retail' or manifest.generations != 5
            or manifest.max_parallel_episodes != 4
            or tuple(len(x) for x in (manifest.evolution_task_ids, manifest.validation_task_ids, manifest.heldout_task_ids)) != (20, 20, 20)
            or not manifest.run_validation or not manifest.run_heldout
            or any(model != 'openai/dashscope/qwen3.7-plus' for _, model in manifest.role_models)):
        raise ValueError('Frozen Retail G5 scope mismatch')
    validate_v2_promotion_readiness(policy, run_validation=True)
    for name, expected in raw['launch_readiness']['execution_source_sha256'].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen launcher changed')
    return project, raw, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tau2-data-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approve-unbounded-requests', action='store_true')
    parser.add_argument('--stop-before-next-episode-file')
    args = parser.parse_args()
    project, _raw, manifest = prepare(args.config)
    load_alternating_tasks(manifest, args.tau2_data_dir, include_validation=True, include_heldout=False)
    print(json.dumps({'experiment_id': manifest.experiment_id, 'generations': 5,
                      'max_parallel_episodes': 4, 'E_V_H': [20, 20, 20],
                      'H_content_loaded': False, 'request_cap': manifest.request_budget_cap,
                      'real_requests_started': False}), flush=True)
    if not args.execute:
        return
    if manifest.request_budget_cap is None and not args.approve_unbounded_requests:
        raise ValueError('Unbounded execution authorization flag required')
    spec = importlib.util.spec_from_file_location('gateway_transport', Path(__file__).with_name('run-airline-gateway.py'))
    gateway = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gateway)
    key = gateway.credential('litellm', 'litellm-api-key')
    os.environ['OPENAI_API_KEY'] = key
    os.environ['TAU2_DATA_DIR'] = args.tau2_data_dir
    for name in ('NO_PROXY', 'no_proxy'):
        os.environ[name] = os.environ.get(name, '') + ',10.130.138.46'
    # Production run_from_config owns the single-writer lock, immutable manifest,
    # checkpoints, stage caches, request accounting and existing format recovery.
    root = project / manifest.output_path
    root.mkdir(parents=True, exist_ok=True)
    gateway.install_transport(root, key)
    del key
    result_path, _ = run_from_config(args.config, tau2_data_dir=args.tau2_data_dir,
                                    stop_before_next_episode_file=args.stop_before_next_episode_file)
    print(json.dumps({'stage': 'complete', 'result_path': str(result_path)}), flush=True)


if __name__ == '__main__':
    main()
