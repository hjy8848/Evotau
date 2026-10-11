"""One fixed Task12 native-customer episode; bounded activation ablation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

from evotau.alternating import _run_panel, _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import (
    _begin_execution_attempt,
    _finish_execution_attempt,
    load_alternating_tasks,
)
from evotau.budget import RequestBudget
from evotau.phase0 import load_config
from evotau.provider_diagnostics import safe_error
from evotau.release_recovery import frozen_run_lock
from evotau.service_skills import ServiceSkillMemoryV2, validate_skill_budgets
from evotau.strategies import PromptStrategy
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import write_manifest_once


def prepare(path):
    path = Path(path).resolve()
    project = path.parent.parent
    raw = load_config(path)
    manifest = AlternatingManifest.from_mapping(raw)
    plan = raw['activation_control']
    mode = raw['experiment']['skill_evolution_v2']['service_skill_runtime']
    if (tuple(manifest.evolution_task_ids) != ('12',) or manifest.seed != 1
            or manifest.run_validation or manifest.run_heldout
            or manifest.request_budget_cap != 60
            or mode not in ('activate_topk_v2', 'render_all_v1')
            or raw['experiment']['evolution']['service_skill_runtime'] != mode
            or plan['condition'] != {'activate_topk_v2': 'B', 'render_all_v1': 'C'}[mode]):
        raise ValueError('Activation control scope differs from frozen protocol')
    for filename, sha in plan['frozen_inputs'].items():
        source = (project / filename).resolve()
        if not source.is_relative_to(project) or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen stimulus/source changed')
    service = ServiceSkillMemoryV2.from_mapping(json.loads((project / plan['skill_path']).read_text()))
    validate_skill_budgets(service, raw['experiment']['skill_evolution_v2']['skill_budgets'])
    return project, raw, manifest, service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tau2-data-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    project, raw, manifest, service = prepare(args.config)
    tasks = load_alternating_tasks(manifest, args.tau2_data_dir, include_validation=False, include_heldout=False)
    if not args.execute:
        print(json.dumps({'task': '12', 'condition': raw['activation_control']['condition'], 'cap': 60, 'real_requests': 0}))
        return
    root = project / manifest.output_path
    with frozen_run_lock(project / manifest.checkpoint_path):
        if root.exists() and any(root.iterdir()):
            raise RuntimeError('Preserve prior artifacts; no blind repeat/resume')
        root.mkdir(parents=True, exist_ok=True)
        write_manifest_once(root / 'manifest.json', manifest)
        _write_json_once(root / 'config.json', raw)
        _write_json_once(root / 'operator-launch.json', {'pid': os.getpid(), 'condition': raw['activation_control']['condition']})
        state, _ = _begin_execution_attempt(root, manifest.sha256)
        started = perf_counter()
        try:
            spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('run-airline-gateway.py'))
            gateway = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gateway)
            key = gateway.credential('litellm', 'litellm-api-key')
            os.environ['OPENAI_API_KEY'] = key
            os.environ['TAU2_DATA_DIR'] = args.tau2_data_dir
            for name in ('NO_PROXY', 'no_proxy'):
                os.environ[name] = os.environ.get(name, '') + ',10.130.138.46'
            gateway.install_transport(root, key)
            del key
            budget = RequestBudget(manifest.request_budget_cap)
            budget.enable_live_usage(root / 'api-usage-live.json')
            runner = TauBenchEpisodeRunner(manifest=manifest, config=raw, data_dir=args.tau2_data_dir,
                                          request_budget=budget, output_directory=root, task_objects=tasks)
            records = _run_panel(runner, task_ids=('12',), tasks=tasks, seed=1,
                                 customer=PromptStrategy(''), service=service,
                                 panel_name=raw['activation_control']['condition'], max_parallel_episodes=2)
            _write_json_once(root / 'result.json', {'condition': raw['activation_control']['condition'],
                'records': [record.to_dict() for record in records],
                'usage': budget.snapshot().to_dict(), 'api_usage_by_call_name': budget.api_usage_by_call_name(),
                'wall_seconds': perf_counter() - started, 'V_H_loaded': False, 'promotion': False})
        except (Exception, KeyboardInterrupt) as error:
            _finish_execution_attempt(state, status='failed', failure={'failure_type': type(error).__name__, 'failure_message': safe_error(error)})
            raise
        _finish_execution_attempt(state)


if __name__ == '__main__':
    main()
