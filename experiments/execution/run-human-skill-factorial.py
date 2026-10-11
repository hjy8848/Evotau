"""Frozen, manually authored 2x2 mechanism diagnostic; no evolution or promotion."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

from evotau.alternating import _run_panel, _write_json_atomic, _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import (
    _api_usage_by_role,
    _begin_execution_attempt,
    _finish_execution_attempt,
    load_alternating_tasks,
)
from evotau.budget import RequestBudget
from evotau.customer_skills import CustomerSkill
from evotau.evolution_candidates import validate_mutation
from evotau.phase0 import load_config
from evotau.provider_diagnostics import safe_error
from evotau.release_recovery import frozen_run_lock
from evotau.service_skills import (
    ServiceSkillMemoryV2,
    apply_v2_mutation,
    validate_skill_budgets,
)
from evotau.strategies import PromptStrategy
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import write_manifest_once

CONDITIONS = ('C0S0', 'C1S0', 'C0S1', 'C1S1')
ASSETS = Path('research/manual-skills/airline-scope-ledger-v1')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(config_path):
    config_path = Path(config_path).resolve()
    project = config_path.parent.parent
    raw = load_config(config_path)
    manifest = AlternatingManifest.from_mapping(raw)
    plan = raw['manual_skill_trial']
    if (plan['protocol_version'] != 'human_skill_factorial_v1'
            or tuple(plan['conditions']) != CONDITIONS or plan['automatic_promotion']
            or manifest.run_validation or manifest.run_heldout
            or tuple(manifest.evolution_task_ids) != ('7', '12', '15')
            or manifest.seed != plan['seed'] or manifest.max_parallel_episodes != 2
            or plan['episodes'] != len(CONDITIONS) * len(manifest.evolution_task_ids)):
        raise ValueError('Manual diagnostic scope differs from preregistration')
    for name, expected in plan['inputs'].items():
        path = (project / name).resolve()
        if not path.is_relative_to(project) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen manual stimulus or execution source changed: ' + name)
    asset_root = project / ASSETS
    evidence = read(asset_root / 'evidence-rows.json')
    if {r['task']['task_id'] for r in evidence} != set(manifest.evolution_task_ids):
        raise ValueError('Evidence is outside selected E tasks')
    for row in evidence:
        if hashlib.sha256(Path(row['original_native_path']).read_bytes()).hexdigest() != row['original_native_sha256']:
            raise ValueError('Original E evidence changed')
    customer = CustomerSkill.from_mapping(read(asset_root / 'customer-skill.json'), evidence_rows=evidence)
    mutation = read(asset_root / 'service-mutation.json')
    validate_mutation(mutation)
    service = apply_v2_mutation(ServiceSkillMemoryV2(), mutation, next_skill_id_number=1)
    if service.to_dict() != read(asset_root / 'service-memory.json'):
        raise ValueError('Service memory differs from immutable mutation')
    validate_skill_budgets(service, raw['experiment']['skill_evolution_v2']['skill_budgets'])
    # These are fixed experimental stimuli, not accepted/installed production skills.
    panels = (
        ('C0S0', PromptStrategy(''), ServiceSkillMemoryV2()),
        ('C1S0', customer.compile(), ServiceSkillMemoryV2()),
        ('C0S1', PromptStrategy(''), service),
        ('C1S1', customer.compile(), service),
    )
    return project, raw, manifest, panels


def execute(args, project, raw, manifest, panels, tasks):
    root = project / manifest.output_path
    with frozen_run_lock(project / manifest.checkpoint_path):
        root.mkdir(parents=True, exist_ok=True)
        if (root / 'result.json').exists():
            manifest.bind_saved_provenance(read(root / 'manifest.json'))
            return read(root / 'result.json')  # No credential reads or repeated calls.
        if (root / 'run-execution-state.json').exists():
            raise RuntimeError('Incomplete prior attempt preserved; review pending requests before an explicit audited continuation')
        if any(root.iterdir()):
            raise RuntimeError('Refuse nonempty independent output directory')
        write_manifest_once(root / 'manifest.json', manifest)
        _write_json_once(root / 'config.json', raw)
        _write_json_once(root / 'operator-launch.json', {'pid': os.getpid(), 'config': str(args.config), 'protocol': 'human_skill_factorial_v1'})
        state_path, _ = _begin_execution_attempt(root, manifest.sha256)
        started = perf_counter()
        try:
            spec = importlib.util.spec_from_file_location('native_gateway_transport', Path(__file__).with_name('run-airline-gateway.py'))
            transport = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(transport)
            key = transport.credential('litellm', 'litellm-api-key')
            os.environ['OPENAI_API_KEY'] = key
            os.environ['TAU2_DATA_DIR'] = args.tau2_data_dir
            for name in ('NO_PROXY', 'no_proxy'):
                os.environ[name] = os.environ.get(name, '') + ',10.130.138.46'
            transport.install_transport(root, key)  # No Evolver route or credential.
            del key
            budget = RequestBudget(manifest.request_budget_cap)
            budget.enable_live_usage(root / 'api-usage-live.json')
            runner = TauBenchEpisodeRunner(manifest=manifest, config=raw, data_dir=args.tau2_data_dir,
                                           request_budget=budget, output_directory=root, task_objects=tasks,
                                           stop_before_next_episode_file=args.stop_before_next_episode_file)
            results = {}
            for name, customer, service in panels:
                _write_json_atomic(root / 'manual-skill-trial-stage.json', {'stage': name, 'completed_panels': list(results), 'manifest_sha256': manifest.sha256})
                records = _run_panel(runner, task_ids=manifest.evolution_task_ids, tasks=tasks,
                                     seed=manifest.seed, customer=customer, service=service,
                                     panel_name=name, max_parallel_episodes=manifest.max_parallel_episodes)
                rows = []
                for record in records:
                    directory = root / record.trajectory_ref
                    native = read(directory)
                    activation = directory.parent / 'skill-activation-trace.json'
                    rows.append({'episode': record.to_dict(), 'native_reward_info': native['reward_info'],
                                 'activation_trace': str(activation.relative_to(root)) if activation.exists() else None,
                                 'customer_validity': 'awaiting independent post-run review',
                                 'behavioral_discovery': 'not yet adjudicated'})
                results[name] = {'accuracy': sum(r.task_success for r in records) / len(records), 'rows': rows}
                _write_json_once(root / f'panel-{name}.json', results[name])
            calls = budget.api_usage_by_call_name()
            result = {'schema_version': 1, 'protocol': 'human_skill_factorial_v1', 'manifest_sha256': manifest.sha256,
                      'panels': results, 'provider_usage': budget.snapshot().to_dict(), 'api_usage_by_call_name': calls,
                      'api_usage_by_role': _api_usage_by_role(calls), 'wall_clock_seconds': perf_counter() - started,
                      'promotion': False, 'V_H_loaded': False, 'causal_skill_effect_proven': False,
                      'scope': 'Targeted E3 one-seed diagnostic; native results await fidelity/exposure adjudication.'}
            _write_json_once(root / 'result.json', result)
        except (Exception, KeyboardInterrupt) as error:
            _finish_execution_attempt(state_path, status='failed', failure={'failure_type': type(error).__name__, 'failure_message': safe_error(error), 'diagnostics_ref': getattr(error, 'diagnostics_ref', None)})
            raise
        _finish_execution_attempt(state_path)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tau2-data-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approve-unbounded-requests', action='store_true')
    parser.add_argument('--stop-before-next-episode-file')
    args = parser.parse_args()
    project, raw, manifest, panels = prepare(args.config)
    tasks = load_alternating_tasks(manifest, args.tau2_data_dir, include_validation=False, include_heldout=False)
    print(json.dumps({'E': list(tasks), 'conditions': list(CONDITIONS), 'episodes': 12, 'real_requests_started': False}), flush=True)
    if not args.execute:
        return
    if manifest.request_budget_cap is None and not args.approve_unbounded_requests:
        raise ValueError('Explicit unbounded authorization flag required')
    result = execute(args, project, raw, manifest, panels, tasks)
    print(json.dumps({k: v['accuracy'] for k, v in result['panels'].items()}), flush=True)


if __name__ == '__main__':
    main()
