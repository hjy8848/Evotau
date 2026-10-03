from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path('/Users/spring/RSI/Evotau')
RUN_ID = 'evotau-single-task-customer-evolver-v2-unbounded-task22-20261003'
TASK_ID = '22'
VALIDATION_ID = '93'
SEED = 1
CONFIRMATION_SEED = 10001
OUTPUT_REL = f'experiments/results/{RUN_ID}'
CONFIG_PATH = ROOT / 'configs/activation-smoke-inferai-deepseek-v4-flash-unbounded-retry3.yaml'
V2_CONFIG_PATH = ROOT / 'configs/activation-smoke-customer-evolver-v2-inferai-deepseek-v4-flash-bounded-20261003.yaml'
REVIEW_PATH = ROOT / 'experiments/task-panels/activation-smoke-review.json'

sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

from evotau.archive import FailureArchive
from evotau.budget import RequestBudget
from evotau.customer_evolver_v2 import LLMCustomerStrategyEvolver
from evotau.lifecycle import TwoGenerationSmoke
from evotau.manifest import ActivationSmokeManifest, canonical_json, role_model_args_for_runtime, sha256_json
from evotau.native_runner import TauBenchEpisodeRunner
from evotau.phase0 import load_config
from evotau.phase0_run import _load_pinned_tasks
from evotau.phase3_run import load_provider_bundle
from evotau.strategies import CustomerStrategy, ServiceStrategy


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def main() -> int:
    output_dir = ROOT / OUTPUT_REL
    checkpoint_path = output_dir / 'checkpoint'
    if output_dir.exists():
        raise FileExistsError(f'run output already exists: {output_dir}')
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Keychain-injected OPENAI_API_KEY is not present')
    data_root = Path(os.environ['TAU2_DATA_DIR']).expanduser().resolve()
    raw_config = load_config(CONFIG_PATH)
    v2_config = load_config(V2_CONFIG_PATH)
    config = json.loads(json.dumps(raw_config))
    experiment = config['experiment']
    experiment['id'] = RUN_ID
    experiment['output_path'] = OUTPUT_REL
    experiment['checkpoint_path'] = f'{OUTPUT_REL}/checkpoint'
    experiment['max_concurrency'] = 1
    experiment['customer_strategy'] = v2_config['experiment']['customer_strategy']
    experiment['customer_evolver_schema'] = 'strategy_v2'
    if experiment['request_budget_cap'] is not None or experiment['unbounded_provider_budget'] is not True:
        raise ValueError('single-task diagnostic must have no provider-attempt cap')
    if any('max_tokens' in values for values in experiment['model_args'].values()):
        raise ValueError('single-task diagnostic must omit max_tokens for every role')
    if any(values.get('thinking_mode') != 'disabled' for values in experiment['model_args'].values()):
        raise ValueError('single-task diagnostic must disable thinking for every role')
    if experiment['task_selection']['heldout']:
        raise ValueError('held-out tasks must remain sealed')

    review = json.loads(REVIEW_PATH.read_text(encoding='utf-8'))
    manifest = ActivationSmokeManifest.from_mapping(config, task_review_document=review)
    reviewed_e = tuple(manifest.evolution_task_ids)
    if TASK_ID not in reviewed_e or VALIDATION_ID not in manifest.validation_task_ids:
        raise ValueError('selected E/V task is outside the frozen reviewed panel')
    tasks = _load_pinned_tasks(
        manifest,
        data_dir=data_root,
        task_selection=experiment['task_selection'],
        task_ids=reviewed_e + manifest.validation_task_ids,
    )
    if TASK_ID not in tasks:
        raise ValueError('selected task is not present in the pinned tau-bench data')

    provider_bundle = load_provider_bundle(
        'evotau.provider_plugins.deepseek_v4_flash:build_callbacks',
        config=config,
        manifest=manifest,
    )
    audit_provider = provider_bundle.callbacks['audit_provider']
    models = dict(manifest.role_models)
    model_args = role_model_args_for_runtime(manifest.role_model_args)
    evolver = LLMCustomerStrategyEvolver(
        model=models['evolver'],
        model_args=model_args['evolver'],
    )
    customer = CustomerStrategy(**experiment['customer_strategy'])
    service = ServiceStrategy(())
    budget = RequestBudget(cap=None)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_root,
        request_budget=budget,
        audit_provider=audit_provider,
        output_directory=output_dir,
    )
    (output_dir / 'run-config.yaml').write_text(
        __import__('yaml').safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    run_context = {
        'schema_version': 1,
        'run_kind': 'single_task_customer_evolution_diagnostic',
        'run_id': RUN_ID,
        'activation_manifest_sha256': manifest.sha256,
        'config_sha256': sha256_json(config),
        'task_review_sha256': manifest.task_semantic_review_sha256,
        'registered_review_panel': {'E': list(reviewed_e), 'V': list(manifest.validation_task_ids), 'H': []},
        'executed_evolution_panel': [TASK_ID],
        'executed_validation_panel': [],
        'heldout_panel': [],
        'confirmation': {'task_ids': [TASK_ID], 'discovery_seed': SEED, 'fresh_seed': CONFIRMATION_SEED},
        'generation': 0,
        'candidate_count': 2,
        'customer_evolver_schema': 'strategy_v2',
        'customer_proposal_mode': 'strategy_v2',
        'service_strategy_frozen': True,
        'provider_budget_cap': None,
        'max_tokens_sent': False,
        'thinking_mode': 'disabled',
        'model': models['agent'],
        'api_base': dict(model_args['agent']).get('api_base'),
        'provider_response_policy': 'retry-empty-successful-completion-until-nonempty',
        'provider_provenance': provider_bundle.provenance,
        'initial_customer_strategy': customer.to_dict(),
        'service_strategy': service.to_dict(),
        'tau2_data_root': str(data_root),
    }
    write_once(output_dir / 'run-context.json', run_context)
    archive = FailureArchive(output_dir / 'archive.sqlite')
    controller_manifest = {
        'phase': '3-multitask-service-repair-activation',
        'generations': 1,
        'max_episodes': 100,
        'max_concurrency': 1,
        'customer_evolver_schema': 'strategy_v2',
    }
    controller = TwoGenerationSmoke(
        manifest=controller_manifest,
        checkpoint_path=str(checkpoint_path),
        runner=runner,
        task_ids=(TASK_ID, VALIDATION_ID),
        evolution_task_ids=(TASK_ID,),
        seed=SEED,
        episode_seed_base=SEED,
        request_budget=budget,
        failure_archive=archive,
        manifest_context=run_context,
        generations=1,
        max_episodes=100,
    )
    controller.episode_attempts = 0
    print(
        f'PREFLIGHT PASSED: task {TASK_ID} is a researcher-reviewed E task; '
        'pinned data, provider callbacks, and unbounded settings verified.',
        flush=True,
    )
    print(
        'RUNNING: one Customer evolution generation, K=2, discovery seed=1, '
        'fresh confirmation seed=10001 when the Core selection protocol requests it; '
        'provider-attempt budget is uncapped and max_tokens is omitted.',
        flush=True,
    )
    try:
        commits = controller.run(
            customer,
            service,
            customer_proposal_provider=evolver,
            customer_proposal_mode='strategy_v2',
            allow_frozen_service=True,
            candidates_per_generation=2,
        )
        commit_payload = {
            'generation': commits[0].generation,
            'customer_id': commits[0].customer_id,
            'service_id': commits[0].service_id,
            'customer_evolved': commits[0].customer_evolved,
            'service_evolved': commits[0].service_evolved,
            'completed': commits[0].completed,
            'note': commits[0].note,
            'decision_record': commits[0].decision_record,
        }
        episode_records = [
            json.loads(path.read_text(encoding='utf-8'))
            for path in sorted(output_dir.glob('episodes/*/episode-record.json'))
        ]
        result = {
            'schema_version': 1,
            'status': 'complete',
            'run_kind': 'single_task_customer_evolution_diagnostic',
            'run_id': RUN_ID,
            'task_id': TASK_ID,
            'task_name': str(getattr(tasks[TASK_ID], 'description', ''))[:240],
            'generation_commit': commit_payload,
            'episode_count': len(episode_records),
            'episodes': [
                {
                    'episode_id': item['episode_id'],
                    'task_id': item['task_id'],
                    'seed': item['seed'],
                    'panel_name': item.get('panel_name'),
                    'customer_strategy_id': item['customer_strategy_id'],
                    'status': item['status'],
                    'task_success': item['task_success'],
                    'native_reward': item['native_reward'],
                    'customer_valid': item['customer_valid'],
                    'strategy_applicable': item['strategy_applicable'],
                    'customer_strategy_adherent': item['customer_strategy_adherent'],
                    'policy_violation': item['policy_violation'],
                    'policy_rule_id': item.get('policy_rule_id'),
                    'mistake_type': item.get('mistake_type'),
                    'workflow_stage': item.get('workflow_stage'),
                    'audit_ref': item.get('audit_ref'),
                }
                for item in episode_records
            ],
            'provider_usage': budget.snapshot().to_dict(),
            'caveat': (
                'One reviewed task is a diagnostic only. A selected Customer change is '
                'evidence of task-level evolution under this task/seed protocol, not a '
                'general effectiveness claim across tasks.'
            ),
        }
        write_once(output_dir / 'single-task-evolution-result.json', result)
        print(json.dumps({
            'status': result['status'],
            'task_id': TASK_ID,
            'customer_evolved': commits[0].customer_evolved,
            'selection_reason': commits[0].note,
            'episodes': len(episode_records),
            'provider_attempts': budget.snapshot().attempts,
            'prompt_tokens': budget.snapshot().prompt_tokens,
            'completion_tokens': budget.snapshot().completion_tokens,
            'artifact': str(output_dir / 'single-task-evolution-result.json'),
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {
            'schema_version': 1,
            'status': 'failed',
            'run_kind': 'single_task_customer_evolution_diagnostic',
            'run_id': RUN_ID,
            'task_id': TASK_ID,
            'error_type': type(exc).__name__,
            'provider_usage': budget.snapshot().to_dict(),
            'completed_episode_records': [
                json.loads(path.read_text(encoding='utf-8'))
                for path in sorted(output_dir.glob('episodes/*/episode-record.json'))
            ],
        }
        write_once(output_dir / 'single-task-evolution-result.json', result)
        print(json.dumps({
            'status': 'failed', 'error_type': type(exc).__name__,
            'provider_attempts': budget.snapshot().attempts,
            'artifact': str(output_dir / 'single-task-evolution-result.json'),
        }), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
