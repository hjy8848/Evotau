from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/Users/spring/RSI/Evotau')
SOURCE_ID = 'evotau-single-task-customer-evolver-v2-unbounded-task22-20261003'
RUN_ID = 'evotau-single-task-customer-evolver-v2-unbounded-task22-recovery1-20261003'
TASK_ID = '22'
VALIDATION_ID = '93'
SEED = 1
CONFIRMATION_SEED = 10001
OUTPUT_REL = f'experiments/results/{RUN_ID}'
SOURCE_ROOT = ROOT / 'experiments/results' / SOURCE_ID
CONFIG_PATH = ROOT / 'configs/activation-smoke-inferai-deepseek-v4-flash-unbounded-retry3.yaml'
V2_CONFIG_PATH = ROOT / 'configs/activation-smoke-customer-evolver-v2-inferai-deepseek-v4-flash-bounded-20261003.yaml'
REVIEW_PATH = ROOT / 'experiments/task-panels/activation-smoke-review.json'

sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)

from evotau.archive import FailureArchive
from evotau.budget import BudgetSnapshot, EpisodeUsageTracker, ModelUsageSnapshot, RequestBudget
from evotau.customer_evolver_v2 import LLMCustomerStrategyEvolver
from evotau.lifecycle import TwoGenerationSmoke
from evotau.manifest import ActivationSmokeManifest, role_model_args_for_runtime, sha256_json
from evotau.native_runner import (
    TauBenchEpisodeRunner,
    _audit_dict,
    _count_tool_calls,
    _write_db_state_trace,
)
from evotau.phase0 import load_config
from evotau.phase0_run import _load_pinned_tasks
from evotau.phase3_run import load_provider_bundle
from evotau.records import EpisodeRecord, EpisodeStatus, customer_strategy_id, service_strategy_id
from evotau.strategies import CustomerStrategy, ServiceStrategy
from evotau.communication import observe_communication_protocol


def write_once(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def sum_snapshots(left: BudgetSnapshot, right: BudgetSnapshot) -> BudgetSnapshot:
    if left.cap != right.cap:
        raise ValueError('cannot aggregate provider accounting with different caps')
    names = {
        'attempts', 'successes', 'failures', 'denied', 'prompt_tokens',
        'completion_tokens', 'usage_responses', 'usage_unavailable', 'cache_hits',
    }
    models = {}
    for row in (*left.model_usage, *right.model_usage):
        target = models.setdefault(row.model_id, {name: 0 for name in names} | {'in_flight': 0})
        for name in names:
            target[name] += getattr(row, name)
    return BudgetSnapshot(
        cap=left.cap,
        **{name: getattr(left, name) + getattr(right, name) for name in names},
        in_flight=0,
        reserved=0,
        model_usage=tuple(
            ModelUsageSnapshot(model_id=model, **values)
            for model, values in sorted(models.items())
        ),
    )


def main() -> int:
    output_dir = ROOT / OUTPUT_REL
    checkpoint_path = output_dir / 'checkpoint'
    if output_dir.exists():
        raise FileExistsError(f'recovery output already exists: {output_dir}')
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
        raise ValueError('recovery must retain uncapped provider-attempt budget')
    if any('max_tokens' in values for values in experiment['model_args'].values()):
        raise ValueError('recovery must continue to omit max_tokens')

    review = json.loads(REVIEW_PATH.read_text(encoding='utf-8'))
    manifest = ActivationSmokeManifest.from_mapping(config, task_review_document=review)
    if TASK_ID not in manifest.evolution_task_ids or VALIDATION_ID not in manifest.validation_task_ids:
        raise ValueError('recovery task is outside the frozen reviewed panel')
    tasks = _load_pinned_tasks(
        manifest,
        data_dir=data_root,
        task_selection=experiment['task_selection'],
        task_ids=manifest.evolution_task_ids + manifest.validation_task_ids,
    )
    task = tasks[TASK_ID]

    source_dirs = sorted((SOURCE_ROOT / 'episodes').glob('*/incomplete-run.json'))
    if len(source_dirs) != 1:
        raise ValueError('expected exactly one preserved incomplete source episode')
    incomplete_path = source_dirs[0]
    source_episode_dir = incomplete_path.parent
    incomplete = json.loads(incomplete_path.read_text(encoding='utf-8'))
    trajectory_path = source_episode_dir / 'native-simulation.json'
    trajectory = json.loads(trajectory_path.read_text(encoding='utf-8'))
    if not incomplete.get('native_simulation_saved') or trajectory.get('task_id') != TASK_ID or trajectory.get('seed') != SEED:
        raise ValueError('saved source trajectory does not match the requested reviewed task')

    provider_bundle = load_provider_bundle(
        'evotau.provider_plugins.deepseek_v4_flash:build_callbacks',
        config=config,
        manifest=manifest,
    )
    audit_provider = provider_bundle.callbacks['audit_provider']
    models = dict(manifest.role_models)
    model_args = role_model_args_for_runtime(manifest.role_model_args)
    evolver = LLMCustomerStrategyEvolver(model=models['evolver'], model_args=model_args['evolver'])
    customer = CustomerStrategy(**experiment['customer_strategy'])
    service = ServiceStrategy(())
    budget = RequestBudget(cap=None)
    source_usage = BudgetSnapshot(**incomplete['budget_after'])
    budget.restore_usage(source_usage)
    runner = TauBenchEpisodeRunner(
        manifest=manifest, config=config, data_dir=data_root,
        request_budget=budget, audit_provider=audit_provider,
        output_directory=output_dir,
    )
    (output_dir / 'run-config.yaml').write_text(
        __import__('yaml').safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8',
    )
    run_context = {
        'schema_version': 1,
        'run_kind': 'single_task_customer_evolution_diagnostic_recovered_audit',
        'run_id': RUN_ID,
        'activation_manifest_sha256': manifest.sha256,
        'task_review_sha256': manifest.task_semantic_review_sha256,
        'registered_review_panel': {'E': list(manifest.evolution_task_ids), 'V': list(manifest.validation_task_ids), 'H': []},
        'executed_evolution_panel': [TASK_ID],
        'executed_validation_panel': [],
        'heldout_panel': [],
        'discovery_seed': SEED,
        'fresh_confirmation_seed': CONFIRMATION_SEED,
        'generation': 0,
        'candidate_count': 2,
        'customer_evolver_schema': 'strategy_v2',
        'service_strategy_frozen': True,
        'provider_budget_cap': None,
        'max_tokens_sent': False,
        'thinking_mode': 'disabled',
        'model': models['agent'],
        'api_base': dict(model_args['agent']).get('api_base'),
        'provider_response_policy': 'retry-empty-successful-completion-until-nonempty',
        'provider_provenance': provider_bundle.provenance,
        'recovered_source': {
            'run_id': SOURCE_ID,
            'incomplete_artifact': str(incomplete_path.relative_to(ROOT)),
            'native_simulation_sha256': __import__('hashlib').sha256(trajectory_path.read_bytes()).hexdigest(),
            'original_attempts': source_usage.attempts,
        },
    }
    write_once(output_dir / 'run-context.json', run_context)
    source_record = json.loads(json.dumps(incomplete))
    write_once(output_dir / 'recovery-source-incomplete.json', source_record)

    episode_dir = output_dir / 'episodes/recovered-baseline'
    episode_dir.mkdir(parents=True, exist_ok=False)
    recovered_trajectory_path = episode_dir / 'native-simulation.json'
    shutil.copyfile(trajectory_path, recovered_trajectory_path)
    simulation = SimpleNamespace(
        messages=trajectory.get('messages') or [],
        model_dump=lambda mode='json': trajectory,
    )
    before_audit = budget.snapshot()
    with budget.track_episode_usage() as audit_usage:
        from tau2.utils import llm_utils
        with budget.instrument_tau_llm_utils(llm_utils, retry_empty_responses=True):
            audit = audit_provider(simulation, task, customer, service, 'discovery')
    after_audit = budget.snapshot()
    old_episode_usage = BudgetSnapshot(**incomplete['episode_budget_delta'])
    recovered_episode_usage = sum_snapshots(old_episode_usage, audit_usage.snapshot(cap=None))
    reward_info = trajectory.get('reward_info') or {}
    reward = reward_info.get('reward')
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        native_reward = None
        task_success = None
        status = EpisodeStatus.UNCERTAIN
    else:
        native_reward = float(reward)
        task_success = native_reward >= 1.0
        if not audit.customer_valid:
            status = EpisodeStatus.INVALID_CUSTOMER
        elif audit.strategy_applicable and not audit.customer_strategy_adherent:
            status = EpisodeStatus.INVALID_STRATEGY
        else:
            status = EpisodeStatus.COMPLETE
    protocol = observe_communication_protocol(
        trajectory.get('messages') or (),
        enforcement_enabled=manifest.enforce_communication_protocol,
    )
    episode_id = str(trajectory.get('id') or incomplete['attempt_id'])
    record = EpisodeRecord(
        episode_id=episode_id,
        task_id=TASK_ID,
        seed=SEED,
        customer_strategy_id=customer_strategy_id(customer),
        service_strategy_id=service_strategy_id(service),
        status=status,
        task_success=task_success,
        native_reward=native_reward,
        termination_reason=trajectory.get('termination_reason'),
        customer_valid=audit.customer_valid,
        strategy_applicable=audit.strategy_applicable,
        customer_strategy_adherent=audit.customer_strategy_adherent,
        policy_violation=audit.policy_violation,
        invalid_repeated_write_calls=audit.invalid_repeated_write_calls,
        policy_rule_id=audit.policy_rule_id,
        mistake_type=audit.mistake_type,
        workflow_stage=audit.workflow_stage,
        evidence=audit.evidence,
        trajectory_ref=recovered_trajectory_path.relative_to(output_dir).as_posix(),
        audit_ref=audit.verifier_ref,
        tool_calls=_count_tool_calls(trajectory.get('messages') or []),
        enforce_communication_protocol=manifest.enforce_communication_protocol,
        mixed_text_tool_call_messages=protocol['mixed_text_tool_call_message_count'],
        raw_review={
            'native_review': trajectory.get('review'),
            'auth_classification': trajectory.get('auth_classification'),
        },
    )
    after = budget.snapshot()
    prior_attempt_before = BudgetSnapshot(**incomplete['budget_before'])
    total_delta = sum_snapshots(prior_attempt_before, after)
    trace_telemetry = _write_db_state_trace(
        manifest=manifest, task=task, trajectory=trajectory,
        trajectory_path=recovered_trajectory_path,
        episode_directory=episode_dir, data_root=data_root,
    )
    write_once(episode_dir / 'audit-recovery.json', {
        'source_incomplete_artifact': str(incomplete_path.relative_to(ROOT)),
        'audit': _audit_dict(audit),
        'audit_retry_usage': audit_usage.snapshot(cap=None).to_dict(),
        'source_episode_usage': old_episode_usage.to_dict(),
    })
    write_once(episode_dir / 'run-telemetry.json', {
        'attempt_id': 'recovered-baseline',
        'simulation_id': episode_id,
        'episode_key': incomplete['episode_key'],
        'episode_key_sha256': incomplete['episode_key_sha256'],
        'panel_name': 'discovery',
        'customer_strategy_id': record.customer_strategy_id,
        'service_strategy_id': record.service_strategy_id,
        'rendered_prompt_sha256': incomplete['rendered_prompt_sha256'],
        'budget_before': before_audit.to_dict(),
        'budget_after': after_audit.to_dict(),
        'budget_delta': audit_usage.snapshot(cap=None).to_dict(),
        'episode_budget_delta': recovered_episode_usage.to_dict(),
        'communication_protocol_observation': protocol,
        'independent_audit': _audit_dict(audit),
        'recovered_prior_attempt_usage': old_episode_usage.to_dict(),
        **trace_telemetry,
    })
    write_once(episode_dir / 'episode-record.json', record.to_dict())

    class RecoveryAwareRunner:
        service_policy_text = runner.service_policy_text
        def __call__(self, **kwargs):
            if kwargs == {
                'task_id': TASK_ID, 'seed': SEED, 'customer': customer,
                'service': service, 'panel_name': 'discovery',
            }:
                return record
            return runner(**kwargs)
        def has_completed_episode(self, **kwargs):
            if kwargs == {
                'task_id': TASK_ID, 'seed': SEED, 'customer': customer,
                'service': service, 'panel_name': 'discovery',
            }:
                return True
            return runner.has_completed_episode(**kwargs)
        def load_trajectory(self, episode):
            if episode.episode_id == record.episode_id:
                return trajectory
            return runner.load_trajectory(episode)

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
        runner=RecoveryAwareRunner(),
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
        f'RECOVERY PREFLIGHT PASSED: reused the already completed native trajectory for task {TASK_ID}; '
        'only the failed independent audit is retried before Core evolution starts.',
        flush=True,
    )
    try:
        commits = controller.run(
            customer, service,
            customer_proposal_provider=evolver,
            customer_proposal_mode='strategy_v2',
            allow_frozen_service=True,
            candidates_per_generation=2,
        )
        commit = commits[0]
        records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(output_dir.glob('episodes/*/episode-record.json'))]
        result = {
            'schema_version': 1,
            'status': 'complete',
            'run_kind': 'single_task_customer_evolution_diagnostic_recovered_audit',
            'run_id': RUN_ID,
            'task_id': TASK_ID,
            'generation_commit': {
                'generation': commit.generation,
                'customer_id': commit.customer_id,
                'customer_evolved': commit.customer_evolved,
                'service_evolved': commit.service_evolved,
                'completed': commit.completed,
                'selection_reason': commit.note,
                'decision_record': commit.decision_record,
            },
            'episode_count': len(records),
            'episodes': [
                {key: item.get(key) for key in (
                    'episode_id','task_id','seed','customer_strategy_id','status','task_success',
                    'native_reward','customer_valid','strategy_applicable','customer_strategy_adherent',
                    'policy_violation','policy_rule_id','mistake_type','workflow_stage','audit_ref',
                )}
                for item in records
            ],
            'provider_usage': budget.snapshot().to_dict(),
            'caveat': 'One task is diagnostic; task-level strategy selection does not establish cross-task effectiveness.',
        }
        write_once(output_dir / 'single-task-evolution-result.json', result)
        print(json.dumps({
            'status': 'complete', 'task_id': TASK_ID,
            'customer_evolved': commit.customer_evolved,
            'selection_reason': commit.note,
            'episodes': len(records),
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
            'run_kind': 'single_task_customer_evolution_diagnostic_recovered_audit',
            'run_id': RUN_ID,
            'task_id': TASK_ID,
            'error_type': type(exc).__name__,
            'provider_usage': budget.snapshot().to_dict(),
            'completed_episode_records': [json.loads(p.read_text(encoding='utf-8')) for p in sorted(output_dir.glob('episodes/*/episode-record.json'))],
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
