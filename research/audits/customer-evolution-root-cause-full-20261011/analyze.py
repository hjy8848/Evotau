"""Read-only, E-only forensic inventory. Never constructs an agent or calls a model.

Run using the pinned EvoTau Python environment. Inputs remain untouched; outputs are
written only next to this script. Task decoding explicitly excludes all V/H objects.
Native environment replay uses recorded messages and deterministic native tools only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from collections import Counter
from pathlib import Path

from loguru import logger

AUDIT = Path(__file__).resolve().parent
WORKTREE = Path('/Users/spring/RSI/Evotau-analyst')
DATA = Path('/Users/spring/.cache/evotau/tau2-data-b7ea9074')
RUNS = {
    'ab-v1-s1': 'evotau-airline-customer-prompt-ab-v1-e10-k2-seed1-20261011',
    'ab-v2-s1': 'evotau-airline-customer-prompt-ab-v2-e10-k2-seed1-20261011',
    'ab-v1-s2-partial': 'evotau-airline-customer-prompt-ab-v1-e10-k2-seed2-20261011',
    'acquisition': 'evotau-open-rc-pair-acquisition-airline-e10-v20-g1-c1-p2-20261010-readfix2',
}


def no_network(*_args, **_kwargs):
    raise RuntimeError('This audit prohibits network connections and model requests')


socket.socket.connect = no_network
socket.create_connection = no_network
os.environ['TAU2_DATA_DIR'] = str(DATA)
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
logger.remove()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, obj):
    (AUDIT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def selected_tasks(ids):
    """Read bytes for provenance, but decode only the selected top-level E tasks."""
    text = (DATA / 'tau2/domains/airline/tasks.json').read_text()
    decoder = json.JSONDecoder()
    tasks = {}
    for match in re.finditer(r'(?m)^    \{\n        "id": "([^"]+)"', text):
        if match[1] in ids:
            tasks[match[1]] = decoder.raw_decode(text, match.start() + 4)[0]
    assert set(tasks) == ids
    return tasks


def canonical(message):
    # Generated IDs/timestamps differ without any semantic dialogue difference.
    return {
        'role': message.get('role'), 'content': message.get('content'),
        'tool_calls': [
            {'name': t['name'], 'arguments': t['arguments']}
            for t in message.get('tool_calls') or []
        ],
    }


def first_divergence(left, right):
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if canonical(a) != canonical(b):
            return {'index': i, 'before': canonical(a), 'after': canonical(b)}
    if len(left) != len(right):
        return {'index': min(len(left), len(right)), 'length_difference': True}
    return None


def replay(native, task):
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    parsed = SimulationRun.model_validate(native)
    native_task = Task.model_validate(task)
    result = EnvironmentEvaluator.calculate_reward(
        get_environment, native_task, parsed.messages, strict_replay=True,
    ).model_dump(mode='json')
    assert result['db_check'] == native['reward_info']['db_check']
    communication = CommunicateEvaluator.calculate_reward(native_task, parsed.messages)
    assert communication.reward == native['reward_info']['reward_breakdown']['COMMUNICATE']
    state = get_environment()
    initial = native_task.initial_state
    state.set_state(
        initialization_data=None if initial is None else initial.initialization_data,
        initialization_actions=None if initial is None else initial.initialization_actions,
        message_history=parsed.messages, strict=True,
    )
    return result, state.tools.db.model_dump(mode='json')


def main():
    from tau2.domains.airline.data_model import FlightDB

    manifests = {
        label: read(WORKTREE / 'experiments/runs' / name / 'manifest.json')
        for label, name in RUNS.items()
    }
    ids = set(manifests['ab-v2-s1']['task_panels']['E'])
    for manifest in manifests.values():
        assert set(manifest['task_panels']['E']) == ids
        assert not ids & set(manifest['task_panels']['V'])
        assert not ids & set(manifest['task_panels']['H'])
    tasks = selected_tasks(ids)
    db = read(DATA / 'tau2/domains/airline/db.json')
    normalized_db = FlightDB.model_validate(db).model_dump(mode='json')
    for manifest in manifests.values():
        for name in ('tasks.json', 'policy.md', 'db.json', 'split_tasks.json'):
            raw = (DATA / 'tau2/domains/airline' / name).read_bytes()
            git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
            assert git_blob == manifest['source_blob_sha1'][f'data/tau2/domains/airline/{name}']
    episodes = []
    native_by_key = {}
    transcripts = []
    for label, name in RUNS.items():
        root = WORKTREE / 'experiments/runs' / name
        files = sorted(root.glob('episodes/*/episode-record.json'))
        for record_file in files:
            record = read(record_file)
            assert record['task_id'] in ids, 'Refuse decoding a non-E trajectory'
            assert record['status'] == 'complete'
            trajectory = root / record['trajectory_ref']
            native = read(trajectory)
            telemetry = read(record_file.parent / 'run-telemetry.json')
            messages = native['messages']
            breakdown = native['reward_info']['reward_breakdown']
            result, state = replay(native, tasks[record['task_id']])
            communicate = tasks[record['task_id']]['evaluation_criteria']['communicate_info'] or []
            texts = [m.get('content') or '' for m in messages if m['role'] == 'assistant']
            comm_result = float(all(any(
                item.lower().replace(',', '') in text.lower().replace(',', '')
                for text in texts
            ) for item in communicate))
            assert comm_result == breakdown['COMMUNICATE']
            writes = []
            for i, m in enumerate(messages):
                for t in m.get('tool_calls') or []:
                    if t['name'] not in {
                        'book_reservation', 'cancel_reservation',
                        'update_reservation_flights', 'update_reservation_passengers',
                        'update_reservation_baggages', 'send_certificate',
                    }:
                        continue
                    matches = [
                        (j, tm) for j, tm in enumerate(messages[i + 1:], i + 1)
                        if tm['role'] == 'tool' and tm.get('id') == t['id']
                    ]
                    assert len(matches) == 1
                    j, tm = matches[0]
                    writes.append({
                        'message_index': i, 'name': t['name'], 'arguments': t['arguments'],
                        'result_index': j, 'result_sha256': hashlib.sha256(
                            (tm.get('content') or '').encode()
                        ).hexdigest(),
                    })
            reservations = sorted({
                w['arguments']['reservation_id'] for w in writes
                if 'reservation_id' in w['arguments']
            })
            changes = {
                rid: {'initial': normalized_db['reservations'][rid], 'final': state['reservations'][rid]}
                for rid in reservations
                if normalized_db['reservations'][rid] != state['reservations'][rid]
            }
            row = {
                'run': label, 'path': str(trajectory), 'sha256': digest(trajectory),
                'manifest_sha256': manifests[label]['manifest_sha256'],
                'panel': telemetry['panel_name'], **record,
                'db': breakdown['DB'], 'communicate': breakdown['COMMUNICATE'],
                'offline_native_db_replay': result['db_check'],
                'first_customer': next(m['content'] for m in messages if m['role'] == 'user'),
                'writes': writes, 'reservation_changes': changes,
                'message_count': len(messages),
                'mixed_indices': [i for i, m in enumerate(messages)
                                  if m['role'] == 'assistant' and m.get('tool_calls')
                                  and (m.get('content') or '').strip()],
            }
            episodes.append(row)
            native_by_key[str(trajectory)] = native
            header = (f"\n## {label} | Task {record['task_id']} seed {record['seed']} | "
                      f"{telemetry['panel_name']} | Reward {record['native_reward']} "
                      f"DB {breakdown['DB']} COMM {breakdown['COMMUNICATE']}\n\n"
                      f"Source: {trajectory}\n\n")
            lines = [header]
            for i, m in enumerate(messages):
                if m['role'] in ('assistant', 'user'):
                    if m.get('content'):
                        lines.append(f"[{i}] {m['role']}: {m['content']}\n")
                    for t in m.get('tool_calls') or []:
                        lines.append(f"[{i}] CALL {t['name']} {json.dumps(t['arguments'])}\n")
                elif m['role'] == 'tool' and any(w['result_index'] == i for w in writes):
                    lines.append(f"[{i}] WRITE RESULT: {m.get('content')}\n")
            transcripts.append(''.join(lines))
    pairs = []
    for label in RUNS:
        rows = [r for r in episodes if r['run'] == label]
        baselines = {
            (r['task_id'], r['seed']): r for r in rows
            if 'incumbent' in r['panel']
        }
        for after in rows:
            if 'customer-candidate' not in after['panel'] and not re.search(r'candidate-\d+$', after['panel']):
                continue
            before = baselines[(after['task_id'], after['seed'])]
            pairs.append({
                'run': label, 'task_id': after['task_id'], 'seed': after['seed'],
                'candidate_panel': after['panel'], 'before_path': before['path'], 'after_path': after['path'],
                'native_before': before['native_reward'], 'native_after': after['native_reward'],
                'db_before': before['db'], 'db_after': after['db'],
                'communicate_before': before['communicate'], 'communicate_after': after['communicate'],
                'first_customer_same': before['first_customer'] == after['first_customer'],
                'first_divergence': first_divergence(
                    native_by_key[before['path']]['messages'], native_by_key[after['path']]['messages'],
                ),
            })
    summary = {
        'read_only': True, 'network_blocked': True, 'api_requests': 0,
        'V_H_task_objects_decoded': 0,
        'E_task_ids': list(manifests['ab-v2-s1']['task_panels']['E']),
        'completed_episodes': len(episodes),
        'runs': {label: {
            'directory': str(WORKTREE / 'experiments/runs' / name),
            'manifest_sha256': manifests[label]['manifest_sha256'],
            'evotau': manifests[label]['evotau'], 'upstream': manifests[label]['upstream'],
            'episode_count': sum(r['run'] == label for r in episodes),
        } for label, name in RUNS.items()},
        'reward_decomposition': dict(Counter(
            f"native={r['native_reward']};DB={r['db']};COMMUNICATE={r['communicate']}"
            for r in episodes
        )),
        'termination_counts': dict(Counter(r['termination_reason'] for r in episodes)),
        'native_db_and_communication_checks_reproduced': len(episodes),
        'episodes_with_mixed_text_tool_call': sum(bool(r['mixed_indices']) for r in episodes),
        'mixed_message_count': sum(len(r['mixed_indices']) for r in episodes),
        'recorded_hard_policy_protocol_violations': sum(
            r['hard_policy_protocol_violations'] for r in episodes
        ),
        'customer_pair_count': len(pairs),
        'customer_first_message_different': sum(not p['first_customer_same'] for p in pairs),
        'source_sha256': {
            name: digest(DATA / 'tau2/domains/airline' / name)
            for name in ('tasks.json', 'policy.md', 'db.json', 'split_tasks.json')
        },
        'limitations': [
            'Selected historical E10 only; not a random domain sample.',
            'Task/seed pairs are independent interactive rollouts, not causal replay.',
            '94 saved episodes are not 94 independent tasks.',
            'Native DB/COMMUNICATE replay is not a full policy compliance audit.',
        ],
    }
    write('inventory.json', {'summary': summary, 'episodes': episodes, 'customer_pairs': pairs})
    write('E-only-task-evaluation.json', tasks)
    (AUDIT / 'E-transcripts.md').write_text(''.join(transcripts))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
