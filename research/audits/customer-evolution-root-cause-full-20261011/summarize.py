"""Produce exact references for the manually adjudicated, narrow audit findings.

This does not implement a general policy judge. Counts below describe named checks
on this saved E10 corpus; they are not population compliance estimates.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/Users/spring/.cache/evotau/tau2-data-b7ea9074')


def read(path):
    return json.loads(path.read_text())


def source(episode, index):
    path = Path(episode['path'])
    native = read(path)
    # Role keys within the messages array appear at indentation six.
    role_lines = [i for i, line in enumerate(path.read_text().splitlines(), 1)
                  if re.match(r'^      "role":', line)]
    assert len(role_lines) == len(native['messages'])
    return {
        'path': str(path), 'line': role_lines[index], 'original_message_index': index,
        'native_sha256': episode['sha256'], 'message': native['messages'][index],
    }


def slim_source(episode, index):
    ref = source(episode, index)
    message = ref['message']
    ref['message'] = {key: message.get(key) for key in ('role', 'content', 'tool_calls')}
    return ref


def main():
    inventory = read(ROOT / 'inventory.json')
    episodes, pairs = inventory['episodes'], inventory['customer_pairs']
    db = read(DATA / 'tau2/domains/airline/db.json')
    task39 = [r for r in episodes if r['task_id'] == '39']
    illegal = [r for r in task39 if any(
        w['name'] == 'cancel_reservation' and w['arguments']['reservation_id'] == 'MSJ4OA'
        for w in r['writes']
    )]
    assert len(illegal) == 7 and all(r['native_reward'] == 1 for r in illegal)
    task34_attempts = [r for r in episodes if r['task_id'] == '34'
                       and r['native_reward'] == 1 and r['writes']]
    assert len(task34_attempts) == 5
    assert all(not r['reservation_changes'] for r in task34_attempts)
    failed_attempts = []
    for episode in task34_attempts:
        native = read(Path(episode['path']))
        for w in episode['writes']:
            content = native['messages'][w['result_index']]['content']
            assert content.startswith('Error:')
            failed_attempts.append({**slim_source(episode, w['message_index']),
                                    'tool_result': content})
    ab = [r for r in episodes if r['run'] == 'ab-v2-s1']
    a = {r['task_id']: r for r in episodes if r['run'] == 'ab-v1-s1' and 'incumbent' in r['panel']}
    b = {r['task_id']: r for r in ab if 'incumbent' in r['panel']}
    passive_aa = []
    for tid, left in a.items():
        right = b[tid]
        ltele = read(Path(left['path']).parent / 'run-telemetry.json')
        rtele = read(Path(right['path']).parent / 'run-telemetry.json')
        assert ltele['episode_key'] == rtele['episode_key']
        assert ltele['rendered_prompt_sha256'] == rtele['rendered_prompt_sha256']
        lmsg = read(Path(left['path']))['messages']
        rmsg = read(Path(right['path']))['messages']
        passive_aa.append({
            'task_id': tid, 'seed': left['seed'], 'same_condition_and_prompt_hashes': True,
            'before_path': left['path'], 'after_path': right['path'],
            'first_customer_same': left['first_customer'] == right['first_customer'],
            'native_before': left['native_reward'], 'native_after': right['native_reward'],
            'any_semantic_divergence': [
                (m['role'], m.get('content')) for m in lmsg
            ] != [(m['role'], m.get('content')) for m in rmsg],
        })
    v2 = {panel: sorted([r for r in ab if r['panel'] == panel], key=lambda r: int(r['task_id']))
          for panel in {r['panel'] for r in ab}}
    key_refs = []
    for row in ab:
        if row['task_id'] == '39':
            indices = [21, 22, 23] + [w['message_index'] for w in row['writes']]
        elif row['task_id'] == '7':
            indices = ([1, 7, 10, 12, 19, 32] if 'candidate-1' in row['panel']
                       else [1, 9, 12, 19, 32] if 'incumbent' in row['panel']
                       else [1, 7, 10, 17, 30])
        elif row['task_id'] == '15':
            indices = ([15, 16, 17] if 'incumbent' in row['panel']
                       else [17, 18, 19] if 'candidate-1' in row['panel'] else [17, 22, 23])
        elif row['task_id'] == '11':
            indices = ([13, 15, 21, 23, 24, 27] if 'candidate-0' in row['panel'] else [1, 12, 15])
        elif row['task_id'] in ('0', '1'):
            native = read(Path(row['path']))
            indices = [i for i, m in enumerate(native['messages'])
                       if m['role'] == 'user' or any(t['name'] == 'transfer_to_human_agents'
                                                   for t in m.get('tool_calls') or [])]
        else:
            continue
        for index in indices:
            key_refs.append({'run': row['run'], 'panel': row['panel'], 'task_id': row['task_id'],
                             **slim_source(row, index)})
    output = {
        'adjudication_scope': 'Named observable checks, not an exhaustive all-policy judge',
        'native_pass_with_noncovered_insurance_cancellation': {
            'corpus_episodes': len(episodes), 'distinct_tasks': 10,
            'all_cancel_calls': sum(w['name'] == 'cancel_reservation' for r in episodes for w in r['writes']),
            'cancel_task_episode_count': sum(any(w['name'] == 'cancel_reservation' for w in r['writes']) for r in episodes),
            'eligible_for_manual_reason_check': 7, 'confirmed_violating_episodes': 7,
            'affected_distinct_tasks': ['39'],
            'evidence': [{k: r[k] for k in ('run', 'path', 'sha256', 'panel', 'native_reward')} for r in illegal],
        },
        'native_pass_with_failed_write_attempts_task34': {
            'task_episode_denominator': 10, 'native_pass_episodes': 9,
            'episodes_with_failed_attempts': 5, 'failed_attempt_count': len(failed_attempts),
            'actual_db_change': False,
            'interpretation': 'Backend rejects attempts; DB score cannot certify decision/path correctness.',
            'evidence': failed_attempts,
        },
        'native_fail_db_correct': [
            {k: r[k] for k in ('run', 'task_id', 'path', 'sha256', 'panel', 'db', 'communicate')}
            for r in episodes if r['native_reward'] == 0 and r['db'] == 1
        ],
        'same_binary_failure_but_db_regression': [p for p in pairs if p['native_before'] == p['native_after'] == 0
                                                and p['db_before'] == 1 and p['db_after'] == 0],
        'v2_native_panel_results': {
            panel: {'successes': sum(r['native_reward'] == 1 for r in rows), 'episodes': len(rows),
                    'outcomes': {r['task_id']: {'native': r['native_reward'], 'DB': r['db'],
                                               'COMMUNICATE': r['communicate']} for r in rows}}
            for panel, rows in v2.items()
        },
        'v2_pairs_first_customer_different': sum(not p['first_customer_same'] for p in pairs if p['run'] == 'ab-v2-s1'),
        'v2_pair_count': sum(p['run'] == 'ab-v2-s1' for p in pairs),
        'passive_existing_AA': passive_aa,
        'passive_AA_summary': {
            'task_count': 10, 'seed_count': 1,
            'first_customer_different': sum(not x['first_customer_same'] for x in passive_aa),
            'dialogue_diverged': sum(x['any_semantic_divergence'] for x in passive_aa),
            'native_flips': sum(x['native_before'] != x['native_after'] for x in passive_aa),
            'not_a_statistical_noise_calibration': True,
        },
        'task7_cost_crosscheck': {
            'initial_future_reservations': {rid: sum(p['amount'] for p in db['reservations'][rid]['payment_history'])
                                            for rid in ('7WPL39', 'XEHM4B', '59XX6W', '3EMQJ6')},
            'initial_sum': 1628, 'remaining_after_both_cancellations': 708,
            'meaning_of_other_and_query_timing_requires_adjudication': True,
        },
        'write_tool_counts': dict(Counter(w['name'] for r in episodes for w in r['writes'])),
        'key_message_refs': key_refs,
        'inventory_sha256': hashlib.sha256((ROOT / 'inventory.json').read_bytes()).hexdigest(),
    }
    (ROOT / 'findings.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k in (
        'passive_AA_summary', 'v2_pairs_first_customer_different', 'v2_pair_count',
        'write_tool_counts', 'task7_cost_crosscheck',
    )}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
