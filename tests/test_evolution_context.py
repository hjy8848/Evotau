from copy import deepcopy

import pytest

from evotau.evolution_context import build_customer_evidence
from evotau.tau_provenance import sha256_json


def row(ident, success):
    return {'task': {'task_id': ident, 'user_scenario': 'Original immutable scenario'},
            'seed': 1, 'customer_strategy_id': 'c', 'service_strategy_id': 's',
            'trajectory_ref': f'episodes/{ident}/native-simulation.json',
            'native_evaluation': {'task_success': success, 'reward': int(success)},
            'trajectory': {'messages': [
                {'role': 'user', 'content': 'Please change two items.'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'read', 'name': 'get_order_details', 'arguments': {'order_id': 'x'}}]},
                {'role': 'tool', 'tool_call_id': 'read', 'content': 'x' * 40000},
                {'role': 'user', 'content': 'Actually, change only the lamp.'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'write', 'name': 'exchange_delivered_order_items',
                     'arguments': {'item_ids': ['cup', 'lamp']}}]},
                {'role': 'tool', 'tool_call_id': 'write', 'content': '{"status":"exchanged"}'},
            ], 'termination_reason': 'user_stop'}}


def test_all_outcomes_and_validator_scenarios_preserved_without_mutation():
    source = [row(str(i), i >= 3) for i in range(20)]
    before = deepcopy(source)
    result = build_customer_evidence(source)
    assert source == before
    assert result == build_customer_evidence(source)
    assert [r['task'] for r in result] == [r['task'] for r in source]
    assert [r['native_evaluation'] for r in result] == [r['native_evaluation'] for r in source]
    detailed = [r for r in result if r['representative_case']]
    assert len(detailed) == 4
    assert sum(r['native_evaluation']['task_success'] for r in detailed) == 1
    assert all(not r['trajectory']['messages'] for r in result if not r['representative_case'])


def test_confirmation_actions_and_results_exact_with_digest_linked_read_excerpt():
    source = [row('a', False), row('b', True)]
    result = build_customer_evidence(source)
    evidence = result[0]['trajectory']['messages']
    by_index = {v['evidence_ref']['projected_message_index']: v for v in evidence}
    for index in (0, 3, 4, 5):
        assert by_index[index]['message'] == source[0]['trajectory']['messages'][index]
        assert by_index[index]['excerpt'] is None
    for index, item in by_index.items():
        assert item['evidence_ref']['message_sha256'] == sha256_json(source[0]['trajectory']['messages'][index])
    assert by_index[2]['message']['content'] == 'x' * 1200
    assert by_index[2]['excerpt']['content_original_chars'] == 40000
    assert result[0]['trajectory_overview']['causal_failure_diagnosis'] is None


def test_explicit_read_tool_error_is_preserved_and_unknown_outcomes_fail_closed():
    source = [row('a', False), row('b', True)]
    source[0]['trajectory']['messages'][2]['content'] = '{"error":"cannot access order"}'
    result = build_customer_evidence(source)
    assert result[0]['trajectory_overview']['observed_tool_error_indices'] == [2]
    item = next(x for x in result[0]['trajectory']['messages'] if x['evidence_ref']['projected_message_index'] == 2)
    assert item['message'] == source[0]['trajectory']['messages'][2]
    source[0]['native_evaluation']['task_success'] = None
    with pytest.raises(ValueError, match='boolean'):
        build_customer_evidence(source)


def test_required_evidence_never_silently_dropped_to_meet_allowance():
    source = [row('a', False)]
    source[0]['trajectory']['messages'][0]['content'] = 'u' * 20000
    with pytest.raises(ValueError, match='allowance'):
        build_customer_evidence(source)


def test_legacy_missing_tool_ids_protects_adjacent_result_block_and_exact_fields():
    import json
    source = [row('a', False)]
    payload = {'order_id': 'x', 'status': 'exchanged', 'exchange_items': ['cup', 'lamp'],
               'items': [{'item_id': 'cup'}, {'item_id': 'lamp'}],
               'address': {'unused_address': 'x' * 10000}, 'return_items': None}
    source[0]['trajectory']['messages'][5] = {'role': 'tool', 'content': json.dumps(payload)}
    result = build_customer_evidence(source)[0]
    assert 5 in result['trajectory_overview']['decisive_tool_message_indices']
    item = next(x for x in result['trajectory']['messages'] if x['evidence_ref']['projected_message_index'] == 5)
    assert item['excerpt']['mode'] == 'exact_json_fields'
    assert item['excerpt']['omitted_fields'] == ['address']
    assert json.loads(item['message']['content']) == {k: v for k, v in payload.items() if k != 'address'}
    assert item['evidence_ref']['message_sha256'] == sha256_json(source[0]['trajectory']['messages'][5])
