"""Rebuild archived Customer context without provider calls or modifying source evidence."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import tiktoken

from evotau.evolution_context import (
    CUSTOMER_EVIDENCE_INSTRUCTIONS,
    build_customer_evidence,
)
from evotau.tau_provenance import sha256_json

root = Path.cwd()
source = root / ('experiments/runs/evotau-v2-qwen37plus-gateway-dsv4flash-thinking-e20-v3-h5-g2-p1-20261008/'
                 'evolver-calls/9b6c93818047417aa0bfdcfda6731afb/input.json')
raw = source.read_bytes()
original = json.loads(raw)
context = original['context']
rows = build_customer_evidence(context['task_interactions'])
compressed = deepcopy(original)
compressed['context']['task_interactions'] = rows
compressed['context']['evidence_instructions'] = CUSTOMER_EVIDENCE_INSTRUCTIONS
compressed['input_sha256'] = sha256_json(compressed['context'])
checks = []
for old, new in zip(context['task_interactions'], rows, strict=True):
    assert old['task'] == new['task']
    assert old['native_evaluation'] == new['native_evaluation']
    assert old['seed'] == new['seed']
    by_index = {m['evidence_ref']['projected_message_index']: m for m in new['trajectory']['messages']}
    for index, item in by_index.items():
        message = old['trajectory']['messages'][index]
        assert sha256_json(message) == item['evidence_ref']['message_sha256']
        if item['excerpt'] is None:
            assert message == item['message']
        elif item['excerpt']['mode'] == 'exact_prefix_excerpt':
            assert item['message']['content'] == message['content'][:item['excerpt']['content_retained_chars']]
        else:
            assert item['excerpt']['mode'] == 'exact_json_fields'
            raw_fields = json.loads(message['content'])
            kept = json.loads(item['message']['content'])
            assert kept == {key: raw_fields[key] for key in item['excerpt']['retained_fields']}
            assert set(raw_fields) == set(kept) | set(item['excerpt']['omitted_fields'])
    if new['representative_case']:
        for index, message in enumerate(old['trajectory']['messages']):
            if message.get('role') == 'user' or index in new['trajectory_overview']['decisive_tool_message_indices']:
                assert index in by_index
                excerpt = by_index[index]['excerpt']
                if message.get('role') != 'tool' or excerpt is None:
                    assert by_index[index]['message'] == message
                    assert excerpt is None
                else:
                    assert excerpt['mode'] == 'exact_json_fields'
    checks.append({'task_id': old['task']['task_id'], 'task_success': old['native_evaluation']['task_success'],
                   'representative_case': new['representative_case'],
                   'exact_or_prefix_evidence_verified': True,
                   'all_user_action_parameters_and_retained_result_fields_exact': True if new['representative_case'] else None,
                   'retained_messages': len(by_index),
                   'omitted_message_indices': [r['projected_message_index'] for r in new['trajectory']['omitted_message_refs']]})
assert source.read_bytes() == raw
enc = tiktoken.get_encoding('cl100k_base')
def count(value):
    return len(enc.encode(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)))
old_tokens = count(original['system_prompt']) + count(context)
new_tokens = count(compressed['system_prompt']) + count(compressed['context'])
report = {'source_input': str(source.relative_to(root)), 'source_file_sha256': hashlib.sha256(raw).hexdigest(),
          'tokenizer': 'cl100k_base proxy estimate; not provider billed DeepSeek tokens; excludes chat framing',
          'old_input_tokens_estimate': old_tokens, 'new_input_tokens_estimate': new_tokens,
          'reduction_percent': round(100 * (1 - new_tokens / old_tokens), 2),
          'overview_including_full_customer_scenarios_tokens': count([{k:v for k,v in r.items() if k != 'trajectory'} for r in rows]),
          'detailed_evidence_tokens': count([r['trajectory'] for r in rows]),
          'all_panel_tasks_visible': len(rows), 'representative_cases': sum(r['representative_case'] for r in rows),
          'provider_requests': 0, 'native_artifacts_modified': False, 'checks': checks,
          'limitations': ['Overview exceeds aspirational 3k–5k because full Customer scenarios and evidence references remain.',
                          'All three observed E failures have exact user/action parameters and selected action-result fields; diagnostic sufficiency is not yet proved by a live Evolver.',
                          'Read-only results are excerpted; bulky action results omit address/fulfillments/payment_history with explicit field lists and original digests; some assistant/read messages are omitted.',
                          'Service Diagnoser/Mutator inputs and runtime evaluation cost are unchanged.',
                          'New runtime source fingerprint; no old frozen experiment is automatically resumed.']}
out = Path(__file__).parent
(out / 'compressed-input.json').write_text(json.dumps(compressed, ensure_ascii=False, indent=2) + '\n')
(out / 'measurement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('old_input_tokens_estimate','new_input_tokens_estimate','reduction_percent',
                                       'overview_including_full_customer_scenarios_tokens','detailed_evidence_tokens',
                                       'all_panel_tasks_visible','representative_cases','provider_requests')}))
