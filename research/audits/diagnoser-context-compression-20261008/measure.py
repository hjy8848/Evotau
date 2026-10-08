"""Offline archived E20 diagnosis reconstruction; no provider calls."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import tiktoken

from evotau.evolution_context import (
    SERVICE_DIAGNOSIS_EVIDENCE_INSTRUCTIONS,
    build_service_diagnosis_evidence,
)
from evotau.tau_provenance import sha256_json

root = Path.cwd()
source = root / ('experiments/runs/evotau-v2-qwen37plus-inferai-v4pro-compressed-output64k-e20-v3-h5-g2-p1-20261008/'
                 'evolver-calls/1f4b4ea080a14f3094929a6fa1d5097a/input.json')
full_source = root / ('experiments/runs/evotau-v2-qwen37plus-gateway-dsv4flash-thinking-e20-v3-h5-g2-p1-20261008/'
                      'evolver-calls/9b6c93818047417aa0bfdcfda6731afb/input.json')
raw, full_raw = source.read_bytes(), full_source.read_bytes()
original = json.loads(raw)
full = json.loads(full_raw)['context']['task_interactions']
# These are the same cached native conditions, not scores imported from a different rollout.
lookup = {(r['task']['task_id'], r['seed']): r for r in full}
for row in original['context']['task_interactions']:
    matching = lookup[(row['task']['task_id'], row['seed'])]
    assert row['trajectory'] == matching['trajectory']
    assert row['native_evaluation'] == matching['native_evaluation']
    assert row['trajectory_ref'] == matching['trajectory_ref']
assert {(r['task_id'], r['seed'], r['task_success']) for r in original['context']['current_outcomes']} == {
    (r['task']['task_id'], r['seed'], r['native_evaluation']['task_success']) for r in full}
rows = build_service_diagnosis_evidence(full)
compressed = deepcopy(original)
compressed['context']['task_interactions'] = rows
compressed['context']['evidence_instructions'] = SERVICE_DIAGNOSIS_EVIDENCE_INSTRUCTIONS
compressed['input_sha256'] = sha256_json(compressed['context'])
checks = []
for old, new in zip(full, rows, strict=True):
    assert new['task'] == {'task_id': old['task']['task_id']}
    assert new['native_evaluation'] == old['native_evaluation']
    assert new['seed'] == old['seed']
    by_index = {item['evidence_ref']['projected_message_index']: item for item in new['trajectory']['messages']}
    for index, item in by_index.items():
        message = old['trajectory']['messages'][index]
        assert item['evidence_ref']['message_sha256'] == sha256_json(message)
        excerpt = item['excerpt']
        if excerpt is None:
            assert item['message'] == message
        elif excerpt['mode'] == 'exact_prefix_excerpt':
            assert item['message']['content'] == message['content'][:excerpt['content_retained_chars']]
        else:
            assert excerpt['mode'] == 'exact_json_fields'
            payload = json.loads(message['content'])
            assert json.loads(item['message']['content']) == {key: payload[key] for key in excerpt['retained_fields']}
            assert set(payload) == set(excerpt['retained_fields']) | set(excerpt['omitted_fields'])
    if new['representative_case']:
        for index, message in enumerate(old['trajectory']['messages']):
            if message.get('role') == 'user' or index in new['trajectory_overview']['decisive_tool_message_indices']:
                assert index in by_index
                assert by_index[index]['excerpt'] is None or by_index[index]['excerpt']['mode'] == 'exact_json_fields'
    checks.append({'task_id': new['task']['task_id'], 'seed': new['seed'],
                   'task_success': new['native_evaluation']['task_success'],
                   'representative_case': new['representative_case'], 'evidence_verified': True})
assert source.read_bytes() == raw and full_source.read_bytes() == full_raw
enc = tiktoken.get_encoding('cl100k_base')
def count(value):
    return len(enc.encode(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)))
old_tokens = count(original['system_prompt']) + count(original['context'])
new_tokens = count(compressed['system_prompt']) + count(compressed['context'])
report = {
    'sources': [{'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(data).hexdigest()}
                for path, data in ((source, raw), (full_source, full_raw))],
    'tokenizer': 'cl100k_base proxy, not provider billed tokens; excludes chat framing',
    'old_input_tokens_estimate': old_tokens, 'new_input_tokens_estimate': new_tokens,
    'reduction_percent': round(100 * (1 - new_tokens / old_tokens), 2),
    'overview_tokens': count([{k: v for k, v in row.items() if k != 'trajectory'} for row in rows]),
    'detailed_evidence_tokens': count([row['trajectory'] for row in rows]),
    'checks': checks, 'provider_requests': 0, 'native_artifacts_modified': False,
    'limitations': ['Diagnostic sufficiency and provider response success are unverified.',
                    'Exact prefix excerpts and explicit JSON field projections omit some original evidence.',
                    'Mutator input, native evaluation cost, gates and budget are unchanged.',
                    'Changed runtime fingerprint requires an explicit audited continuation; no live resume performed.']}
out = Path(__file__).parent
(out / 'measurement.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(out / 'compressed-input.json').write_text(json.dumps(compressed, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('checks', 'sources', 'limitations')}))
print([(r['task_id'], r['task_success']) for r in checks if r['representative_case']])
