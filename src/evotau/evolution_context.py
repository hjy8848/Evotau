"""Deterministic reflection evidence; never rewrite native artifacts or scores."""
import json
from copy import deepcopy

from .tau_provenance import sha256_json


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _calls(message):
    return message.get('tool_calls') or []


def _read_only(name):
    # Evidence prioritization only. Unknown tools are conservatively kept in full.
    return isinstance(name, str) and (name.startswith(('get_', 'find_')) or name == 'calculate')


def _ref(row, index, message):
    return {'projected_message_index': index, 'message_sha256': sha256_json(message)}


def _excerpt(message, limit):
    value = deepcopy(message)
    content = value.get('content')
    if isinstance(content, str) and len(content) > limit:
        value['content'] = content[:limit]
        return value, {'content_original_chars': len(content), 'content_retained_chars': limit,
                       'content_sha256': sha256_json(content), 'mode': 'exact_prefix_excerpt'}
    return value, None


def _action_result(message):
    """Exact JSON field projection of bulky Retail order results, never an LLM summary."""
    content = message.get('content')
    if not isinstance(content, str) or len(content) <= 1200:
        return deepcopy(message), None
    try:
        payload = json.loads(content)
    except ValueError:
        return deepcopy(message), None
    if not isinstance(payload, dict) or not {'order_id', 'status'} <= set(payload):
        return deepcopy(message), None
    omitted = [key for key in ('address', 'fulfillments', 'payment_history') if key in payload]
    if not omitted:
        return deepcopy(message), None
    fields = {key: value for key, value in payload.items() if key not in omitted}
    value = deepcopy(message)
    value['content'] = json.dumps(fields, ensure_ascii=False, sort_keys=True)
    return value, {'mode': 'exact_json_fields', 'retained_fields': sorted(fields),
                   'omitted_fields': omitted, 'content_sha256': sha256_json(content)}


def _build_evidence(rows, *, representative_cases=4, case_chars=12000, selection=None):
    """Keep every task/outcome/scenario, plus bounded, digest-linked raw case evidence.

    Indices address the existing flattened trajectory projection, not native message
    indices. Message digests permit matching back to native messages/tool submessages.
    Excerpts and omitted messages are explicit; no causal diagnosis is fabricated.
    """
    if representative_cases < 2 or case_chars < 1000:
        raise ValueError('Reflection evidence needs contrast cases and a usable case allowance')
    rows = deepcopy(list(rows))
    failed = [i for i, row in enumerate(rows) if row['native_evaluation']['task_success'] is False]
    passed = [i for i, row in enumerate(rows) if row['native_evaluation']['task_success'] is True]
    if len(failed) + len(passed) != len(rows):
        raise ValueError('Reflection requires complete boolean native outcomes')
    order = lambda i: (str(rows[i]['task']['task_id']), rows[i]['seed'])
    # Stable outcome-stratified selection, with a success control whenever available.
    if selection is None:
        selection = sorted(failed, key=order)[:representative_cases - bool(passed)]
        selection += sorted(passed, key=order)[:representative_cases - len(selection)]
        if len(selection) < min(representative_cases, len(rows)):
            selection += [i for i in sorted(failed, key=order) if i not in selection][
                :representative_cases - len(selection)]
    selected = set(selection)
    result = []
    for index, row in enumerate(rows):
        messages = row['trajectory']['messages']
        path, actions, action_ids = [], set(), set()
        for mi, message in enumerate(messages):
            for call in _calls(message):
                name = call.get('name')
                path.append({'name': name, 'projected_message_index': mi})
                if not _read_only(name):
                    actions.add(mi)
                    action_ids.add(call.get('id'))
        # Older flattened projections lost ToolMessage.id. Keep the whole adjacent
        # tool-result block when any dispatched call changes state; never guess a
        # one-to-one pairing inside that block.
        for call_index in list(actions):
            following = call_index + 1
            while following < len(messages) and messages[following].get('role') == 'tool':
                actions.add(following)
                following += 1
        errors = []
        for mi, message in enumerate(messages):
            if message.get('role') == 'tool':
                try:
                    payload = json.loads(message.get('content') or '')
                except (ValueError, TypeError):
                    payload = None
                explicit_error = isinstance(payload, dict) and (
                    bool(payload.get('error')) or payload.get('success') is False)
                if explicit_error:
                    errors.append(mi)
                if message.get('tool_call_id') in action_ids or explicit_error:
                    actions.add(mi)
        users = [mi for mi, message in enumerate(messages) if message.get('role') == 'user']
        events = []
        for kind, mi in [('initial_user_message', users[0] if users else None),
                         ('latest_user_message', users[-1] if users else None)]:
            if mi is not None:
                value, omission = _excerpt(messages[mi], 80)
                events.append({'type': kind, 'content': value.get('content'),
                               'content_is_prefix_excerpt': omission is not None,
                               'projected_message_index': mi})
        overview = {k: v for k, v in row.items() if k not in
                    {'trajectory', 'customer_strategy_id', 'service_strategy_id'}}
        overview['trajectory_overview'] = {
            'message_count': len(messages), 'projection_sha256': sha256_json(row['trajectory']),
            'tool_path': {'names': [v['name'] for v in path],
                          'message_indices': [v['projected_message_index'] for v in path]},
            'interaction_events': events,
            'decisive_tool_message_indices': sorted(actions),
            'observed_tool_error_indices': errors,
            'causal_failure_diagnosis': None,
        }
        overview['representative_case'] = index in selected
        evidence, omitted = [], []
        if index in selected:
            priority = sorted(actions) + [mi for mi in users if mi not in actions]
            dialogue = [mi for mi, message in enumerate(messages)
                        if message.get('role') == 'assistant' and mi not in actions]
            priority += dialogue
            priority += [mi for mi in range(len(messages))
                         if mi not in actions and mi not in users and mi not in dialogue]
            used = 0
            for mi in priority:
                message = messages[mi]
                # Action parameters/results remain exact. Read results are explicitly excerpted.
                if mi in actions and message.get('role') == 'tool':
                    value, excerpt = _action_result(message)
                elif mi in actions or mi in users:
                    value, excerpt = deepcopy(message), None
                else:
                    value, excerpt = _excerpt(message, 1200 if message.get('role') == 'tool' else 4000)
                item = {'message': value, 'excerpt': excerpt,
                        'evidence_ref': _ref(row, mi, message)}
                size = _size(item)
                if used + size > case_chars:
                    if mi in actions or mi in users:
                        raise ValueError('Decisive tool/user evidence exceeds reflection case allowance')
                    omitted.append({'projected_message_index': mi})
                    continue
                evidence.append(item)
                used += size
            evidence.sort(key=lambda item: item['evidence_ref']['projected_message_index'])
        overview['trajectory'] = {'messages': evidence,
                                  'omitted_message_refs': sorted(omitted, key=lambda x: x['projected_message_index']),
                                  'termination_reason': row['trajectory'].get('termination_reason')}
        result.append(overview)
    return result


CUSTOMER_EVIDENCE_INSTRUCTIONS = (
    'Each row trajectory_ref is the source file for its message references. All panel tasks and native outcomes are listed. Representative cases contain raw evidence; '
    'other cases contain structural overviews only. Cite task IDs and message evidence_refs in '
    'your proposal rationale. Indices refer to the flattened projection; digests link to source '
    'messages. Explicit prefix excerpts, selected JSON fields and omitted messages are not complete evidence. Do not '
    'invent omitted facts or treat an observed failure as an established root cause.'
)


def build_customer_evidence(rows, *, representative_cases=4, case_chars=12000):
    return _build_evidence(rows, representative_cases=representative_cases, case_chars=case_chars)


def build_service_diagnosis_evidence(rows, *, representative_cases=4, case_chars=12000):
    """All E outcomes; deterministic failure cases and visible-tool-path success controls.

    Only observed evidence is admitted. Hidden task metadata and strategy text are
    excluded structurally, even if a caller accidentally passes Customer rows.
    """
    safe = [
        {'task': {'task_id': row['task']['task_id']},
         **{key: deepcopy(row[key]) for key in
            ('seed', 'native_evaluation', 'trajectory_ref', 'trajectory')}}
        for row in rows
    ]
    order = lambda i: (str(safe[i]['task']['task_id']), safe[i]['seed'])
    failed = sorted([i for i, row in enumerate(safe)
                     if row['native_evaluation']['task_success'] is False], key=order)
    passed = [i for i, row in enumerate(safe)
              if row['native_evaluation']['task_success'] is True]

    def tools(i):
        return {call.get('name') or (call.get('function') or {}).get('name')
                for message in safe[i]['trajectory']['messages'] for call in _calls(message)} - {None}

    failure_tools = set().union(*(tools(i) for i in failed))
    passed.sort(key=lambda i: (-len(tools(i) & failure_tools), order(i)))
    chosen = failed[:representative_cases - bool(passed)]
    chosen += passed[:representative_cases - len(chosen)]
    chosen += [i for i in failed if i not in chosen][:representative_cases - len(chosen)]
    return _build_evidence(safe, representative_cases=representative_cases,
                           case_chars=case_chars, selection=chosen)


SERVICE_DIAGNOSIS_EVIDENCE_INSTRUCTIONS = (
    CUSTOMER_EVIDENCE_INSTRUCTIONS +
    ' Only observed interactions are provided; hidden Customer scenarios are unavailable. '
    'All listed task IDs may be referenced by the existing diagnosis schema. '
    'A structural overview alone does not establish a failure mechanism. '
    'Use detailed failure evidence and passing controls to support a cluster; '
    'do not infer missing dialogue, hidden intentions or causal categories from tool names or rewards.'
)
