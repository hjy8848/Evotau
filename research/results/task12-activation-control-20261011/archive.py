"""Archive settled Task12 controls; no provider calls and no old-artifact writes."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
OLD = ROOT / 'research/results/airline-human-scope-ledger-v1-20261011'


def read(path):
    return json.loads(path.read_text())


def copy(source, target, hashes):
    blob = source.read_bytes()
    if re.search(rb'sk-[A-Za-z0-9]{15,}', blob):
        raise ValueError('Possible credential')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    hashes[str(target.relative_to(DEST))] = hashlib.sha256(blob).hexdigest()


def main():
    hashes, summary, excerpts = {}, {}, {}
    old = read(OLD / 'result.json')
    conditions = [('A', OLD, next(r['episode'] for r in old['panels']['C0S0']['rows'] if r['episode']['task_id'] == '12'))]
    for condition in ['B', 'C']:
        source = ROOT / f'experiments/runs/evotau-task12-activation-control-{condition.lower()}-seed1-20261011'
        assert read(source / 'run-execution-state.json')['status'] == 'complete'
        result = read(source / 'result.json')
        conditions.append((condition, source, result['records'][0]))
    for name, source, record in conditions:
        native_file = source / record['trajectory_ref']
        native = read(native_file)
        trace_file = native_file.with_name('skill-activation-trace.json')
        trace = read(trace_file)
        for filename in ['manifest.json', 'result.json']:
            copy(source / filename, DEST / name / filename, hashes)
        for filename in ['native-simulation.json', 'skill-activation-trace.json']:
            copy(native_file.with_name(filename), DEST / name / filename, hashes)
        if name != 'A':
            for filename in ['config.json', 'api-usage-live.json', 'actual-provider-http.jsonl', 'run-execution-state.json']:
                copy(source / filename, DEST / name / filename, hashes)
        assistants = [i for i,m in enumerate(native['messages']) if m['role'] == 'assistant' and i != 0]
        upgrades = [i for i in assistants if any(c['name'] == 'update_reservation_flights' and c['arguments']['cabin'] == 'business' for c in native['messages'][i].get('tool_calls') or [])]
        first = upgrades[0] if upgrades else None
        before = trace['decisions'][assistants.index(first)] if first is not None else None
        disclosed = [i for i,m in enumerate(native['messages']) if m['role'] == 'user' and '650' in (m.get('content') or '') and (first is None or i < first)]
        summary[name] = {'record': record, 'budget_disclosure_before_upgrade': disclosed,
                         'upgrade_message_indices': upgrades, 'critical_activation_decision': before,
                         'active_turns': [d['turn'] for d in trace['decisions'] if d['decision']['active_skill_ids']],
                         'total_decisions': len(trace['decisions']), 'reward_info': native['reward_info']}
        excerpts[name] = [{'message_index': i, 'original': m} for i,m in enumerate(native['messages']) if m['role'] in ('user','assistant','tool')]
    oldmanifest = read(OLD / 'manifest.json')
    manifest_b = read(DEST / 'B/manifest.json')
    fields = ['upstream','domain','seed','max_steps','role_models','role_model_args','source_blob_sha1','communication_enforcement']
    compatible = {k: oldmanifest[k] == manifest_b[k] for k in fields}
    compatible['runtime_source_sha256'] = oldmanifest['evotau']['runtime_source_sha256'] == manifest_b['evotau']['runtime_source_sha256']
    assert all(compatible.values())
    (DEST / 'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (DEST / 'original-message-excerpts.json').write_text(json.dumps(excerpts,ensure_ascii=False,indent=2)+'\n')
    (DEST / 'compatibility-audit.json').write_text(json.dumps({'verified_equal':compatible,
        'changed': ['independent directory', 'E panel subset Task12', 'request cap 60', 'C selector render_all_v1'],
        'historical_A_reused_as_reference_not_imported_as_new_score':True,
        'same_task_seed_not_same_actual_dialogue':True, 'backend_weights_not_fingerprinted':True,
        'old_B_not_budget_disclosed_before_charge':True},indent=2)+'\n')
    (DEST / 'archive-provenance.json').write_text(json.dumps({'source_sha256':hashes,'no_old_result_edits':True},indent=2)+'\n')
    for name,s in summary.items():
        print(name,s['budget_disclosure_before_upgrade'],s['active_turns'],s['record']['native_reward'])
    for name in ['B','C']:
        r=read(DEST/name/'result.json');print(name,r['usage'],r['wall_seconds'])


if __name__ == '__main__':
    main()
