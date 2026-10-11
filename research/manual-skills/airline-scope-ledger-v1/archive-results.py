"""Read-only archive of completed diagnostic, no provider calls or score edits."""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/runs/evotau-airline-human-scope-ledger-v1-e3-factorial-p2-20261011'
ARCHIVE = ROOT / 'research/results/airline-human-scope-ledger-v1-20261011'


def main():
    result = json.loads((RUN / 'result.json').read_text())
    state = json.loads((RUN / 'run-execution-state.json').read_text())
    if state['status'] != 'complete' or result['provider_usage']['in_flight']:
        raise ValueError('Only completed, settled execution can be archived')
    files = ['result.json', 'manifest.json', 'config.json', 'operator-launch.json',
             'run-execution-state.json', 'operator-working-tree-audit.json',
             'api-usage-live.json', 'actual-provider-http.jsonl']
    compact = []
    for condition, panel in result['panels'].items():
        files.append(f'panel-{condition}.json')
        for row in panel['rows']:
            e = row['episode']
            files.extend([e['trajectory_ref'], row['activation_trace']])
            native = json.loads((RUN / e['trajectory_ref']).read_text())
            trace = json.loads((RUN / row['activation_trace']).read_text())
            compact.append({'condition': condition, 'task_id': e['task_id'],
                            'native_reward': e['native_reward'], 'reward_breakdown': native['reward_info']['reward_breakdown'],
                            'termination_reason': e['termination_reason'],
                            'active_turns': [d['turn'] for d in trace['decisions'] if d['decision']['active_skill_ids']],
                            'native_ref': e['trajectory_ref'], 'activation_ref': row['activation_trace']})
    hashes = {}
    for name in files:
        source = RUN / name
        blob = source.read_bytes()
        if re.search(rb'sk-[a-zA-Z0-9]{15,}', blob):
            raise ValueError('Potential credential in archive source')
        target = ARCHIVE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[name] = hashlib.sha256(blob).hexdigest()
    (ARCHIVE / 'episode-summary.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2)+'\n')
    (ARCHIVE / 'archive-provenance.json').write_text(json.dumps({'source_run': str(RUN), 'copied_without_score_edits': True, 'sha256': hashes}, indent=2)+'\n')
    print(json.dumps({'archived_episodes': len(compact), 'calls': result['provider_usage']['attempts'], 'wall_seconds': result['wall_clock_seconds']}))


if __name__ == '__main__':
    main()
