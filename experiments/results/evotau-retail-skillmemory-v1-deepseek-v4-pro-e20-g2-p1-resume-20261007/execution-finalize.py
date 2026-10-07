"""One-shot archival of the existing run; never makes a model request."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
ROOT=Path('/Users/spring/RSI/Evotau-alternating');sys.path.insert(0,str(ROOT/'src'))
from evotau.tau_provenance import capture_code_provenance
from evotau.alternating_run import _api_usage_by_role
spec=importlib.util.spec_from_file_location('g5analysis','/tmp/evotau-retail-one-round-analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
s=module.analyze();rid=s['experiment_id'];m=s['manifest'];run=ROOT/'experiments/runs'/rid;arc=ROOT/'experiments/results'/rid
assert s['status'] in ('complete','failed','paused'), 'run must settle before archive'
assert capture_code_provenance().source_sha256==m['evotau']['source_sha256']=='965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e'
assert s['completed_generations']>=1 and s['unique_completed_episodes']>=60
assert s['integrity']['all_requests_settled'] and s['integrity']['wire_budget_and_log_reconciled']
assert s['integrity']['budget_inflight_and_reserved_zero'] and s['integrity']['successful_response_ids_unique']
if s['status']=='complete':assert s['completed_generations']==2
cp=ROOT/m['checkpoint_path'];checkpoint=json.loads(cp.read_text());assert checkpoint['manifest_sha256']==m['manifest_sha256']
assert checkpoint['completed_generation']==s['completed_generations']-1
flash=s['wire_by_model']['deepseek-v4-flash'];assert flash['requests']==flash['thinking_disabled'] and flash['reasoning_tokens']==0
assert all(s['api_usage_by_role'][r]['calls']==0 for r in ('reviewer','customer_judge','service_judge'))
# Refuse to mix snapshots. A single archive is produced after terminal state.
assert not arc.exists()
shutil.copytree(run,arc)
shutil.copy2(ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g2-p1-resume.yaml',arc/'config.yaml')
shutil.copy2(cp,arc/'checkpoint-snapshot.json')
for name in ('prepare','launch','progress','analyze','finalize','supervise'):
 f=Path('/tmp/evotau-retail-one-round-'+name+'.py')
 if f.exists():shutil.copy2(f,arc/('execution-'+name+'.py'))
(arc/'formal-attempt-summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
parent=s['continuation'];usage=s['provider_usage'];delta=s['incremental_provider_usage'];roles=s['api_usage_by_role'];parent_roles=_api_usage_by_role(parent['parent_provider_usage']['api_usage_by_call_name'])
report=f'''# Retail E20/G2/P1 — checkpoint continuation

Status: **{s['status']}**. Completed generations: **{s['completed_generations']}/2**. No V/H evaluation.

| Gen | Customer before | Customer after | S before acc | challenged acc | repaired acc | Service accepted |
|---|---|---|---:|---:|---:|---|
'''
for i in range(2):
 g=next((g for g in s['generations'] if g['generation']==i),None)
 if g is None:report+=f'| {i} | N/A | N/A | N/A | N/A | N/A | not completed |\n';continue
 sp=g['service_phase'];repair=sp.get('proposed_accuracy');r='skipped (NO_OP)' if repair is None else f'{repair:.0%}'
 report+=f"| {i} | {g['customer_before']['strategy_id']} | {g['customer_after']['strategy_id']} | {g['incumbent_accuracy']:.0%} | {g['selected_accuracy']:.0%} | {r} | {sp.get('accepted')} ({sp.get('operation')}) |\n"
report+=f'''
## Continuation conditions

Gen0 and 62 complete native episodes were imported from the interrupted G5 continuation. Gen1 reuses its existing Customer candidate (no new Customer Evolver request), retries only incomplete conditions, and stops after Gen1. All runtime and Evolver model args remain identical to the G5 continuation, including max_tokens=65536 for Pro. Gen0 came from the original output settings; this is a checkpoint resume, not a fresh replicate.

Target experiment: `{rid}`. Source commit: `{m['evotau']['git_commit']}`. Source SHA256: `{m['evotau']['source_sha256']}`. Manifest SHA256: `{m['manifest_sha256']}`. Config SHA256: `{m['config_sha256']}`.

E20 IDs: `{' '.join(m['task_panels']['E'])}`. Fixed seed1, candidate1, serial episodes, max_steps32, unlimited request budget, V/H disabled. Runtime Flash roles/args, Pro model/thinking/high-effort request, prompts, frozen source, policy/tools/native scoring and selection remain unchanged. Actual wire parameters are recorded; provider compliance with named high effort remains unverified.

`continuation-provenance.json` retains parent usage/timing and original artifact hashes; `resume-origin/` retains the interrupted G5 manifest/checkpoint/state/context/proposal and `continuation-origin/` retains earlier origin evidence. The cloned checkpoint, reference index and unfinished Gen1 stage/proposal manifest bindings were explicitly rebound; original Gen0 documents preserve parent provenance. Original run remains intact.

## API / time

| Role | Total calls (incl. parent) | New calls | Prompt tokens | Completion tokens | API seconds | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|
'''
for name in ('customer','service','evaluator','customer_evolver','service_evolver','reviewer','customer_judge','service_judge'):
 r=roles[name];old=parent_roles[name]
 report+=f"| {name} | {r['calls']} | {r['calls']-old['calls']} | {r['prompt_tokens']:,} | {r['completion_tokens']:,} | {r['total_elapsed_seconds']:.2f} | {r['average_elapsed_seconds']:.2f} |\n"
report+=f'''
Including parent: {usage['attempts']:,} API calls, {usage['prompt_tokens']:,} prompt tokens, {usage['completion_tokens']:,} completion tokens. New continuation: {delta['attempts']:,} calls, {delta['prompt_tokens']:,} prompt tokens, {delta['completion_tokens']:,} completion tokens.

Continuation wall-clock: {s['execution_state']['total_wall_clock_seconds']:.2f}s. Including parent active runtime: {s['timing']['including_parent_wall_clock_seconds']:.2f}s (excludes pause gaps). Unique complete episodes: {s['unique_completed_episodes']}; panel references: {s['panel_references']}; reused references: {s['panel_reused_references']}; incomplete attempts: {s['incomplete_attempts']}. Peak observed provider concurrency: {s['observed_peak_concurrent_provider_requests']}.

Flash actual thinking disabled {flash['thinking_disabled']}/{flash['requests']}; reasoning tokens {flash['reasoning_tokens']}; empty successful responses {flash['empty_successful_responses']}. Pro request/response metadata and role/call-name usage are fully recorded in the JSON summary and wire log.

## Per-generation proposals and paired outcomes
'''
for g in s['generations']:
 sp=g['service_phase'];report+=f"\n### Gen{g['generation']}\n\nIncumbent E accuracy {g['incumbent_accuracy']:.0%}; candidate {g['customer_candidate_accuracy']:.0%}; selected {g['selected_accuracy']:.0%}, selected Customer={g['selected_customer']}. Incumbent cache reuse {g['incumbent_cache_reuse']}/20, newly run {g['incumbent_new_episodes']}.\n\nCustomer strategy:\n\n{g['candidate_strategy']['text']}\n\nCustomer paired outcomes:\n\n"
 for name,ids in g['customer_paired']['task_ids'].items():report+=f"- {name}: {len(ids)}; `{', '.join(ids) or 'none'}`\n"
 report+=f"\nService operation {sp.get('operation')}, accepted {sp.get('accepted')}. Old E accuracy {sp.get('old_accuracy')}; candidate E accuracy {sp.get('proposed_accuracy')}. V disabled.\n\nService analysis:\n\n{sp.get('analysis','')}\n"
 if g['service_paired']:
  report+='\nService paired outcomes:\n\n'
  for name,ids in g['service_paired']['task_ids'].items():report+=f"- {name}: {len(ids)}; `{', '.join(ids) or 'none'}`\n"
 else:report+='\nNO_OP: candidate replay skipped; Service state unchanged.\n'
 report+='\nMemory before:\n\n```json\n'+json.dumps(g['service_before'],ensure_ascii=False,indent=2)+'\n```\n\nMemory after:\n\n```json\n'+json.dumps(g['service_after'],ensure_ascii=False,indent=2)+'\n```\n'
report+='\n## Last accepted active memory and provenance\n\n```json\n'+json.dumps({'service':checkpoint['service'],'provenance':checkpoint['service_provenance'],'skill_count_trajectory':s['skill_count_trajectory']},ensure_ascii=False,indent=2)+'\n```\n'
report+='\n## Failure / validity / resume\n\n```json\n'+json.dumps({'failure':s['failure'],'incomplete_attempts':s['incomplete_attempts'],'integrity':s['integrity']},ensure_ascii=False,indent=2)+'\n```\n\nFailures remain fail-closed. No automatic provider retries, model changes, JSON repair or skipping tasks. The original failed Gen1 Evolver call remains in inherited diagnostics and is distinguished from continuation failures by call ID and lifecycle state. Cache/checkpoint remain available for explicit resume. All tables use complete panels; no unknown result is counted as failure. Cross-Customer accuracy is not a same-condition Service improvement. V/H disabled, Retail only and seed1: no generalization claim.\n'
report+='\nFull native conversations, tools and evaluation are in `episodes/`; generation and SkillMemory artifacts, `actual-provider-http.jsonl`, checkpoint snapshot, provider logs, summary and SHA256 inventory are archived. Execution helpers only supervise/archive the existing runner; no algorithm/source edits and no extra model calls.\n'
(arc/'README.md').write_text(report)
for account in ('openai-api-key','openai-gpt-api-key'):
 r=subprocess.run(['security','find-generic-password','-s','inferaiapi.com/v1','-a',account,'-w'],capture_output=True)
 key=r.stdout.strip()
 if r.returncode==0 and key:
  for f in arc.rglob('*'):
   if f.is_file():assert key not in f.read_bytes(),f.name
files=sorted(f for f in arc.rglob('*') if f.is_file())
(arc/'archive-sha256.json').write_text(json.dumps({'schema_version':1,'files':{f.relative_to(arc).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
index=ROOT/'experiments/results/README.md';old=index.read_text()
line=f'\n[`{rid}/`]({rid}/README.md) is the explicit serial E20/G5 continuation: imported Gen0/62 complete native episodes and the existing Gen1 Customer candidate, retained model args, completed {s["completed_generations"]}/2 generations; terminal status {s["status"]}. Parent and incremental usage/provenance are distinguished.\n'
assert rid not in old
index.write_text(old+line)
def git(*args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
assert git('branch','--show-current')=='codex/alternating-evolution-refactor'
git('add','configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g2-p1-resume.yaml','experiments/results/README.md',str(arc.relative_to(ROOT)))
staged=git('diff','--cached','--name-only').splitlines()
assert all(f.startswith('experiments/results/') or f=='configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g2-p1-resume.yaml' for f in staged),'unrelated staged changes'
git('diff','--cached','--check')
git('commit','-m',f'Archive E20 one-additional-generation continuation ({s["status"]}, {s["completed_generations"]} generations)')
sha=git('rev-parse','HEAD')
git('-c','http.proxy=http://127.0.0.1:65533','push','origin','codex/alternating-evolution-refactor')
assert git('rev-parse','origin/codex/alternating-evolution-refactor')==sha
print(json.dumps({'status':s['status'],'completed_generations':s['completed_generations'],'commit':sha,'pushed':True,'archive':str(arc)},ensure_ascii=False))
