"""Audited one-time Evolver-only continuation; zero provider calls."""
from pathlib import Path
import json, yaml, shutil, hashlib
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.budget import RequestBudget
from evotau.tau_provenance import sha256_json
root=Path.cwd()
config=root/'configs/v2-release-qwen37plus-inferai-v4pro-diagnoser-compressed-output64k-e20-v3-h5-g2-p1.yaml'
raw=yaml.safe_load(config.read_text());child=AlternatingManifest.from_mapping(raw)
source=root/'experiments/runs/evotau-v2-qwen37plus-inferai-v4pro-compressed-output64k-e20-v3-h5-g2-p1-20261008'
parent=json.loads((source/'manifest.json').read_text());target=child.to_document()
assert json.loads((source/'run-execution-state.json').read_text())['status']=='failed'
a=json.loads(json.dumps(parent));b=json.loads(json.dumps(target))
for k in ['manifest_sha256','config_sha256','experiment_id','output_path','checkpoint_path','provider_provenance','evotau','max_parallel_episodes']:a.pop(k,None);b.pop(k,None)
a['role_models'].pop('evolver');b['role_models'].pop('evolver');a['role_model_args'].pop('evolver');b['role_model_args'].pop('evolver')
assert a==b, 'native/search/gate condition changed'
import subprocess
# Audit exactly the Customer-context-only source delta; all native/evaluator/cache files stay identical.
changed = subprocess.check_output(['git','diff','--name-only',parent['evotau']['git_commit'],'--','src/evotau','pyproject.toml'],text=True).splitlines()
assert sorted(changed)==['src/evotau/evolution_context.py','src/evotau/skill_evolution.py'], changed
assert parent['role_models']==target['role_models']
old_args=parent['role_model_args']['evolver'];new_args=dict(target['role_model_args']['evolver'])
assert new_args==old_args and new_args['max_tokens']==65536
source_delta = {'parent_source_sha256':parent['evotau']['runtime_source_sha256'],
                'child_source_sha256':target['evotau']['runtime_source_sha256'],
                'changed_files':changed,
                'reviewed_scope':'Only Diagnoser evidence compression; Customer builder behavior covered by regression tests; native/evaluator/cache and Mutator code paths unchanged.'}
assert parent['max_parallel_episodes']==1 and target['max_parallel_episodes']==1
usage=json.loads((source/'api-usage-live.json').read_text());assert usage['provider_usage']['in_flight']==0
budget_parent=RequestBudget(100000);budget_parent.enable_live_usage(source/'api-usage-live.json')
data='/Users/spring/.cache/evotau/tau2-data-b7ea9074'
parent_raw=yaml.safe_load((root/'configs/v2-release-qwen37plus-dsv41flash-thinking-e20-v3-h5-g2-p1-continuation.yaml').read_text())
from evotau.tau_episode_runner import _publish_episode_completion
from evotau.records import EpisodeRecord
tasks=load_alternating_tasks(child,data,include_validation=True,include_heldout=False)
parent_records={}
for folder in (source/'episodes').iterdir():
 if (folder/'episode-completion.json').exists():
  _publish_episode_completion(folder,parent['manifest_sha256'])
  bundle=json.loads((folder/'episode-completion.json').read_text())
  record=EpisodeRecord.from_dict(bundle['record'])
  key=bundle['telemetry']['episode_key_sha256']
  assert key not in parent_records
  parent_records[key]=record
assert len(parent_records)==20
out=root/child.output_path;assert not (out/'run-execution-state.json').exists();out.mkdir(parents=True,exist_ok=True)
(out/'manifest.json').write_text(json.dumps(target,indent=2)+'\n')
def rebind(x):
 if isinstance(x,list):return [rebind(v) for v in x]
 if not isinstance(x,dict):return x
 y={k:rebind(v) for k,v in x.items()}
 if y.get('manifest_sha256')==parent['manifest_sha256']:y['manifest_sha256']=child.sha256
 for k in ['artifact_sha256','envelope_sha256','completion_sha256']:
  if k in y:y[k]=sha256_json({n:v for n,v in y.items() if n!=k})
 return y
copies=[]
for completion in source.glob('episodes/*/episode-completion.json'):
 for f in completion.parent.rglob('*'):
  assert not f.is_symlink()
  if not f.is_file():continue
  d=out/f.relative_to(source);d.parent.mkdir(parents=True,exist_ok=True)
  before=f.read_bytes()
  if f.suffix=='.json':d.write_text(json.dumps(rebind(json.loads(before)),indent=2)+'\n')
  else:shutil.copyfile(f,d)
  copies.append({'path':str(f.relative_to(source)),'parent_sha256':hashlib.sha256(before).hexdigest(),'child_sha256':hashlib.sha256(d.read_bytes()).hexdigest()})
for n in ['api-usage-live.json','actual-provider-http.jsonl']:shutil.copyfile(source/n,out/n)
budget=RequestBudget(100000);budget.enable_live_usage(out/'api-usage-live.json')
cr=TauBenchEpisodeRunner(manifest=child,config=raw,data_dir=data,request_budget=budget,output_directory=out,task_objects=tasks)
assert len(cr._completed_episode_cache)==20
for key,r in parent_records.items():assert cr._completed_episode_cache[key][0]==r
proof={'parent_run':str(source.relative_to(root)),'parent_manifest_sha256':parent['manifest_sha256'],'manifest_sha256':child.sha256,'imported_complete_episodes':20,'imported_stages':['g0000-customer_incumbent','g0000-customer_candidates','g0000-customer-validator-0','g0000-customer_selection'],'provider_calls_during_import':0,'parent_attempts_charged':budget.snapshot().attempts,'concurrency_change':[1,1],'evolver_change':['openai/deepseek-v4-pro','openai/deepseek-v4-pro'],'copies':copies,'source_compatibility_audit':source_delta,'failed_attempts_preserved_in_parent':True,'interpretation':'Evolver-only continuation; unchanged native scores, conditions and runtime model args; audited Diagnoser-context-only change. Previously generated Customer candidate and rejection are reused unchanged; only failed diagnosis is re-dispatched under user authorization; previous costs retained.'}
from evotau.evolution_artifacts import EvolutionJournal
journal=EvolutionJournal(source,parent['manifest_sha256'])
for name in proof['imported_stages']:
 old_path=journal.root/(name+'.json')
 document=journal.read(old_path)
 new_path=out/'evolution-v2'/(name+'.json');new_path.parent.mkdir(parents=True,exist_ok=True)
 new_path.write_text(json.dumps(rebind(document),indent=2)+'\n')
EvolutionJournal(out,child.sha256)
proof['completed_customer_stage_authorship']='InferAI V4 Pro; original request omitted max_tokens; original raw outputs and failed diagnosis preserved in parent.'
(out/'continuation-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
audit=root/'research/audits/v2-release-v4pro-diagnoser-compressed-20261008'
(audit/'frozen-manifest.json').write_text(json.dumps(target,indent=2)+'\n');(audit/'import-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ['imported_complete_episodes','provider_calls_during_import','parent_attempts_charged','concurrency_change']}))
