"""Audited one-time Evolver-only continuation; zero provider calls."""
from pathlib import Path
import json, yaml, shutil, hashlib
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.budget import RequestBudget
from evotau.tau_provenance import sha256_json
root=Path.cwd()
config=root/'configs/v2-release-qwen37plus-dsv41flash-thinking-e20-v3-h5-g2-p1-continuation.yaml'
raw=yaml.safe_load(config.read_text());child=AlternatingManifest.from_mapping(raw)
source=root/'experiments/runs/evotau-v2-qwen37plus-gpt61sol-retail-e20-v3-h5-g2-p1-continuation-20261008'
parent=json.loads((source/'manifest.json').read_text());target=child.to_document()
assert json.loads((source/'run-execution-state.json').read_text())['status']=='failed'
a=json.loads(json.dumps(parent));b=json.loads(json.dumps(target))
for k in ['manifest_sha256','config_sha256','experiment_id','output_path','checkpoint_path','provider_provenance','evotau','max_parallel_episodes']:a.pop(k,None);b.pop(k,None)
a['role_models'].pop('evolver');b['role_models'].pop('evolver');a['role_model_args'].pop('evolver');b['role_model_args'].pop('evolver')
assert a==b, 'native/search/gate condition changed'
assert parent['evotau']['runtime_source_sha256']==target['evotau']['runtime_source_sha256']
assert parent['max_parallel_episodes']==1 and target['max_parallel_episodes']==1
usage=json.loads((source/'api-usage-live.json').read_text());assert usage['provider_usage']['in_flight']==0
budget_parent=RequestBudget(100000);budget_parent.enable_live_usage(source/'api-usage-live.json')
data='/Users/spring/.cache/evotau/tau2-data-b7ea9074'
parent_raw=yaml.safe_load((root/'configs/v2-release-qwen37plus-gpt61sol-e20-v3-h5-g2-p1-continuation.yaml').read_text())
pm=AlternatingManifest.from_mapping(parent_raw).bind_saved_provenance(parent)
tasks=load_alternating_tasks(pm,data,include_validation=True,include_heldout=False)
pr=TauBenchEpisodeRunner(manifest=pm,config=parent_raw,data_dir=data,request_budget=budget_parent,output_directory=source,task_objects=tasks)
assert len(pr._completed_episode_cache)==20
out=root/child.output_path;assert not out.exists();out.mkdir(parents=True)
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
for key,r in pr._completed_episode_cache.items():assert cr._completed_episode_cache[key]==r
proof={'parent_run':str(source.relative_to(root)),'parent_manifest_sha256':parent['manifest_sha256'],'manifest_sha256':child.sha256,'imported_complete_episodes':20,'imported_stages':[],'provider_calls_during_import':0,'parent_attempts_charged':budget.snapshot().attempts,'concurrency_change':[1,1],'evolver_change':['openai/gpt-6.1-sol','openai/deepseek-v4.1-flash'],'copies':copies,'failed_attempts_preserved_in_parent':True,'interpretation':'Evolver-only continuation; unchanged native scores, conditions, runtime model args and source. No GPT candidate exists; parent failed GPT attempt remains charged and unscored.'}
(out/'continuation-provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
audit=root/'research/audits/v2-release-dsv41flash-evolver-20261008'
(audit/'frozen-manifest.json').write_text(json.dumps(target,indent=2)+'\n');(audit/'import-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ['imported_complete_episodes','provider_calls_during_import','parent_attempts_charged','concurrency_change']}))
