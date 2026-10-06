from pathlib import Path
import copy,hashlib,json,shutil,sys,yaml
ROOT=Path('/Users/spring/RSI/Evotau-alternating');sys.path.insert(0,str(ROOT/'src'))
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.tau_provenance import capture_code_provenance
OLD='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p1-formal-20261006'
NEW='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007'
oldcfg=ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g2-p1-formal.yaml'
newcfg=ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g5-p1-continuation.yaml'
oldrun=ROOT/'experiments/runs'/OLD;newrun=ROOT/'experiments/runs'/NEW
cfg=yaml.safe_load(oldcfg.read_text());before=copy.deepcopy(cfg);ex=cfg['experiment']
ex['id']=NEW;ex['generations']=5;ex['output_path']='experiments/runs/'+NEW;ex['checkpoint_path']='experiments/checkpoints/'+NEW+'.json'
ex['model_args']['evolver']['max_tokens']=65536
assert not newrun.exists()
if newcfg.exists(): assert yaml.safe_load(newcfg.read_text())==cfg
assert ex['models']==before['experiment']['models'] and ex['task_selection']==before['experiment']['task_selection']
for role in ('agent','customer','evaluator'):assert ex['model_args'][role]==before['experiment']['model_args'][role]
newcfg.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
m=AlternatingManifest.from_mapping(cfg);oldm=json.loads((oldrun/'manifest.json').read_text())
assert capture_code_provenance().source_sha256==m.evotau_source_sha256==oldm['evotau']['source_sha256']
tasks=load_alternating_tasks(m,'/Users/spring/.cache/evotau/tau2-data-b7ea9074',include_validation=False,include_heldout=False)
assert set(tasks)==set(m.evolution_task_ids) and len(tasks)==20
assert m.generations==5 and m.max_parallel_episodes==1 and m.customer_candidates==1 and m.evolution_fitness_seed==1 and not m.run_validation and not m.run_heldout
cp_path=ROOT/before['experiment']['checkpoint_path'];checkpoint=json.loads(cp_path.read_text());assert checkpoint['completed_generation']==0 and checkpoint['manifest_sha256']==oldm['manifest_sha256']
assert len(list((oldrun/'episodes').glob('*/episode-record.json')))==60
shutil.copytree(oldrun,newrun)
origin=newrun/'continuation-origin';origin.mkdir()
for name in ('manifest.json','run-context.json','run-execution-state.json','generation-0001-stage.json'):
 shutil.move(str(newrun/name),str(origin/name))
shutil.copy2(oldcfg,origin/'config.yaml');shutil.copy2(cp_path,origin/'checkpoint.json')
(newrun/'manifest.json').write_text(json.dumps(m.to_document(),ensure_ascii=False,indent=2)+'\n')
checkpoint['manifest_sha256']=m.sha256
cp_new=ROOT/ex['checkpoint_path'];cp_new.write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2)+'\n')
refs_path=newrun/'episode-panel-references.json';refs=json.loads(refs_path.read_text());assert refs['manifest_sha256']==oldm['manifest_sha256'];refs['manifest_sha256']=m.sha256
refs_path.write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
# Preserve all imported episode and Gen0 artifacts byte-for-byte. Only the
# continuation checkpoint and reference index bind to the new manifest.
imported=[f for f in oldrun.rglob('*') if f.is_file() and f.relative_to(oldrun).parts[0] not in {'manifest.json','run-context.json','run-execution-state.json','generation-0001-stage.json','episode-panel-references.json'}]
assert all(f.read_bytes()==(newrun/f.relative_to(oldrun)).read_bytes() for f in imported)
parentstate=json.loads((oldrun/'run-execution-state.json').read_text());parentusage=json.loads((oldrun/'api-usage-live.json').read_text())
evidence={'schema_version':1,'kind':'explicit_checkpoint_continuation','source_experiment_id':OLD,'source_manifest_sha256':oldm['manifest_sha256'],'target_experiment_id':NEW,'target_manifest_sha256':m.sha256,'frozen_source_sha256':m.evotau_source_sha256,'imported_completed_generation':0,'imported_complete_episodes':60,'total_target_generations':5,'remaining_generations':[1,2,3,4],'changes':{'generations':{'old':2,'new':5},'evolver.max_tokens':{'old':None,'new':65536},'identity_paths':'new independent continuation identity/output/checkpoint'},'runtime_roles_model_args_unchanged':True,'evolution_prompts_unchanged':True,'V_H_loaded':False,'parent_provider_usage':parentusage,'parent_total_wall_clock_seconds':parentstate['total_wall_clock_seconds'],'imported_file_sha256':{f.relative_to(oldrun).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in imported},'manifest_rebindings':{'checkpoint':'only manifest_sha256; original checkpoint preserved in continuation-origin','episode-panel-references.json':'only manifest_sha256; all references unchanged'},'interpretation':'Gen0 was generated under the original Evolver output settings. Gen1 onward request max_tokens=65536. This is an explicit continuation with an execution-argument change, not a fresh five-generation replicate. Inherited usage remains included; incremental usage must be reported separately.'}
(newrun/'continuation-provenance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'experiment_id':NEW,'manifest_sha256':m.sha256,'source_sha256':m.evotau_source_sha256,'reuse_episodes':60,'start_generation':1,'generations':5,'evolver_args':m.role_model_args_dict['evolver'],'E_only_preflight':'passed'},ensure_ascii=False))
