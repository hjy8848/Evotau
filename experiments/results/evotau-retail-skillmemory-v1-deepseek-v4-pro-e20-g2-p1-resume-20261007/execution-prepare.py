from pathlib import Path
import copy,hashlib,json,shutil,sys,yaml
ROOT=Path('/Users/spring/RSI/Evotau-alternating');sys.path.insert(0,str(ROOT/'src'))
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.tau_provenance import capture_code_provenance,sha256_json
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.budget import RequestBudget
from evotau.alternating import _context_episodes,_service_context_fields,_validate_customer_proposal_document
from evotau.strategies import PromptStrategy
from evotau.service_skills import ServiceSkillMemory
from evotau.records import EpisodeRecord
OLD='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007';NEW='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p1-resume-20261007'
oldcfg=ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g5-p1-continuation.yaml';newcfg=ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g2-p1-resume.yaml'
oldrun=ROOT/'experiments/runs'/OLD;newrun=ROOT/'experiments/runs'/NEW
cfg=yaml.safe_load(oldcfg.read_text());before=copy.deepcopy(cfg);ex=cfg['experiment']
ex.update(id=NEW,generations=2,output_path='experiments/runs/'+NEW,checkpoint_path='experiments/checkpoints/'+NEW+'.json')
assert not newrun.exists() and not newcfg.exists()
assert ex['model_args']==before['experiment']['model_args'] and ex['models']==before['experiment']['models']
assert {k:v for k,v in ex.items() if k not in ('id','generations','output_path','checkpoint_path')}=={k:v for k,v in before['experiment'].items() if k not in ('id','generations','output_path','checkpoint_path')}
newcfg.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True));m=AlternatingManifest.from_mapping(cfg);oldm=json.loads((oldrun/'manifest.json').read_text());oldcp=ROOT/before['experiment']['checkpoint_path'];checkpoint=json.loads(oldcp.read_text())
assert capture_code_provenance().source_sha256==m.evotau_source_sha256==oldm['evotau']['source_sha256']
assert checkpoint['completed_generation']==0 and checkpoint['manifest_sha256']==oldm['manifest_sha256']
assert len(list((oldrun/'episodes').glob('*/episode-record.json')))==62
shutil.copytree(oldrun,newrun);origin=newrun/'resume-origin';origin.mkdir()
for name in ('manifest.json','run-context.json','run-execution-state.json','generation-0001-stage.json','generation-0001-customer-proposals.json','episode-panel-references.json','continuation-provenance.json'):
 shutil.copy2(newrun/name,origin/name)
shutil.copy2(oldcfg,origin/'config.yaml');shutil.copy2(oldcp,origin/'checkpoint.json')
for name in ('run-context.json','run-execution-state.json'):(newrun/name).unlink()
(newrun/'manifest.json').write_text(json.dumps(m.to_document(),ensure_ascii=False,indent=2)+'\n')
checkpoint['manifest_sha256']=m.sha256;cp_new=ROOT/ex['checkpoint_path'];cp_new.write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2)+'\n')
rebound=('generation-0001-stage.json','generation-0001-customer-proposals.json','episode-panel-references.json')
for name in rebound:
 p=newrun/name;d=json.loads(p.read_text());assert d['manifest_sha256']==oldm['manifest_sha256'];d['manifest_sha256']=m.sha256;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
parentstate=json.loads((oldrun/'run-execution-state.json').read_text());priororigin=json.loads((oldrun/'continuation-provenance.json').read_text());parentusage=json.loads((oldrun/'api-usage-live.json').read_text())
imported=[f for f in oldrun.rglob('*') if f.is_file() and f.relative_to(oldrun).as_posix() not in (*rebound,'manifest.json','run-context.json','run-execution-state.json','continuation-provenance.json')]
assert all(f.read_bytes()==(newrun/f.relative_to(oldrun)).read_bytes() for f in imported)
evidence={'schema_version':1,'kind':'explicit_one_remaining_generation_resume','source_experiment_id':OLD,'source_manifest_sha256':oldm['manifest_sha256'],'target_experiment_id':NEW,'target_manifest_sha256':m.sha256,'frozen_source_sha256':m.evotau_source_sha256,'imported_completed_generation':0,'imported_complete_episodes':62,'total_target_generations':2,'remaining_generations':[1],'changes':{'generations':{'old':5,'new':2},'identity_paths':'new one-round continuation identity/output/checkpoint'},'all_model_args_unchanged':True,'runtime_roles_model_args_unchanged':True,'evolution_prompts_unchanged':True,'V_H_loaded':False,'parent_provider_usage':parentusage,'parent_total_wall_clock_seconds':priororigin['parent_total_wall_clock_seconds']+parentstate['total_wall_clock_seconds'],'imported_file_sha256':{f.relative_to(oldrun).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in imported},'manifest_rebindings':list(rebound)+['checkpoint.manifest_sha256'],'interpretation':'Explicit resume limited to Gen1: retain the existing Customer candidate and 62 complete episodes, rerun only incomplete conditions, and stop normally after Gen1. Gen0 had the original Evolver output setting; Gen1 max_tokens=65536. Original failures remain evidence; no automatic retries.'}
(newrun/'continuation-provenance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
# Prove that the existing mutation input and cache are valid without model calls.
tasks=load_alternating_tasks(m,'/Users/spring/.cache/evotau/tau2-data-b7ea9074',include_validation=False,include_heldout=False)
assert set(tasks)==set(m.evolution_task_ids) and len(tasks)==20
budget=RequestBudget(None);budget.enable_live_usage(newrun/'api-usage-live.json')
runner=TauBenchEpisodeRunner(manifest=m,config=cfg,data_dir='/Users/spring/.cache/evotau/tau2-data-b7ea9074',request_budget=budget,output_directory=newrun,task_objects=tasks)
assert len(runner._completed_episode_cache)==62
stage=json.loads((newrun/'generation-0001-stage.json').read_text());runs=tuple(EpisodeRecord.from_dict(x) for x in checkpoint['final_evolution_episodes'])
customer=PromptStrategy(checkpoint['customer']['text']);service=ServiceSkillMemory.from_mapping(checkpoint['service'])
context={'generation':1,'task_interactions':_context_episodes(runs,runner,tasks),'incumbent_accuracy':sum(r.task_success for r in runs)/len(runs),'service_policy':runner.service_policy_text,'current_customer_strategy':customer.text,'accuracy_history':checkpoint['history']};context.update(_service_context_fields(service,m.service_carrier))
proposal=json.loads((newrun/'generation-0001-customer-proposals.json').read_text());_validate_customer_proposal_document(proposal,generation=1,manifest_sha256=m.sha256,customer=customer,service=service,input_sha256=sha256_json(context))
assert budget.snapshot().attempts==parentusage['provider_usage']['attempts']
print(json.dumps({'experiment_id':NEW,'start_generation':1,'stop_after_generation':1,'cached_episodes':62,'existing_customer_candidate':'validated and reused','manifest_sha256':m.sha256,'source_unchanged':True},ensure_ascii=False))
