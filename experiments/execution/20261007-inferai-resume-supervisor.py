"""Supervise the authorized live resume; one explicit Pro-to-GPT continuation, no retry loop."""
from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
import yaml
ROOT=Path('/Users/spring/RSI/Evotau-alternating')
FROZEN=Path('/Users/spring/.cache/evotau/frozen-runtime-965497fcfd0260fd')
PYTHON='/Users/spring/RSI/Evotau/.venv/bin/python'
SOURCE_SHA='965497fcfd0260fdf20cc2525fecc024d8853bd82d1baa44730f1b0ce3902f3e'
RUN='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g2-p1-resume-20261007'
STATUS=Path('/tmp/evotau-inferai-resume-supervisor.json')
sys.path.insert(0,str(FROZEN/'src'))
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.budget import RequestBudget
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import capture_code_provenance

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def status(state,**extra):write(STATUS,{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':state,'experiment_id':RUN,**extra})
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def archive(rid):
 run=ROOT/'experiments/runs'/rid;state=read(run/'run-execution-state.json');manifest=read(run/'manifest.json')
 assert state['status'] in ('complete','failed','paused')
 arc=ROOT/'experiments/results'/(rid+'-live-resume-attempt'+str(state['invocation_count'])+'-20261007')
 assert not arc.exists()
 shutil.copytree(run,arc)
 shutil.copy2(ROOT/manifest['checkpoint_path'],arc/'checkpoint-snapshot.json')
 config=next(p for p in (ROOT/'configs').glob('*.yaml') if yaml.safe_load(p.read_text()).get('experiment',{}).get('id')==rid)
 shutil.copy2(config,arc/'config.yaml')
 shutil.copy2('/tmp/evotau-inferai-preflight-20261007.json',arc/'provider-availability-preflight.json')
 generations=[read(f) for f in sorted(run.glob('generation-[0-9][0-9][0-9][0-9].json'))]
 usage=read(run/'api-usage-live.json')
 summary={'experiment_id':rid,'status':state['status'],'execution_state':state,'generations':generations,'api_usage':usage,'complete_episodes':len(list(run.glob('episodes/*/episode-record.json'))),'incomplete_attempts':len(list(run.glob('episodes/*/incomplete-run.json'))),'manifest':manifest,'continuation':read(run/'continuation-provenance.json')}
 write(arc/'live-resume-summary.json',summary)
 report=f"# InferAI live resume — {rid}\n\nStatus: **{state['status']}**. Completed generations: {len(generations)}/2. Serial E20, seed1, V/H disabled. This is a resumed experiment; completed native episodes are reused.\n\n| Gen | Incumbent | Challenged | Repaired | Service accepted |\n|---|---:|---:|---:|---|\n"
 for g in generations:
  c=g['customer_phase'];s=g['service_phase'];a=s.get('proposed_accuracy');report+=f"| {g['generation']} | {c['incumbent_accuracy']:.0%} | {c['selected_accuracy']:.0%} | {a:.0%} | {s['accepted']} |\n" if a is not None else f"| {g['generation']} | {c['incumbent_accuracy']:.0%} | {c['selected_accuracy']:.0%} | skipped NO_OP | {s['accepted']} |\n"
 report+='\nSee `live-resume-summary.json` for attempt-specific failures, API usage, timing and continuation provenance. Availability probes are separate from experiment API counts. Model change, if any, is recorded in the manifest and continuation evidence. Imported proposals preserve their original generating model. No unknown score is counted as failure. No V/H generalization claim.\n'
 (arc/'README.md').write_text(report)
 for account in ('openai-api-key','openai-gpt-api-key'):
  c=subprocess.run(['security','find-generic-password','-s','inferaiapi.com/v1','-a',account,'-w'],capture_output=True)
  if c.returncode==0 and c.stdout.strip():
   assert all(c.stdout.strip() not in f.read_bytes() for f in arc.rglob('*') if f.is_file())
 write(arc/'archive-sha256.json',{'files':{f.relative_to(arc).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in arc.rglob('*') if f.is_file()}})
 index=ROOT/'experiments/results/README.md';index.write_text(index.read_text()+f'\n[{arc.name}]({arc.name}/README.md): authorized live resume, terminal {state["status"]}, {len(generations)}/2 generations.\n')
 git('add',str(arc.relative_to(ROOT)),'experiments/results/README.md',str(config.relative_to(ROOT)))
 git('commit','-m',f'Archive InferAI live resume {state["status"]}')
 git('-c','http.proxy=http://127.0.0.1:65533','push','origin','codex/alternating-evolution-refactor')
 return git('rev-parse','HEAD')

def gpt_continuation(rid):
 # Model fallback is only for a failed Pro Evolver request, never for a Flash episode.
 parent=ROOT/'experiments/runs'/rid;old=read(parent/'manifest.json')
 assert old['evotau']['source_sha256']==capture_code_provenance().source_sha256==SOURCE_SHA
 new='evotau-retail-skillmemory-v1-pro-to-gpt61sol-e20-g2-p1-continuation-20261007'
 config=next(p for p in (ROOT/'configs').glob('*.yaml') if yaml.safe_load(p.read_text()).get('experiment',{}).get('id')==rid)
 cfg=yaml.safe_load(config.read_text());before=json.loads(json.dumps(cfg));e=cfg['experiment']
 e.update(id=new,output_path='experiments/runs/'+new,checkpoint_path='experiments/checkpoints/'+new+'.json')
 e['models']['evolver']='openai/gpt-6.1-sol'
 e['model_args']['evolver']={'api_base':'https://inferaiapi.com/v1','api_protocol':'responses','api_key_env':'INFERAI_API_KEY','reasoning_effort':'high'}
 assert all(e['models'][r]==before['experiment']['models'][r] and e['model_args'][r]==before['experiment']['model_args'][r] for r in ('agent','customer','evaluator'))
 m=AlternatingManifest.from_mapping(cfg);newrun=ROOT/e['output_path'];assert not newrun.exists()
 # Preflight once through the same GPT adapter; failure stops, without switching again.
 from evotau.inferai_responses import generate_text
 assert json.loads(generate_text(model=e['models']['evolver'],api_base='https://inferaiapi.com/v1',api_key_env='INFERAI_API_KEY',reasoning_effort='high',system_prompt='Return only {"ok":true} as JSON.',user_prompt='{}',call_name='inferai_gpt_fallback_preflight'))=={'ok':True}
 shutil.copytree(parent,newrun);origin=newrun/'gpt-fallback-origin';origin.mkdir()
 for name in ('manifest.json','run-context.json','run-execution-state.json','continuation-provenance.json'):
  if (newrun/name).exists():shutil.copy2(newrun/name,origin/name)
 oldcp=ROOT/before['experiment']['checkpoint_path'];shutil.copy2(oldcp,origin/'checkpoint.json');shutil.copy2(config,origin/'config.yaml')
 for name in ('run-context.json','run-execution-state.json','alternating-result.json'):
  if (newrun/name).exists():(newrun/name).unlink()
 write(newrun/'manifest.json',m.to_document())
 cp=read(oldcp);cp['manifest_sha256']=m.sha256;write(ROOT/e['checkpoint_path'],cp)
 rebound=[]
 for p in [newrun/'episode-panel-references.json',*newrun.glob('generation-*-stage.json'),*newrun.glob('generation-*-customer-proposals.json'),*newrun.glob('generation-*-service-proposal.json')]:
  if not p.exists():continue
  d=read(p)
  if d.get('manifest_sha256')==old['manifest_sha256']:
   shutil.copy2(p,origin/p.name);d['manifest_sha256']=m.sha256;write(p,d);rebound.append(p.name)
 newcfg=ROOT/'configs/alternating-skill-memory-v1-pro-to-gpt61sol-retail-e20-g2-p1-continuation.yaml';assert not newcfg.exists();newcfg.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
 parentstate=read(parent/'run-execution-state.json');provenance=read(parent/'continuation-provenance.json')
 write(newrun/'continuation-provenance.json',{'schema_version':1,'kind':'explicit_pro_to_gpt_fallback','source_experiment_id':rid,'source_manifest_sha256':old['manifest_sha256'],'target_experiment_id':new,'target_manifest_sha256':m.sha256,'frozen_source_sha256':SOURCE_SHA,'parent_provider_usage':read(parent/'api-usage-live.json'),'parent_total_wall_clock_seconds':provenance['parent_total_wall_clock_seconds']+parentstate['total_wall_clock_seconds'],'imported_complete_episodes':len(list(parent.glob('episodes/*/episode-record.json'))),'manifest_rebindings':rebound,'changes':{'evolver_model':{'old':'openai/deepseek-v4-pro','new':'openai/gpt-6.1-sol'},'evolver_args':e['model_args']['evolver']},'runtime_roles_model_args_unchanged':True,'interpretation':'Mixed-model continuation under explicit user authorization. Existing Customer proposals were generated by Pro and are retained. Only remaining Evolver calls use GPT. No pure-model comparison or generalization claim.'})
 tasks=load_alternating_tasks(m,'/Users/spring/.cache/evotau/tau2-data-b7ea9074',include_validation=False,include_heldout=False)
 budget=RequestBudget(None);budget.enable_live_usage(newrun/'api-usage-live.json')
 runner=TauBenchEpisodeRunner(manifest=m,config=cfg,data_dir='/Users/spring/.cache/evotau/tau2-data-b7ea9074',request_budget=budget,output_directory=newrun,task_objects=tasks)
 assert len(runner._completed_episode_cache)==len(list(parent.glob('episodes/*/episode-record.json')))
 launcher=Path('/tmp/evotau-retail-live-resume-launch.py').read_text().replace(rid,new).replace(str(config.relative_to(ROOT)),str(newcfg.relative_to(ROOT)))
 # Responses adapter uses urllib: observe its actual sanitized payload separately.
 launcher=launcher.replace("sys.argv=['evotau.alternating_run'",'''from evotau import inferai_responses
responses_original=inferai_responses._post_responses
def observed_responses(url,payload,api_key,*,timeout):
    event_id=uuid.uuid4().hex
    record({'event':'request','event_id':event_id,'method':'POST','url':url,'body_parameters':{'model':payload['model'],'reasoning':payload['reasoning']},'tools_count':0,'messages_count':len(payload['input'])})
    started=perf_counter()
    try:
        result=responses_original(url,payload,api_key,timeout=timeout)
    except BaseException as exc:
        record({'event':'transport_failure','event_id':event_id,'failure_type':type(exc).__name__,'failure_message':safe_error(exc),'elapsed_seconds':perf_counter()-started})
        raise
    record({'event':'response','event_id':event_id,'http_status':200,'metadata':response_metadata(result),'elapsed_seconds':perf_counter()-started})
    return result
inferai_responses._post_responses=observed_responses
sys.argv=['evotau.alternating_run' ''')
 launchpath=Path('/tmp/evotau-retail-gpt-fallback-launch.py');launchpath.write_text(launcher)
 log=open('/tmp/evotau-retail-gpt-fallback.log','ab')
 process=subprocess.Popen([PYTHON,str(launchpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return new,process

if __name__=='__main__':
 try:
  status('supervising');fallback_used=False
  while True:
   state=read(ROOT/'experiments/runs'/RUN/'run-execution-state.json')
   if state['status']=='running':
    pid=int(Path('/tmp/evotau-retail-one-round-process.pid').read_text());os.kill(pid,0);time.sleep(15);continue
   if state['status'] not in ('complete','failed','paused'):raise RuntimeError('Unexpected terminal state')
   failure=state.get('failure') or {};call=failure.get('call_name') or ''
   if state['status']=='failed' and 'evolver' in call and not fallback_used:
    # Preserve and publish the original failure before creating the changed identity.
    archive(RUN);RUN,process=gpt_continuation(RUN);fallback_used=True;status('gpt_fallback_launched',process_pid=process.pid);time.sleep(15);continue
   sha=archive(RUN);status('archived_and_pushed',experiment_status=state['status'],commit=sha);break
 except BaseException as error:
  status('needs_attention',failure_type=type(error).__name__,failure_message=str(error)[:600]);raise
