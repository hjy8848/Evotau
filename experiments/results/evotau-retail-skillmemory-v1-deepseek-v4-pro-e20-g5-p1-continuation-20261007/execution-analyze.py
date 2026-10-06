"""Analysis only: read saved artifacts; never request provider calls."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path('/Users/spring/RSI/Evotau-alternating');sys.path.insert(0,str(ROOT/'src'))
from evotau.alternating_run import _api_usage_by_role
RUN_ID='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007'
P=ROOT/'experiments/runs'/RUN_ID

def read(path,default=None):
    if not path.exists():return default
    return json.loads(path.read_text())
def paired(old,new):
    a={r['task_id']:r for r in old};b={r['task_id']:r for r in new}
    assert len(a)==len(old)==20 and len(b)==len(new)==20 and set(a)==set(b)
    groups={k:[] for k in ('pass_to_fail','fail_to_pass','pass_to_pass','fail_to_fail')}
    for task_id,r in a.items():
        assert type(r['task_success']) is bool and type(b[task_id]['task_success']) is bool
        x,y=r['task_success'],b[task_id]['task_success']
        groups['pass_to_pass' if x and y else 'pass_to_fail' if x else 'fail_to_pass' if y else 'fail_to_fail'].append(task_id)
    return {'counts':{k:len(v) for k,v in groups.items()},'task_ids':groups}

def analyze():
    manifest=read(P/'manifest.json');state=read(P/'run-execution-state.json')
    result=read(P/'alternating-result.json',{});live=read(P/'api-usage-live.json')
    usage=result.get('provider_usage',live['provider_usage']);calls=result.get('api_usage_by_call_name',live['api_usage_by_call_name'])
    records=[read(f) for f in P.glob('episodes/*/episode-record.json')]
    incomplete=[read(f) for f in P.glob('episodes/*/incomplete-run.json')]
    refs=read(P/'episode-panel-references.json',{}).get('references',{})
    by_panel=defaultdict(list)
    for r in refs.values():by_panel[r['panel_name']].append(r)
    wire=[json.loads(s) for s in (P/'actual-provider-http.jsonl').read_text().splitlines()]
    req={r['event_id']:r for r in wire if r['event']=='request'}
    res={r['event_id']:r for r in wire if r['event']=='response'}
    per_model={}
    for model in ('deepseek-v4-flash','deepseek-v4-pro'):
        rs=[r for r in req.values() if r['body_parameters']['model']==model]
        responses=[res[r['event_id']] for r in rs if r['event_id'] in res]
        success=[r for r in responses if r['http_status']==200]
        per_model[model]={'requests':len(rs),'thinking_disabled':sum(r['body_parameters'].get('thinking',{}).get('type')=='disabled' for r in rs),'thinking_enabled':sum(r['body_parameters'].get('thinking',{}).get('type')=='enabled' for r in rs),'http_statuses':dict(Counter(r['http_status'] for r in responses)),'reasoning_tokens':sum(r['metadata'].get('reasoning_tokens') or 0 for r in success),'reasoning_usage_available':sum(r['metadata'].get('reasoning_tokens') is not None for r in success),'reasoning_content_responses':sum((r['metadata'].get('reasoning_content_chars') or 0)>0 for r in success),'empty_successful_responses':sum(r['metadata']['output_state']=='empty' for r in success),'actual_parameters':list({json.dumps(r['body_parameters'],sort_keys=True):r['body_parameters'] for r in rs}.values())}
    active={};peak=0
    for r in wire:
        if r['event']=='request':
            assert r['thread'] not in active.values()
            active[r['event_id']]=r['thread'];peak=max(peak,len(active))
        elif r['event'] in ('response','transport_failure'):active.pop(r['event_id'])
    assert peak<=1
    generations=[]
    for f in sorted(P.glob('generation-[0-9][0-9][0-9][0-9].json')):
        g=read(f);cp,sp=g['customer_phase'],g['service_phase'];i=g['generation']
        candidate=cp['candidates'][0]
        ref=by_panel[f'generation-{i}-customer-incumbent']
        detail={'generation':i,'incumbent_accuracy':cp['incumbent_accuracy'],'customer_candidate_accuracy':candidate['accuracy'],'selected_accuracy':cp['selected_accuracy'],'selected_customer':cp['selected_customer'],'candidate_strategy':candidate['strategy'],'customer_paired':paired(cp['incumbent_episodes'],candidate['episodes']),'incumbent_cache_reuse':sum(r['reused'] for r in ref),'incumbent_new_episodes':sum(not r['reused'] for r in ref),'service_phase':sp,'service_proposal':read(P/f'generation-{i:04d}-service-proposal.json'),'customer_before':g['customer_before'],'customer_after':g['customer_after'],'service_before':g['service_before'],'service_after':g['service_after'],'timing':g['timing']}
        assert cp['selected_customer']==(0 if candidate['accuracy']<cp['incumbent_accuracy'] else 'incumbent')
        if sp['operation']=='no_op':
            assert not sp['challenge_episodes'] and g['service_before']['strategy_id']==g['service_after']['strategy_id']
            detail['service_paired']=None
        else:
            old=cp['incumbent_episodes'] if cp['selected_customer']=='incumbent' else candidate['episodes']
            detail['service_paired']=paired(old,sp['challenge_episodes'])
            assert sp['accepted']==(sp['proposed_accuracy']>sp['old_accuracy'])
        generations.append(detail)
    logs=[json.loads(s) for f in P.rglob('provider-calls.jsonl') for s in f.read_text().splitlines()]
    response_ids=[r['response']['response_id'] for r in logs if r['api_success']]
    diagnostics={'wire_requests':len(req),'wire_responses':len(res),'provider_log_calls':len(logs),'provider_budget_calls':usage['attempts'],'wire_budget_and_log_reconciled':len(req)==len(logs)==usage['attempts'],'successful_response_ids_unique':len(response_ids)==len(set(response_ids)),'all_provider_retries_zero':all(r['request_args'].get('num_retries')==0 for r in logs),'all_requests_settled':not active,'budget_inflight_and_reserved_zero':usage.get('in_flight',0)==usage.get('reserved',0)==0,'panel_order_checks':{f'generation-{g["generation"]}':all([r['task_id'] for r in panel]==manifest['task_panels']['E'] for panel in (read(P/f'generation-{g["generation"]:04d}.json')['customer_phase']['incumbent_episodes'],read(P/f'generation-{g["generation"]:04d}.json')['customer_phase']['candidates'][0]['episodes'])) for g in generations}}
    summary={'experiment_id':RUN_ID,'status':state['status'],'manifest':manifest,'execution_state':state,'provider_usage':usage,'api_usage_by_call_name':calls,'api_usage_by_role':_api_usage_by_role(calls),'wire_by_model':per_model,'observed_peak_concurrent_provider_requests':peak,'episode_job_telemetry':result.get('episode_job_telemetry'),'unique_completed_episodes':len(records),'incomplete_attempts':len(incomplete),'partial_failures':incomplete,'panel_references':len(refs),'panel_reused_references':sum(r['reused'] for r in refs.values()),'panel_reference_counts':{k:len(v) for k,v in by_panel.items()},'completed_generations':len(generations),'generations':generations,'final_active_skill_memory':result.get('final_service'),'final_active_skill_provenance':result.get('final_service_provenance'),'skill_count_trajectory':[0]+[g['service_after']['skill_count'] for g in generations],'integrity':diagnostics,'timing':result.get('timing',{'total_wall_clock_seconds':state.get('total_wall_clock_seconds')}),'failure':state.get('failure')}
    origin=read(P/'continuation-provenance.json');parent=origin['parent_provider_usage']['provider_usage']
    summary['continuation']=origin
    summary['incremental_provider_usage']={k:usage[k]-parent[k] for k in ('attempts','successes','failures','prompt_tokens','completion_tokens')}
    summary['inherited_provider_usage']=parent
    summary['last_accepted_checkpoint']=read(ROOT/manifest['checkpoint_path'])
    summary['timing']['including_parent_wall_clock_seconds']=origin['parent_total_wall_clock_seconds']+state.get('total_wall_clock_seconds',0)
    return summary
if __name__=='__main__':
    summary=analyze()
    Path('/tmp/evotau-retail-g5-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','unique_completed_episodes','incomplete_attempts','completed_generations','wire_by_model','failure')},ensure_ascii=False,indent=2))
