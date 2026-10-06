import json,pathlib,time
p=pathlib.Path('/Users/spring/RSI/Evotau-alternating/experiments/runs/evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007')
def read(f):return json.loads((p/f).read_text())
s=read('run-execution-state.json');u=read('api-usage-live.json')['provider_usage'];stages=sorted(p.glob('generation-*-stage.json'));stage=json.loads(stages[-1].read_text()) if stages else {}
gen=[]
for f in sorted(p.glob('generation-[0-9][0-9][0-9][0-9].json')):
 g=json.loads(f.read_text());c=g['customer_phase'];v=g['service_phase'];gen.append({'gen':g['generation'],'before':c['incumbent_accuracy'],'challenge':c['selected_accuracy'],'repair':v.get('proposed_accuracy'),'operation':v.get('operation'),'accepted':v.get('accepted')})
w=[json.loads(l) for l in (p/'actual-provider-http.jsonl').read_text().splitlines()];new=w[2398:]
request=next((r for r in reversed(new) if r['event']=='request'),None);response=next((r for r in reversed(new) if r['event']=='response'),None)
print(json.dumps({'status':s['status'],'wall_clock_seconds':s.get('total_wall_clock_seconds',0),'stage':{'generation':stage.get('generation'),'stage':stage.get('stage')},'complete_episodes':len(list(p.glob('episodes/*/episode-record.json'))),'new_calls':u['attempts']-1199,'new_prompt_tokens':u['prompt_tokens']-6497201,'new_completion_tokens':u['completion_tokens']-159632,'completed_generations':gen,'last_request':None if request is None else request['body_parameters'],'last_response':None if response is None else {'http_status':response['http_status'],'elapsed_seconds':response['elapsed_seconds'],'metadata':response['metadata']},'failure':s.get('failure')},ensure_ascii=False))
