"""Frozen source invocation with sanitized wire observation; no request rewriting."""
from pathlib import Path
from datetime import datetime, UTC
from time import perf_counter
import json, os, runpy, subprocess, sys, threading, uuid

ROOT=Path('/Users/spring/RSI/Evotau-alternating')
RUN_ID='evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007'
CONFIG=ROOT/'configs/alternating-skill-memory-v1-deepseek-v4-pro-retail-e20-g5-p1-continuation.yaml'
OUTPUT=ROOT/'experiments/runs'/RUN_ID
# Secret stays inside this process and is never printed or written.
credential=subprocess.run(['security','find-generic-password','-s','inferaiapi.com/v1','-a','openai-api-key','-w'],capture_output=True,text=True,check=True).stdout.strip()
if not credential:
    raise RuntimeError('Missing InferAI Keychain credential')
os.environ['OPENAI_API_KEY']=credential
del credential
os.environ['TAU2_DATA_DIR']='/Users/spring/.cache/evotau/tau2-data-b7ea9074'
os.environ['HTTP_PROXY']='http://127.0.0.1:65533'
os.environ['HTTPS_PROXY']='http://127.0.0.1:65533'
os.environ['NO_PROXY']='localhost,127.0.0.1'
sys.path.insert(0,str(ROOT/'src'))
os.chdir(ROOT)
Path("/tmp/evotau-retail-g5-process.pid").write_text(str(os.getpid()))
import httpx
from evotau.provider_diagnostics import response_metadata, safe_error
send_original=httpx.Client.send
wire_lock=threading.Lock()

def record(event):
    event['at']=datetime.now(UTC).isoformat()
    event['thread']=threading.current_thread().name
    with wire_lock:
        # run_from_config already froze manifest before first request.
        with (OUTPUT/'actual-provider-http.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')

def observed_send(client,request,*args,**kwargs):
    if request.url.host!='inferaiapi.com' or request.method!='POST':
        return send_original(client,request,*args,**kwargs)
    body=json.loads(request.content)
    event_id=uuid.uuid4().hex
    selected={k:body[k] for k in ('model','thinking','reasoning_effort','temperature','max_tokens','max_completion_tokens','stream') if k in body}
    if isinstance(body.get('reasoning'),dict):
        selected['reasoning']={k:v for k,v in body['reasoning'].items() if k=='effort'}
    record({'event':'request','event_id':event_id,'method':request.method,'url':str(request.url.copy_with(query=None)), 'body_parameters':selected,'tools_count':len(body.get('tools') or []),'messages_count':len(body.get('messages') or [])})
    model=body.get('model')
    if model=='deepseek-v4-pro' and body.get('max_tokens',body.get('max_completion_tokens'))!=65536:
        record({'event':'condition_anomaly','event_id':event_id,'reason':'Pro output allowance was not passed'})
        raise RuntimeError('Continuation condition anomaly: Pro output allowance was not passed')
    if model=='deepseek-v4-flash' and body.get('thinking',{}).get('type')!='disabled':
        record({'event':'condition_anomaly','event_id':event_id,'reason':'Flash actual thinking parameter is not disabled'})
        raise RuntimeError('Formal experiment condition anomaly: Flash thinking is not disabled')
    started=perf_counter()
    try:
        response=send_original(client,request,*args,**kwargs)
    except BaseException as exc:
        record({'event':'transport_failure','event_id':event_id,'failure_type':type(exc).__name__,'failure_message':safe_error(exc),'elapsed_seconds':perf_counter()-started})
        raise
    response.read()
    try:
        payload=response.json()
    except ValueError:
        payload={}
    metadata=response_metadata(payload)
    record({'event':'response','event_id':event_id,'http_status':response.status_code,'metadata':metadata,'elapsed_seconds':perf_counter()-started})
    if response.is_success and model=='deepseek-v4-flash' and ((metadata.get('reasoning_tokens') or 0)>0 or (metadata.get('reasoning_content_chars') or 0)>0):
        record({'event':'condition_anomaly','event_id':event_id,'reason':'Flash response contains reasoning despite thinking disabled'})
        raise RuntimeError('Formal experiment condition anomaly: Flash reasoning leakage')
    return response

httpx.Client.send=observed_send
sys.argv=['evotau.alternating_run','--config',str(CONFIG),'--tau2-data-dir',os.environ['TAU2_DATA_DIR']]
runpy.run_module('evotau.alternating_run',run_name='__main__')
