"""Flagship gateway diagnosis sweep; no benchmark episodes or retries."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/execution'))
from evotau.alternating import LLMAlternatingEvolvers
from evotau.budget import RequestBudget
from evotau.evolution_candidates import V2Providers
from evotau.provider_diagnostics import safe_error
from evotau.tau_provenance import sha256_json

os.chdir(ROOT)
out = ROOT / 'experiments/diagnostics/gateway-flagship-diagnoser-sweep-standard-gpt-20261008'
out.mkdir(parents=True, exist_ok=True)
if (out / "report.json").exists():
    raise RuntimeError("Existing probe report; refusing to duplicate requests")
key = subprocess.run(['security', 'find-generic-password', '-s', 'litellm',
                      '-a', 'litellm-api-key', '-w'], capture_output=True,
                     text=True, check=True).stdout.strip()
if not key:
    raise RuntimeError('Missing gateway credential')
os.environ['OPENAI_API_KEY'] = key
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request('http://10.130.138.46:8010/v1/models',
                 headers={'Authorization': 'Bearer ' + key}), timeout=30) as response:
    available = [m['id'] for m in json.load(response)['data']]
(out / 'models.json').write_text(json.dumps({'checked_at': datetime.now(UTC).isoformat(),
                               'model_ids': available}, indent=2) + '\n')
del key
source = ROOT / ('experiments/runs/evotau-v2-qwen37plus-inferai-v4pro-diagnoser-compressed-output64k-e20-v3-h5-g2-p1-20261008/'
                 'evolver-calls/631c38a0660742bc9fa3cfcfab5cc2f2/input.json')
context = json.loads(source.read_text())['context']
budget = RequestBudget(3)
budget.enable_live_usage(out / 'api-usage-live.json')
import httpx

from evotau.provider_diagnostics import response_metadata

original_send = httpx.Client.send
wire_lock = Lock()

def record(row):
    row['at'] = datetime.now(UTC).isoformat()
    with wire_lock, (out / 'actual-provider-http.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')

def send(client, request, *args, **kwargs):
    if request.url.host != '10.130.138.46' or request.method != 'POST':
        return original_send(client, request, *args, **kwargs)
    body = json.loads(request.content)
    record({'event': 'request', 'args': {k: body[k] for k in
            ('model', 'max_tokens', 'thinking', 'temperature', 'reasoning_effort') if k in body},
            'tools_count': len(body.get('tools') or [])})
    started = perf_counter()
    try:
        response = original_send(client, request, *args, **kwargs)
        response.read()
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        record({'event': 'response', 'model': body['model'], 'http_status': response.status_code,
                'metadata': response_metadata(payload), 'elapsed_seconds': perf_counter() - started})
        return response
    except Exception as error:
        record({'event': 'failure', 'model': body['model'], 'error': safe_error(error),
                'elapsed_seconds': perf_counter() - started})
        raise
httpx.Client.send = send
models = []
identifiers = ('openai/gpt-5.6', 'openai/gpt-5.5', 'openai/gpt-5.4')
for ident in identifiers:
    assert ident in available, ident
    args = {'api_base': 'http://10.130.138.46:8010/v1', 'max_tokens': 65536, 'timeout': 180}
    # Preserve exact gateway ID. Outer openai/ only selects the compatible client.
    models.append(('openai/' + ident, args))
report = {'started_at': datetime.now(UTC).isoformat(), 'source_input': str(source.relative_to(ROOT)),
          'context_sha256': sha256_json(context), 'models': models, 'request_cap': 3, 'max_parallel_requests': 1,
          'automatic_retries': 0, 'native_episodes': 0, 'results': [],
          'interpretation': 'Availability and strict diagnosis-schema check; no mutation quality claim or formal run model replacement.'}
report_path = out / 'report.json'
def publish():
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
publish()
def test_model(model, args):
    print(json.dumps({'stage': 'starting', 'model': model}), flush=True)
    directory = out / model.split('/')[-1]
    provider = V2Providers(LLMAlternatingEvolvers(model=model, model_args=args,
                          request_budget=budget, output_directory=directory))
    start = perf_counter()
    result = {'model': model, 'model_args': args}
    try:
        response = provider.diagnose(context)
        result.update(status='SCHEMA_VALID', cluster_count=len(response['clusters']),
                      response=response)
    except Exception as error:  # noqa: BLE001 -- isolate failures across explicitly authorized models
        result.update(status='FAILED', error_type=type(error).__name__, error=safe_error(error))
    result['elapsed_seconds'] = round(perf_counter() - start, 3)
    return result
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(test_model, model, args): model for model, args in models}
    for future in as_completed(futures):
        result = future.result()
        report['results'].append(result)
        publish()
        print(json.dumps({k: v for k, v in result.items() if k != 'response'}, ensure_ascii=False), flush=True)
report['ended_at'] = datetime.now(UTC).isoformat()
report['api_usage_by_call_name'] = budget.api_usage_by_call_name()
report['complete'] = True
publish()
print(json.dumps({'complete': True, 'report': str(report_path)}), flush=True)
