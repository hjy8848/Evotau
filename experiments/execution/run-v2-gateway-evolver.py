"""Explicit mixed-provider launch: Keychain gateway auth, unchanged native runtime."""
import importlib.util
import json
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
import httpx
from evotau.provider_diagnostics import response_metadata, safe_error

ROOT = Path(__file__).resolve().parents[2]

def main():
    key = subprocess.run(['security', 'find-generic-password', '-s', 'litellm',
                          '-a', 'litellm-api-key', '-w'], capture_output=True,
                         text=True, check=True).stdout.strip()
    if not key:
        raise RuntimeError('LiteLLM gateway Keychain credential missing')
    original = httpx.Client.send
    lock = threading.Lock()
    def record(row):
        process = json.loads(Path('/tmp/evotau-v2-live-process.json').read_text())
        row['at'] = datetime.now(UTC).isoformat()
        with lock, (Path(process['output']) / 'actual-provider-http.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    def send(client, request, *args, **kwargs):
        if request.url.host != '10.130.138.46' or request.method != 'POST':
            return original(client, request, *args, **kwargs)
        body = json.loads(request.content)
        request.headers['Authorization'] = 'Bearer ' + key
        record({'event':'request', 'provider':'local-litellm-gateway',
                'args':{k:body[k] for k in ('model','thinking','reasoning_effort','temperature','max_tokens') if k in body},
                'tools_count':len(body.get('tools') or [])})
        start = perf_counter()
        try:
            response = original(client, request, *args, **kwargs)
            response.read()
            try: payload = response.json()
            except ValueError: payload = {}
            record({'event':'response','provider':'local-litellm-gateway',
                    'model':body.get('model'),'http_status':response.status_code,
                    'metadata':response_metadata(payload),'elapsed_seconds':perf_counter()-start})
            return response
        except BaseException as error:
            record({'event':'failure','provider':'local-litellm-gateway',
                    'error':safe_error(error),'elapsed_seconds':perf_counter()-start})
            raise
    httpx.Client.send = send
    path = Path(__file__).with_name('run-v2-live-evolution.py')
    spec = importlib.util.spec_from_file_location('frozen_live_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

if __name__ == '__main__':
    main()
