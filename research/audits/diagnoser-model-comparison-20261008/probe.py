"""Four isolated representative diagnosis requests; no benchmark episodes or retries."""
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
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
out = ROOT / 'experiments/diagnostics/diagnoser-model-comparison-20261008'
out.mkdir(parents=True, exist_ok=True)
if (out / "report.json").exists():
    raise RuntimeError("Existing probe report; refusing to duplicate requests")
for env, account in [('OPENAI_API_KEY', 'openai-api-key'), ('INFERAI_API_KEY', 'openai-gpt-api-key')]:
    key = subprocess.run(['security', 'find-generic-password', '-s', 'inferaiapi.com/v1',
                          '-a', account, '-w'], capture_output=True, text=True, check=True).stdout.strip()
    if not key:
        raise RuntimeError('Missing credential group')
    os.environ[env] = key
    del key
source = ROOT / ('experiments/runs/evotau-v2-qwen37plus-inferai-v4pro-diagnoser-compressed-output64k-e20-v3-h5-g2-p1-20261008/'
                 'evolver-calls/631c38a0660742bc9fa3cfcfab5cc2f2/input.json')
context = json.loads(source.read_text())['context']
budget = RequestBudget(4)
budget.enable_live_usage(out / 'api-usage-live.json')
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments/execution' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
launcher = load('probe_launcher', 'run-v2-live-evolution.py')
wire = load('probe_wire', 'run-v2-activation-morphology-smoke.py')
route = launcher.WireOutput(out)
wire.install_transport(route)
launcher.install_responses_observer(route)
models = [
    ('openai/glm-5.2', {'api_base': 'https://inferaiapi.com/v1', 'max_tokens': 65536}),
    ('openai/kimi-k2.7', {'api_base': 'https://inferaiapi.com/v1', 'max_tokens': 65536}),
    ('openai/qwen3.7-max', {'api_base': 'https://inferaiapi.com/v1', 'max_tokens': 65536}),
    ('openai/gpt-6.1-sol', {'api_base': 'https://inferaiapi.com/v1', 'api_protocol': 'responses',
                          'api_key_env': 'INFERAI_API_KEY', 'reasoning_effort': 'high'}),
]
report = {'started_at': datetime.now(UTC).isoformat(), 'source_input': str(source.relative_to(ROOT)),
          'context_sha256': sha256_json(context), 'models': models, 'request_cap': 4,
          'automatic_retries': 0, 'native_episodes': 0, 'results': [],
          'interpretation': 'Availability and strict diagnosis-schema check; no mutation quality claim or formal run model replacement.'}
report_path = out / 'report.json'
def publish():
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
publish()
for model, args in models:
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
    report['results'].append(result)
    publish()
    print(json.dumps({k: v for k, v in result.items() if k != 'response'}, ensure_ascii=False), flush=True)
report['ended_at'] = datetime.now(UTC).isoformat()
report['api_usage_by_call_name'] = budget.api_usage_by_call_name()
report['complete'] = True
publish()
print(json.dumps({'complete': True, 'report': str(report_path)}), flush=True)
