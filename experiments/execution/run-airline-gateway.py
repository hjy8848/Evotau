"""Frozen Airline gateway execution; native gateway key and official Evolver are isolated."""

import argparse
import hashlib
import json
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from evotau.aa_diagnostic import run_independent_repetitions
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks, run_from_config
from evotau.phase0 import load_config
from evotau.provider_diagnostics import response_metadata, safe_error
from evotau.release_recovery import frozen_run_lock
from evotau.skill_evolution_config import validate_v2_promotion_readiness
from evotau.tau_provenance import write_manifest_once


def credential(service, account):
    value = subprocess.run(
        ['security', 'find-generic-password', '-s', service, '-a', account, '-w'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not value:
        raise RuntimeError(f'Keychain credential unavailable: {service}')
    return value


def install_transport(output, gateway_key, evolver_key=None):
    original = httpx.Client.send
    lock = threading.Lock()
    active = 0

    def record(row):
        row['at'] = datetime.now(UTC).isoformat()
        with lock, (output / 'actual-provider-http.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def send(client, request, *args, **kwargs):
        nonlocal active
        if request.method != 'POST':
            return original(client, request, *args, **kwargs)
        body = json.loads(request.content)
        host = request.url.host
        if host == '10.130.138.46':
            if body.get('model') != 'dashscope/qwen3.7-plus' or body.get('enable_thinking') is not False:
                raise ValueError('Gateway model or thinking args differ from frozen runtime')
            key = gateway_key
        elif host == 'api.deepseek.com' and evolver_key:
            if body.get('model') != 'deepseek-flash' or body.get('tools'):
                raise ValueError('Official Evolver credential scope mismatch')
            key = evolver_key
        else:
            raise ValueError(f'Unexpected provider POST host: {host}')
        request.headers['Authorization'] = 'Bearer ' + key
        with lock:
            active += 1
            concurrent = active
        record({'event': 'request', 'host': host, 'in_flight': concurrent,
                'args': {k: body[k] for k in ('model', 'temperature', 'thinking', 'enable_thinking',
                                             'reasoning_effort', 'max_tokens', 'tool_choice') if k in body},
                'tools_count': len(body.get('tools') or [])})
        start = perf_counter()
        try:
            response = original(client, request, *args, **kwargs)
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            record({'event': 'response', 'host': host, 'http_status': response.status_code,
                    'metadata': response_metadata(payload), 'elapsed_seconds': perf_counter() - start})
            return response
        except BaseException as error:
            record({'event': 'failure', 'host': host, 'error_type': type(error).__name__,
                    'error': safe_error(error), 'elapsed_seconds': perf_counter() - start})
            raise
        finally:
            with lock:
                active -= 1

    httpx.Client.send = send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tau2-data-dir', required=True)
    parser.add_argument('--mode', choices=('aa', 'formal'), required=True)
    parser.add_argument('--approved-request-cap', type=int)
    parser.add_argument('--approve-unbounded-requests', action='store_true')
    parser.add_argument('--stop-before-next-episode-file')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    raw = load_config(args.config)
    manifest = AlternatingManifest.from_mapping(raw)
    if args.mode == "formal":
        validate_v2_promotion_readiness(json.loads(manifest.skill_evolution_v2_json),
                                       run_validation=manifest.run_validation)
    for name, expected in raw['launch_readiness']['execution_source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen launcher differs: {name}')
    policy = json.loads(manifest.skill_evolution_v2_json)
    if manifest.request_budget_cap is None:
        if not policy['evaluation'].get('allow_unbounded_requests', False) or (args.execute and not args.approve_unbounded_requests):
            raise ValueError('Unbounded requests require frozen user authorization and explicit launch flag')
        if args.mode != 'formal':
            raise ValueError('A/A diagnostics still require a finite explicit budget')
    elif args.execute and args.approved_request_cap != manifest.request_budget_cap:
        raise ValueError('Explicit approved cap must match the frozen config')
    if args.mode == 'formal':
        policy = json.loads(manifest.skill_evolution_v2_json)
        if (not policy['evaluation']['calibration_confirmed']
                and not policy['evaluation'].get('allow_uncalibrated_launch', False)):
            raise ValueError('Formal launch blocked: Airline A/A calibration is not confirmed')
    load_alternating_tasks(manifest, args.tau2_data_dir,
                           include_validation=args.mode == 'formal', include_heldout=False)
    print(json.dumps({'id': manifest.experiment_id, 'mode': args.mode,
                      'parallel_episodes': manifest.max_parallel_episodes,
                      'request_cap': manifest.request_budget_cap,
                      'real_requests_started': False}), flush=True)
    if not args.execute:
        return
    os.environ['TAU2_DATA_DIR'] = args.tau2_data_dir
    for name in ('NO_PROXY', 'no_proxy'):
        os.environ[name] = os.environ.get(name, '') + ',10.130.138.46'
    gateway_key = credential('litellm', 'litellm-api-key')
    os.environ['OPENAI_API_KEY'] = gateway_key
    evolver_key = credential('api.deepseek.com/v1', 'evotau-evolver-api-key') if args.mode == 'formal' else None
    output = Path(manifest.output_path)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'formal' and not (output / 'manifest.json').exists():
        operator_files = {'operator-launch.json', 'launch.log', 'operator-failure.json'}
        if any(path.name not in operator_files for path in output.iterdir()):
            raise ValueError('Cannot bind nonempty output with unrecognized artifacts')
        write_manifest_once(output / 'manifest.json', manifest)
    install_transport(output, gateway_key, evolver_key)
    try:
        if args.mode == 'aa':
            with frozen_run_lock(output / 'diagnostic-checkpoint.json'):
                result = run_independent_repetitions(
                    raw, data_dir=args.tau2_data_dir, output=output,
                    tasks=list(manifest.evolution_task_ids), seeds=[1, 2, 3, 4],
                    repetitions=2, approved_cap=args.approved_request_cap,
                    stop_before_next_episode_file=args.stop_before_next_episode_file,
                )
            print(json.dumps({'stage': 'complete', 'episode_count': result['episode_count']}), flush=True)
        else:
            result_path, _ = run_from_config(args.config, tau2_data_dir=args.tau2_data_dir,
                                            stop_before_next_episode_file=args.stop_before_next_episode_file)
            print(json.dumps({'stage': 'complete', 'result_path': str(result_path)}), flush=True)
    except Exception as error:
        (output / 'operator-failure.json').write_text(json.dumps(
            {'error_type': type(error).__name__, 'message': safe_error(error),
             'at': datetime.now(UTC).isoformat()}, indent=2))
        raise


if __name__ == '__main__':
    main()
