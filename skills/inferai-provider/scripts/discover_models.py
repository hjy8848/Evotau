#!/usr/bin/env python3
"""Read credential-scoped model IDs; never emit credentials."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

BASE = 'https://inferaiapi.com/v1'
GROUPS = {
    'deepseek': ('OPENAI_API_KEY', 'openai-api-key'),
    'gpt': ('INFERAI_API_KEY', 'openai-gpt-api-key'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all', *GROUPS], default='all')
    args = parser.parse_args()
    result = {'checked_at': datetime.datetime.now(datetime.UTC).isoformat(), 'groups': {}}
    for name, (environment, account) in GROUPS.items():
        if args.group not in ('all', name):
            continue
        key = os.environ.get(environment, '').strip()
        if not key and sys.platform == 'darwin':
            completed = subprocess.run(
                ['security', 'find-generic-password', '-s', 'inferaiapi.com/v1', '-a', account, '-w'],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if completed.returncode == 0:
                key = completed.stdout.strip()
        if not key:
            result['groups'][name] = {'error': 'credential unavailable', 'environment': environment}
            continue
        request = urllib.request.Request(BASE + '/models', headers={'Authorization': 'Bearer ' + key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
                result['groups'][name] = {'status': response.status, 'model_ids': sorted(x['id'] for x in payload['data'])}
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            result['groups'][name] = {'error_type': type(error).__name__, 'http_status': getattr(error, 'code', None)}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
