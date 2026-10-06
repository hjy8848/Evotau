"""Supervise a single already-running experiment; no dispatch or retries."""
from pathlib import Path
from datetime import UTC,datetime
import json,os,subprocess,time
ROOT=Path('/Users/spring/RSI/Evotau-alternating')
RUN=ROOT/'experiments/runs/evotau-retail-skillmemory-v1-deepseek-v4-pro-e20-g5-p1-continuation-20261007'
STATUS=Path('/tmp/evotau-retail-g5-supervisor-status.json')
PID_FILE=Path('/tmp/evotau-retail-g5-process.pid')
Path('/tmp/evotau-retail-g5-supervisor.pid').write_text(str(os.getpid()))
def save(**data):
 value={'at':datetime.now(UTC).isoformat(),'supervisor_pid':os.getpid(),**data}
 temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(STATUS)
while True:
 try:s=json.loads((RUN/'run-execution-state.json').read_text())
 except (FileNotFoundError,json.JSONDecodeError):s={}
 state=s.get('status')
 if state in ('complete','failed','paused'):
  save(status='finalizing',experiment_status=state)
  time.sleep(2)
  with Path('/tmp/evotau-retail-g5-finalize.log').open('w') as log:
   result=subprocess.run(['/Users/spring/RSI/Evotau/.venv/bin/python','/tmp/evotau-retail-g5-finalize.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  save(status='archived_and_pushed' if result.returncode==0 else 'archive_needs_attention',experiment_status=state,finalizer_exit_code=result.returncode,report='/tmp/evotau-retail-g5-finalize.log')
  break
 try:os.kill(int(PID_FILE.read_text()),0)
 except (OSError,ValueError,FileNotFoundError):
  save(status='process_missing_needs_attention',experiment_status=state)
  break
 save(status='monitoring',experiment_status=state)
 time.sleep(15)
