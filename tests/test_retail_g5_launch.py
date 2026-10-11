"""Offline freeze checks and five-generation/no-op resume control."""
import importlib.util
import json
from pathlib import Path

from test_customer_skill_protocol import SkillProviders
from test_failure_analysis import AnalystProviders
from test_skill_evolution_v2 import run

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('retail_g5', ROOT / 'experiments/execution/run-retail-gateway.py')
LAUNCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCH)
CONFIG = ROOT / 'configs/retail-customer-v2-analyst-qwen37plus-e20-v20-h20-g5-p4.yaml'


def test_frozen_retail_scope_models_and_seeds():
    _, raw, manifest = LAUNCH.prepare(CONFIG)
    assert manifest.generations == 5 and manifest.max_parallel_episodes == 4
    e,v,h = map(set,(manifest.evolution_task_ids,manifest.validation_task_ids,manifest.heldout_task_ids))
    assert len(e)==len(v)==len(h)==20
    assert not(e&v or e&h or v&h)
    assert not (e|v|h)&{'46','47'}
    policy=json.loads(manifest.skill_evolution_v2_json)
    assert policy['customer_evolution']['evolver_prompt_version']=='v2'
    assert policy['evaluation']['gate_seeds']==[1,2]
    assert policy['evaluation']['heldout_seeds']==[1,2]
    assert not policy['evaluation']['calibration_confirmed']
    assert policy['evaluation']['allow_uncalibrated_launch']
    assert all(args['enable_thinking'] is False for args in raw['experiment']['model_args'].values())


class FiveRoundMock(SkillProviders):
    analyze_service_failures = AnalystProviders.analyze_service_failures
    deduplicate_failure_hypotheses = AnalystProviders.deduplicate_failure_hypotheses
    count=3
    same=False
    invalid=False


def test_five_rounds_legal_noop_and_resume(tmp_path):
    _,_,manifest = LAUNCH.prepare(CONFIG)
    policy=json.loads(manifest.skill_evolution_v2_json)
    providers=FiveRoundMock()
    result,_,runner=run(tmp_path,provider=providers,policy=policy,validation=True,generations=5)
    assert len(result.generations)==5
    assert not result.service.skills
    calls=list(providers.calls),list(runner.calls)
    resumed,_,_=run(tmp_path,provider=providers,runner=runner,policy=policy,validation=True,generations=5)
    assert resumed.generations==result.generations
    assert calls==(providers.calls,runner.calls)
