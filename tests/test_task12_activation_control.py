"""Scope, unchanged stimuli and native-runtime controls, without API calls."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('task12_control', ROOT / 'experiments/execution/run-task12-activation-control.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_selector_differs():
    _, b, bm, bs = MODULE.prepare(ROOT / 'configs/task12-activation-control-b-seed1.yaml')
    _, c, cm, cs = MODULE.prepare(ROOT / 'configs/task12-activation-control-c-seed1.yaml')
    assert bs.to_dict() == cs.to_dict()
    assert b['experiment']['models'] == c['experiment']['models']
    assert b['experiment']['model_args'] == c['experiment']['model_args']
    assert bm.seed == cm.seed == 1
    assert bm.max_steps == cm.max_steps == 200
    assert bm.request_budget_cap == cm.request_budget_cap == 60
    assert not bm.run_validation and not cm.run_heldout
    assert bm.output_path != cm.output_path


def test_unbounded_rejected(monkeypatch):
    original = MODULE.load_config
    def changed(path):
        config = original(path)
        config['experiment']['request_budget_cap'] = None
        return config
    monkeypatch.setattr(MODULE, 'load_config', changed)
    with pytest.raises(ValueError, match='scope'):
        MODULE.prepare(ROOT / 'configs/task12-activation-control-c-seed1.yaml')


def test_changed_skill_rejected(monkeypatch):
    original = MODULE.load_config
    def changed(path):
        config = original(path)
        skill = config['activation_control']['skill_path']
        config['activation_control']['frozen_inputs'][skill] = '0' * 64
        return config
    monkeypatch.setattr(MODULE, 'load_config', changed)
    with pytest.raises(ValueError, match='Frozen'):
        MODULE.prepare(ROOT / 'configs/task12-activation-control-c-seed1.yaml')
