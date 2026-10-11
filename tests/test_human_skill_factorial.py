"""Offline checks for fixed human stimuli and independent diagnostic controls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evotau.records import customer_strategy_id, service_strategy_id

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('human_trial', ROOT / 'experiments/execution/run-human-skill-factorial.py')
TRIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIAL)
CONFIG = ROOT / 'configs/airline-human-scope-ledger-v1-e3-factorial-p2.yaml'


def test_four_unique_conditions_strict_stimuli():
    _, raw, manifest, panels = TRIAL.prepare(CONFIG)
    assert len({(customer_strategy_id(c), service_strategy_id(s)) for _, c, s in panels}) == 4
    assert tuple(manifest.evolution_task_ids) == ('7', '12', '15')
    assert not manifest.run_validation and not manifest.run_heldout
    assert manifest.request_budget_cap is None
    assert raw['manual_skill_trial']['automatic_promotion'] is False
    assert len(panels[0][2].skills) == 0
    assert len(panels[2][2].skills) == 1
    compiled = panels[1][1].text
    assert 'message_sha256' not in compiled and 'evidence_refs' not in compiled
    assert 'MSJ4OA' not in panels[2][2].skills[0].guidance


def test_changed_stimulus_rejected(monkeypatch):
    original = TRIAL.load_config
    def changed(path):
        raw = original(path)
        key = next(iter(raw['manual_skill_trial']['inputs']))
        raw['manual_skill_trial']['inputs'][key] = '0' * 64
        return raw
    monkeypatch.setattr(TRIAL, 'load_config', changed)
    with pytest.raises(ValueError, match='Frozen manual stimulus'):
        TRIAL.prepare(CONFIG)


def test_panel_scope_rejected(monkeypatch):
    original = TRIAL.load_config
    def changed(path):
        raw = original(path)
        raw['manual_skill_trial']['episodes'] = 13
        return raw
    monkeypatch.setattr(TRIAL, 'load_config', changed)
    with pytest.raises(ValueError, match='scope'):
        TRIAL.prepare(CONFIG)


def test_incomplete_attempt_never_blindly_reissued(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'run-execution-state.json').write_text(json.dumps({'status': 'failed'}))
    manifest = SimpleNamespace(output_path='run', checkpoint_path='checkpoint.json')
    with pytest.raises(RuntimeError, match='review pending requests'):
        TRIAL.execute(None, tmp_path, {}, manifest, (), {})
