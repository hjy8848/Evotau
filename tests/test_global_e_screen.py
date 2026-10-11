"""Global E eligibility is opt-in and cannot replace V or Discovery gates."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from test_skill_evolution_v2 import FakeProviders, FakeRunner, record, run

from evotau.alternating_manifest import AlternatingManifest
from evotau.evolution_gate import cheap_screen, global_e_screen
from evotau.skill_evolution_config import DEFAULT_V2, freeze_v2_policy
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).parents[1]


def historical_cells():
    return (
        [
            record("12", False),
            record("34", False),
            record("49", True),
            record("15", True),
        ],
        [
            record("12", False),
            record("34", True),
            record("49", True),
            record("15", True),
        ],
    )


def test_off_target_improvement_enters_V_only_in_global_protocol():
    old, new = historical_cells()
    assert not cheap_screen(old, new, {"12"}, {"49", "15"})["passed"]
    result = global_e_screen(old, new, {"12"}, {"49", "15"})
    assert result["passed"] and result["net_gain"] == 1
    assert result["fixed_cells"] == 0 and result["overall_fixed_cells"] == 1
    assert result["overall_fixed_cell_refs"] == [{"task_id": "34", "seed": 1}]
    assert result["inference_scope"] == "observed_training_screen_only"


@pytest.mark.parametrize(
    "kind", ["tie", "worse", "unknown", "hard", "stuck", "regression"]
)
def test_global_screen_risks_fail_closed(kind):
    old, new = historical_cells()
    if kind == "tie":
        new = old
    if kind == "worse":
        new = [old[0], old[1], record("49", False), old[3]]
    if kind == "unknown":
        new[0] = record("12", None)
    if kind == "hard":
        new[1] = record("34", True, hard_policy_protocol_violations=1)
    if kind == "stuck":
        new[0] = record("12", False, termination_reason="max_steps")
    if kind == "regression":
        # Positive net gain still cannot waive the configured risk threshold.
        new = [record("12", True), record("34", True), record("49", False), old[3]]
    assert not global_e_screen(old, new, {"12"}, {"49", "15"})["passed"]


def test_limited_regression_only_with_positive_net_gain_and_explicit_allowance():
    old, new = historical_cells()
    new = [record("12", True), record("34", True), record("49", False), old[3]]
    result = global_e_screen(old, new, {"12"}, {"49", "15"}, regression_allowance=1)
    assert (
        result["passed"]
        and result["net_gain"] == 1
        and result["protected_regressions"] == 1
    )
    new[1] = old[1]
    assert not global_e_screen(old, new, regression_allowance=1)["passed"]


def test_pairing_duplicates_and_cell_protection():
    old = [record("22", False, 1), record("22", True, 2), record("34", False, 1)]
    new = [record("22", True, 1), record("22", False, 2), record("34", True, 1)]
    result = global_e_screen(old, new, {"22"})
    assert not result["passed"] and result["net_gain"] == 1
    assert result["broken_cells"] == [{"task_id": "22", "seed": 2}]
    for a, b in [(old + [old[0]], new + [new[0]]), (old, new[:-1])]:
        with pytest.raises(ValueError, match="paired"):
            global_e_screen(a, b)
    assert not global_e_screen([], [])["passed"]


def policy():
    p = deepcopy(DEFAULT_V2)
    p["algorithm_version"] = "direct_skill_v_validation_v2"
    p["evaluation"].update(
        promotion_protocol="v_primary", screen_protocol="global_e_v1"
    )
    p["service_evolution"].update(crossover=False, candidates_per_generation=1)
    p["statistical_gate"].update(method="task_block_bootstrap", bootstrap_samples=100)
    return p


def test_policy_default_serialization_and_strict_opt_in():
    old = deepcopy(DEFAULT_V2)
    frozen = freeze_v2_policy(
        old, {"agent": "offline/test"}, {"agent": {"temperature": 0.0}}
    )
    assert "screen_protocol" not in frozen["evaluation"]
    assert (
        freeze_v2_policy(
            policy(), {"agent": "offline/test"}, {"agent": {"temperature": 0.0}}
        )["evaluation"]["screen_protocol"]
        == "global_e_v1"
    )
    for value in ["invented", None, 1]:
        p = policy()
        p["evaluation"]["screen_protocol"] = value
        with pytest.raises(ValueError, match="screen protocol"):
            freeze_v2_policy(
                p, {"agent": "offline/test"}, {"agent": {"temperature": 0.0}}
            )
    old["evaluation"]["screen_protocol"] = "global_e_v1"
    with pytest.raises(ValueError, match="V-primary"):
        freeze_v2_policy(
            old, {"agent": "offline/test"}, {"agent": {"temperature": 0.0}}
        )


class OffTargetRunner(FakeRunner):
    def __call__(self, **kw):
        r = super().__call__(**kw)  # keep native mock trajectory support
        success = kw["task_id"] not in ("1", "3")
        if kw["service"].skills and kw["task_id"] == "3":
            success = True
        # V regresses: superiority must reject, regardless of E improvement.
        if kw["task_id"] == "4":
            success = not bool(kw["service"].skills)
        return replace(r, task_success=success, native_reward=float(success))


def test_e_pass_V_reject_no_deployment_G2_and_resume(tmp_path):
    p = policy()
    providers = FakeProviders()
    runner = OffTargetRunner()
    result, _, _ = run(
        tmp_path,
        policy=p,
        provider=providers,
        runner=runner,
        validation=True,
        generations=2,
        manifest_sha=sha256_json(p),
    )
    for generation in result.generations[:1]:
        row = generation["service_phase"]["candidates"][0]
        assert row["screen"]["passed"] and row["screen"]["fixed_cells"] == 0
        assert row["gate"]["verdict"] == "REJECTED"
        assert row["gate"]["repair_superiority"] is None
        assert row["decision"] == "REJECTED"
        assert all(g["panel"] == "V" for g in row["gate"]["opponents"])
    assert not result.service.skills
    calls = list(providers.calls), list(runner.calls)
    resumed, _, _ = run(
        tmp_path,
        policy=p,
        provider=providers,
        runner=runner,
        validation=True,
        generations=2,
        manifest_sha=sha256_json(p),
    )
    assert resumed.generations == result.generations
    assert calls == (providers.calls, runner.calls)
    legacy = deepcopy(p)
    legacy["evaluation"].pop("screen_protocol")
    with pytest.raises(ValueError):
        run(tmp_path, policy=legacy, validation=True, manifest_sha=sha256_json(legacy))


def test_protocol_changes_manifest_and_independent_config():
    raw = yaml.safe_load(
        (ROOT / "configs/airline-global-e-v1-hybrid-seeds-p2.yaml").read_text()
    )
    a = AlternatingManifest.from_mapping(raw)
    braw = deepcopy(raw)
    braw["experiment"]["skill_evolution_v2"]["evaluation"].pop("screen_protocol")
    b = AlternatingManifest.from_mapping(braw)
    assert a.sha256 != b.sha256
    assert a.skill_evolution_v2_json != b.skill_evolution_v2_json
    assert "global-e-v1" in raw["experiment"]["output_path"]


def test_global_protocol_can_promote_only_after_V_accepts(tmp_path):
    class ImprovedV(OffTargetRunner):
        def __call__(self, **kw):
            r = super().__call__(**kw)
            if kw["task_id"] == "4":
                return replace(
                    r,
                    task_success=bool(kw["service"].skills),
                    native_reward=float(bool(kw["service"].skills)),
                )
            return r

    p = policy()
    p["statistical_gate"].update(
        min_tasks=1, max_harmfulness=1, max_stuck_delta=1
    )  # fixture only
    result, _, _ = run(tmp_path, policy=p, runner=ImprovedV(), validation=True)
    row = result.generations[0]["service_phase"]["candidates"][0]
    assert row["screen"]["fixed_cells"] == 0 and row["screen"]["passed"]
    assert row["gate"]["verdict"] == "ACCEPTED" and result.service.skills


def test_frozen_discovery_always_uses_target_protocol():
    from evotau.skill_evolution import _screen_protocol

    assert _screen_protocol({"screen_protocol": "global_e_v1"}, []) == "global_e_v1"
    assert (
        _screen_protocol(
            {"screen_protocol": "global_e_v1"}, [{"discovery_id": "frozen"}]
        )
        == "target_fix_v1"
    )
    assert _screen_protocol({}, []) == "target_fix_v1"
