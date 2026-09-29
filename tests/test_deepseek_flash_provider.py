from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from evotau.manifest import MechanismManifest
from evotau.provider_plugins import deepseek_v4_flash
from evotau.records import EvidenceRef
from evotau.strategies import CustomerStrategy, ServiceStrategy

ROOT = Path(__file__).resolve().parents[1]


def _flash_manifest() -> MechanismManifest:
    config = yaml.safe_load(
        (
            ROOT / "configs/phase3-inferai-deepseek-v4-flash-evolution-smoke.yaml"
        ).read_text(encoding="utf-8")
    )
    return MechanismManifest.from_mapping(config)


def test_flash_auditor_returns_taxonomy_bound_trajectory_references(
    monkeypatch, tmp_path
) -> None:
    data_root = tmp_path / "tau-data"
    (data_root / "tau2/domains/retail").mkdir(parents=True)
    (data_root / "tau2/user_simulator").mkdir(parents=True)
    (data_root / "tau2/domains/retail/policy.md").write_text(
        "Fixed Retail policy",
        encoding="utf-8",
    )
    (data_root / "tau2/user_simulator/simulation_guidelines.md").write_text(
        "Fixed user guidelines",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAU2_DATA_DIR", str(data_root))
    calls = []

    def fake_generate(model, model_args, *, call_name, system, payload):
        calls.append((model, model_args, call_name, system, payload))
        if len(calls) == 1:
            return {
                "customer_valid": True,
                "strategy_applicable": True,
                "customer_strategy_adherent": True,
                "policy_violation": True,
                "invalid_repeated_write_calls": 0,
                "policy_rule_id": "retail.policy:explicit_confirmation",
                "mistake_type": "missing_explicit_confirmation",
                "workflow_stage": "pre_write",
                "evidence": [
                    {
                        "turn_index": 2,
                        "source": "assistant",
                        "summary": "write request submitted without the policy-required confirmation",
                    }
                ],
                "summary": "additional model explanation is ignored",
            }
        return {
            "customer_valid": True,
            "strategy_applicable": True,
            "customer_strategy_adherent": True,
            "policy_violation": False,
            "invalid_repeated_write_calls": 0,
            "policy_rule_id": "retail.policy:explicit_confirmation",
            "mistake_type": "missing_explicit_confirmation",
            "workflow_stage": "pre_write",
            "evidence": "non-violation evidence is dropped",
        }

    monkeypatch.setattr(deepseek_v4_flash, "_generate_json", fake_generate)
    callbacks = deepseek_v4_flash.build_callbacks(config={}, manifest=_flash_manifest())
    simulation = SimpleNamespace(
        messages=[
            {"role": "user", "content": "Please return these items."},
            {"role": "assistant", "content": "I can do that."},
            {"role": "assistant", "content": "Calling return tool."},
        ],
        model_dump=lambda *, mode: {
            "id": "sim-1",
            "task_id": "73",
            "seed": 42,
            "messages": [
                {"role": "user", "content": "Please return these items."},
                {"role": "assistant", "content": "I can do that."},
                {"role": "assistant", "content": "Calling return tool."},
            ],
            "reward_info": {"reward": 0.0},
        },
    )
    task = SimpleNamespace(
        id="73",
        user_scenario={"instructions": {"known_info": "fixed scenario fact"}},
        evaluation_criteria={
            "actions": [{"arguments": {"secret": "must not be sent"}}]
        },
    )
    audit = callbacks["audit_provider"](
        simulation,
        task,
        CustomerStrategy(),
        ServiceStrategy(),
        "discovery",
    )

    assert audit.policy_violation is True
    assert audit.evidence == (
        EvidenceRef(
            2,
            "assistant",
            "write request submitted without the policy-required confirmation",
        ),
    )
    model, args, call_name, _system, payload = calls[0]
    assert model == "openai/deepseek-v4-flash"
    assert args["extra_body"] == {"thinking": {"type": "disabled"}}
    assert args["api_base"] == "https://inferaiapi.com/v1"
    assert call_name == "evotau_independent_episode_audit"
    assert payload["scenario"] == {
        "instructions": {"known_info": "fixed scenario fact"}
    }
    assert "evaluation_criteria" not in payload
    assert "reward_info" not in payload["trajectory"]
    assert payload["official_retail_policy"] == "Fixed Retail policy"
    non_violation = callbacks["audit_provider"](
        simulation,
        task,
        CustomerStrategy(),
        ServiceStrategy(),
        "discovery",
    )
    assert non_violation.policy_violation is False
    assert non_violation.policy_rule_id is None
    assert non_violation.evidence == ()
    assert set(callbacks) == {
        "audit_provider",
        "service_proposal_provider",
        "service_repair_audit_provider",
    }
