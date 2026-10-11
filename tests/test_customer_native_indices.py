"""Exact raw-message citation contracts; no model responses are repaired."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from evotau.customer_skills import validate_customer_policy
from evotau.customer_trajectory_validity import (
    indexed_native_trajectory,
    trajectory_context,
    validate_trajectory_report,
)
from evotau.evolution_candidates import V2Providers
from evotau.tau_provenance import sha256_json


def native():
    return {"messages": [
        {"role": "assistant", "content": "How can I help?"},
        {"role": "multi_tool", "tool_messages": [
            {"role": "tool", "content": "first"}, {"role": "tool", "content": "second"}]},
        {"role": "user", "content": "My budget is $650."},
        {"role": "assistant", "content": "Understood."},
    ], "termination_reason": "user_stop"}


def context():
    # The projected view would move the original user from index 2 to index 3.
    rows = [{"task": {"task_id": "12", "user_scenario": "Budget $650"}, "seed": 1,
             "trajectory_ref": "native-simulation.json", "trajectory": {"messages": []}}]
    return trajectory_context(rows, {}, "native policy", validator_version="v2",
                              original_trajectories={"native-simulation.json": native()})


def report(indices):
    return {"cells": [{"task_id": "12", "seed": 1, "status": "valid",
                       "reason": "Original user stated budget.", "evidence_message_indices": indices}]}


def test_raw_indices_and_hashes_survive_multi_tool_without_flattening():
    ctx = context()
    assert ctx["cells"][0]["allowed_customer_message_indices"] == [2]
    messages = ctx["cells"][0]["trajectory"]["messages"]
    assert len(messages) == len(native()["messages"])
    for i, message in enumerate(messages):
        assert message["original_message_index"] == i
        assert message["original_message_sha256"] == sha256_json(native()["messages"][i])
    assert validate_trajectory_report(report([2]), ctx) == report([2])


@pytest.mark.parametrize("indices", [[3], [1], [5], [-1], [2, 2], [True], []])
def test_bad_citations_are_rejected_without_remapping(indices):
    value = report(indices)
    before = deepcopy(value)
    with pytest.raises(ValueError):
        validate_trajectory_report(value, context())
    assert value == before


@pytest.mark.parametrize("tamper", ["index", "allowlist", "hash"])
def test_broken_input_contract_fails_closed(tamper):
    ctx = context()
    if tamper == "index":
        ctx["cells"][0]["trajectory"]["messages"][2]["original_message_index"] = 3
    elif tamper == "allowlist":
        ctx["cells"][0]["allowed_customer_message_indices"] = [3]
    else:
        ctx["cells"][0]["trajectory"]["messages"][2]["original_message_sha256"] = "missing"
    with pytest.raises(ValueError, match="index contract"):
        validate_trajectory_report(report([2]), ctx)


def test_versioned_validator_call_and_historical_default():
    assert "trajectory_validator_version" not in validate_customer_policy({})
    p = V2Providers(SimpleNamespace())
    calls = []
    p.call = lambda prompt, ctx, name: (calls.append((prompt, name)) or report([2]))
    p.validate_customer_trajectory(context())
    assert calls[0][1] == "evotau_customer_trajectory_validator_v2"
    with pytest.raises(ValueError):
        validate_customer_policy({"trajectory_validator_version": "v3"})
    with pytest.raises(TypeError):
        indexed_native_trajectory({"messages": [None]})


def test_indexed_review_offline_G2_and_resume_do_not_repeat_stages(tmp_path):
    from test_customer_skill_protocol import SkillProviders, policy
    from test_skill_evolution_v2 import FakeRunner, run

    p = policy()
    p["customer_evolution"].update(evolver_prompt_version="v2", trajectory_validator_version="v2")
    provider, runner = SkillProviders(), FakeRunner()
    first, _, _ = run(tmp_path, provider=provider, runner=runner, generations=2, policy=p)
    calls = list(provider.calls)
    second, _, _ = run(tmp_path, provider=provider, runner=runner, generations=2, policy=p)
    assert second.generations == first.generations
    assert provider.calls == calls
    assert len(first.generations) == 2
