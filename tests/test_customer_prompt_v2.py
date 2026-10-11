"""Offline contracts; mocked validators are not evidence of live semantic quality."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_customer_skill_protocol import SkillProviders, policy, run, skill

from evotau.alternating_manifest import AlternatingManifest
from evotau.customer_evolution import (
    CUSTOMER_SKILL_PROMPT,
    customer_evolver_prompt,
    prepare_candidate,
)
from evotau.customer_skills import CustomerSkill, validate_customer_policy
from evotau.customer_trajectory_validity import assess_trajectories
from evotau.evolution_candidates import V2Providers
from evotau.open_repair_search import PROTOCOL
from evotau.tau_provenance import sha256_json

ROOT = Path(__file__).parents[1]
LEGACY_HASHES = {
    "ordinary": "c2c45ca2663c45ccf9df0c31b09f98f334b36835d3c14e9cf50f82701214f0fb",
    "bandit": "a0307f636e11972d94f6d8746a87eb57fcccac8fff060d34dba3f4ef7fb4d7fe",
    "open": "3cb3a65563af7ab6270f46bcb2e3a126de32996392e567ad5054d39414b652cf",
}


def mode_context(mode):
    return ({"operator_family_hint": "open_exploration"} if mode == "bandit" else
            {"open_repair_search": {"protocol_version": PROTOCOL}} if mode == "open" else {})


@pytest.mark.parametrize("mode", LEGACY_HASHES)
def test_legacy_prompt_bytes_and_call_identity(mode):
    prompt, name = customer_evolver_prompt(validate_customer_policy({}), mode_context(mode))
    assert hashlib.sha256(prompt.encode()).hexdigest() == LEGACY_HASHES[mode]
    assert name.endswith("_v1")
    assert "evolver_prompt_version" not in validate_customer_policy({})
    if mode == "ordinary":
        assert prompt == CUSTOMER_SKILL_PROMPT


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("mode", LEGACY_HASHES)
def test_provider_explicit_dispatch_preserves_skill_schema(version, mode):
    provider = V2Providers(SimpleNamespace())
    cp = validate_customer_policy({"evolver_prompt_version": version})
    provider.configure_customer({"customer_evolution": cp})
    calls = []

    def capture(prompt, context, name):
        calls.append((prompt, context, name))
        return {"candidates": [skill()]}

    provider.call = capture
    context = {"task_interactions": [], **mode_context(mode)}
    result = provider.customers(context, 1)
    expected = customer_evolver_prompt(cp, context)
    assert (calls[0][0], calls[0][2]) == expected
    assert calls[0][2].endswith("_" + version)
    assert calls[0][1]["customer_evolution"] == cp
    assert CustomerSkill.from_mapping(result["candidates"][0]).to_dict() == skill()
    assert customer_evolver_prompt({"evolver_prompt_version": "v1"}, context) != customer_evolver_prompt(
        {"evolver_prompt_version": "v2"}, context)


@pytest.mark.parametrize("version", ["v3", None, 1, "V2"])
def test_unknown_prompt_versions_fail_closed(version):
    with pytest.raises(ValueError):
        validate_customer_policy({"evolver_prompt_version": version})


def test_conflicting_rc_and_open_modes_rejected():
    with pytest.raises(ValueError):
        customer_evolver_prompt({"evolver_prompt_version": "v2"},
                                {**mode_context("open"), **mode_context("bandit")})


def test_manifest_and_checkpoint_cannot_mix_prompt_versions(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/airline-customer-skill-v1-e10-diagnostic.yaml").read_text())
    a = AlternatingManifest.from_mapping(raw)
    raw2 = deepcopy(raw)
    raw2["experiment"]["skill_evolution_v2"]["customer_evolution"]["evolver_prompt_version"] = "v2"
    b = AlternatingManifest.from_mapping(raw2)
    assert a.sha256 != b.sha256 and a.skill_evolution_v2_json != b.skill_evolution_v2_json
    p1 = policy()
    p2 = deepcopy(p1)
    p2["customer_evolution"]["evolver_prompt_version"] = "v2"
    run(tmp_path, provider=SkillProviders(), policy=p1, manifest_sha=sha256_json(p1))
    with pytest.raises(ValueError):
        run(tmp_path, provider=SkillProviders(), policy=p2, manifest_sha=sha256_json(p2))


@pytest.mark.parametrize("case", ["budget", "dependencies", "clarification"])
def test_task_faithful_fixtures_through_schema_and_existing_trajectory_validator(case):
    fixtures = {
        "budget": ("Upgrade if the cost is at most $650; otherwise keep the booking.",
                   ["I can pay up to $650. What is the price?", "That exceeds my budget; keep the original booking."],
                   ["State the scenario's budget truthfully.", "Give informed consent only within that budget."],
                   "Service may commit an upgrade before resolving conditional consent."),
        "dependencies": ("Change my return date and cabin together, only if the combined cost fits my budget.",
                         ["I need both original changes as a package within my existing budget.", "Please preserve both goals; I cannot accept only one."],
                         ["Communicate all existing dependent goals.", "Answer requested clarifications without prescribing an internal tool sequence."],
                         "Service may lose one requested goal when resolving a dependency."),
        "clarification": ("Use my existing reservation and identity to change the date.",
                          ["My reservation is ABC123, as requested.", "The date you just stated is not the date I requested; please clarify."],
                          ["Truthfully answer necessary identity questions.", "Point out a real mismatch with the original request without inventing a new goal."],
                          "Service may propagate an incorrect date after a lawful correction."),
    }
    scenario, messages, procedure, hypothesis = fixtures[case]
    proposal = skill()
    proposal.update(procedure=procedure, hypothesis=hypothesis)
    row, parsed = prepare_candidate(proposal, {"task_interactions": []},
                                    validate_customer_policy({"evolver_prompt_version": "v2"}))
    assert parsed and row["pre_rollout_rejection"] is None
    assert parsed.to_dict()["schema_version"] == 1
    assert "obtain adequate cost explanations" in parsed.compile().text
    context = {"cells": [{"task_id": "12", "seed": 1, "user_scenario": scenario,
                          "trajectory": {"messages": [{"role": "user", "content": m} for m in messages]}}]}

    def validator(ctx):
        assert ctx["cells"][0]["user_scenario"] == scenario
        assert [m["content"] for m in ctx["cells"][0]["trajectory"]["messages"]] == messages
        return {"cells": [{"task_id": "12", "seed": 1, "status": "valid",
                           "reason": "Fixture verdict: original facts and conditions retained.",
                           "evidence_message_indices": [0, 1]}]}

    assert assess_trajectories(context, validator)["status"] == "valid"
    invalid = deepcopy(context)
    invalid["cells"][0]["trajectory"]["messages"][0]["content"] = "Use a different identity and unlimited budget."
    def reject(ctx):
        return {"cells": [{"task_id": "12", "seed": 1, "status": "invalid",
                           "reason": "Fixture verdict: original facts changed.", "evidence_message_indices": [0]}]}
    assert assess_trajectories(invalid, reject)["status"] == "invalid"
    # No local test purports to reproduce an LLM semantic judgement.
    assert json.dumps(parsed.to_dict())
