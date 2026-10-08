import json
from copy import deepcopy

import pytest

from evotau.evolution_candidates import DIAGNOSER_PROMPT, V2Providers
from evotau.service_skills import V2_MUTATION_TYPES


def diagnosis(surface="skill", mutations=None):
    return {"clusters": [{
        "cluster_id": "scope",
        "root_cause": "The confirmed item scope was not retained before execution.",
        "evidence_task_ids": ["8"],
        "protected_success_task_ids": ["66"],
        "recommended_surface": surface,
        "recommended_mutation_types": ["add"] if mutations is None else mutations,
        "risk": "Unnecessary confirmation on already clear requests.",
    }]}


def validate(monkeypatch, response):
    provider = V2Providers(None)
    monkeypatch.setattr(provider, "call", lambda *args: response)
    return provider.diagnose({"task_interactions": [
        {"task": {"task_id": task}} for task in ("8", "66")
    ]})


@pytest.mark.parametrize("operation", V2_MUTATION_TYPES)
def test_diagnoser_accepts_each_declared_mutation(monkeypatch, operation):
    response = diagnosis(mutations=[operation])
    assert validate(monkeypatch, response) == response


@pytest.mark.parametrize("operation", [
    "action_scope_validation", "replacement_mapping_contract_clarification",
    "failure_category", "ADD",
])
def test_diagnoser_rejects_invented_labels_without_repair(monkeypatch, operation):
    response = diagnosis(mutations=[operation])
    original = deepcopy(response)
    with pytest.raises(ValueError, match="unknown recommended mutation"):
        validate(monkeypatch, response)
    assert response == original


@pytest.mark.parametrize("surface", ["tool_boundary", "runtime_protocol", "stochastic_or_weak"])
def test_non_skill_diagnosis_can_recommend_no_mutation(monkeypatch, surface):
    response = diagnosis(surface=surface, mutations=[])
    assert validate(monkeypatch, response) == response


@pytest.mark.parametrize("value", ["", "add", {}, None, [1], [[]]])
def test_mutation_types_must_be_a_string_array(monkeypatch, value):
    with pytest.raises(ValueError, match="JSON string array"):
        validate(monkeypatch, diagnosis(mutations=value) if value is not None else {
            "clusters": [{**diagnosis()["clusters"][0], "recommended_mutation_types": None}]
        })


def test_diagnoser_prompt_exposes_the_enum_contract():
    for operation in V2_MUTATION_TYPES:
        assert operation in DIAGNOSER_PROMPT
    for surface in ("skill", "tool_boundary", "runtime_protocol", "stochastic_or_weak"):
        assert surface in DIAGNOSER_PROMPT
    assert "natural language" in DIAGNOSER_PROMPT
    assert "invented labels" in DIAGNOSER_PROMPT
    assert "empty recommended_mutation_types array is valid" in DIAGNOSER_PROMPT


@pytest.mark.parametrize("method", ["mutate", "crossover"])
def test_provider_mutation_schema_error_is_fatal_before_stage_publication(monkeypatch, method):
    provider = V2Providers(None)
    response = {"operation": "invented"}
    monkeypatch.setattr(provider, "call", lambda *args: response)
    with pytest.raises(ValueError):
        getattr(provider, method)({})
    assert response == {"operation": "invented"}


def test_raw_gpt_enum_failure_remains_rejected_after_prompt_fix(monkeypatch):
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / (
        "experiments/results/v2-readiness-gpt61sol-20261008/"
        "v2-readiness-gpt61sol-representative-20261008/evolver-calls"
    )
    response = next(json.loads(path.read_text())["response"] for path in directory.glob("*/output.json")
                    if "clusters" in json.loads(path.read_text()).get("response", {}))
    provider = V2Providers(None)
    monkeypatch.setattr(provider, "call", lambda *args: response)
    tasks = {task for cluster in response["clusters"] for task in (
        cluster["evidence_task_ids"] + cluster["protected_success_task_ids"]
    )}
    with pytest.raises(ValueError, match="unknown recommended mutation"):
        provider.diagnose({"task_interactions": [{"task": {"task_id": task}} for task in tasks]})
