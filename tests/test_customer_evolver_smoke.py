from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evotau.customer_evolver_smoke import (
    CustomerEvolverSmokeConfig,
    CustomerEvolverSmokeFailed,
    run_from_config,
)
from evotau.strategies import CustomerStrategy


def _config(tmp_path: Path) -> Path:
    document = {
        "proposal_smoke": {
            "schema_version": 1,
            "experiment_id": "proposal-smoke-test",
            "real_provider_enabled": True,
            "model": "openai/deepseek-v4-flash",
            "model_args": {
                "temperature": 0.0,
                "api_base": "https://inferaiapi.com/v1",
                "thinking_mode": "disabled",
            },
            "generation": 0,
            "candidate_count": 2,
            "proposal_seed": 1,
            "incumbent_strategy": CustomerStrategy.v2_baseline().to_dict(),
            "output_path": "results/proposal-smoke-test",
        },
    }
    path = tmp_path / "proposal.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _response(incumbent: CustomerStrategy) -> dict:
    strategies = (
        replace(incumbent, request_decomposition="one_by_one"),
        replace(incumbent, correction_behavior="correct_once"),
    )
    rows = []
    old = incumbent.to_dict()
    for strategy in strategies:
        new = strategy.to_dict()
        rows.append({
            "strategy": new,
            "changed_fields": [key for key in old if old[key] != new[key]],
            "hypothesis": "Exploration of a different interaction sequence before commitment.",
            "evidence_refs": [],
        })
    return {"candidates": rows}


def test_provider_only_smoke_saves_validated_candidates_without_task_or_episode_data(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    contexts = []

    def provider(context):
        contexts.append(context.to_dict())
        return _response(context.incumbent)

    result = run_from_config(config_path, proposal_provider=provider)

    assert result["status"] == "complete"
    assert len(result["candidate_proposals"]) == 2
    assert result["request_scope"] == {
        "task_ids_provided": [],
        "verified_failure_summaries_provided": 0,
        "tau_bench_episodes_run": 0,
    }
    assert contexts[0]["verified_failure_summaries"] == []
    assert "task_id" not in json.dumps(contexts[0])
    artifact = Path(result["artifact_path"])
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["candidate_proposals"] == result["candidate_proposals"]
    assert saved["provider_budget"]["attempts"] == 0
    assert saved["response_candidate_count"] == 2
    assert saved["response_sha256"]


def test_provider_only_smoke_saves_failure_without_provider_error_text(tmp_path: Path) -> None:
    config_path = _config(tmp_path)

    def provider(_context):
        raise RuntimeError("secret-shaped provider diagnostic")

    with pytest.raises(CustomerEvolverSmokeFailed, match="RuntimeError"):
        run_from_config(config_path, proposal_provider=provider)
    artifact_path = tmp_path / "results/proposal-smoke-test/proposal-smoke.json"
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["error_type"] == "RuntimeError"
    assert "secret-shaped" not in artifact_path.read_text(encoding="utf-8")


def test_provider_only_smoke_records_safe_candidate_validation_diagnostics(tmp_path: Path) -> None:
    config_path = _config(tmp_path)

    def provider(context):
        response = _response(context.incumbent)
        response["candidates"][0]["changed_fields"] = ["disclosure"]
        return response

    with pytest.raises(CustomerEvolverSmokeFailed, match="ValueError"):
        run_from_config(config_path, proposal_provider=provider)
    artifact_path = tmp_path / "results/proposal-smoke-test/proposal-smoke.json"
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved["failure_stage"] == "candidate_validation"
    assert "changed_fields must exactly match" in saved["validation_error"]
    assert saved["response_candidate_count"] == 2
    assert len(saved["response_candidate_shapes"]) == 2


def test_provider_only_smoke_requires_an_explicit_live_provider_config(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["proposal_smoke"]["real_provider_enabled"] = False
    with pytest.raises(ValueError, match="real_provider_enabled: true"):
        CustomerEvolverSmokeConfig.from_mapping(document)
