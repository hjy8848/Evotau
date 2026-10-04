from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from evotau.alternating_run import run_from_config


def test_one_generation_uses_pinned_tau_runtime_with_local_completion_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = Path(
        os.environ.get(
            "EVOTAU_TAU2_DATA_DIR",
            os.environ.get("TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074"),
        )
    )
    if not data_dir.is_dir():
        pytest.skip("pinned tau-bench data is not available locally")
    monkeypatch.setenv("TAU2_DATA_DIR", str(data_dir))
    pytest.importorskip("tau2")
    pytest.importorskip("litellm")

    from litellm import ModelResponse
    from tau2.utils import llm_utils

    root = tmp_path / "project"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    source_config = Path(__file__).resolve().parents[1] / "configs/alternating-evolution.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    experiment["id"] = "offline-native-alternating-smoke"
    experiment["generations"] = 1
    experiment["customer_candidates"] = 1
    experiment["clean_panel_size"] = 1
    experiment["max_steps"] = 2
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        role: f"offline-{role}"
        for role in ("agent", "customer", "reviewer", "evaluator", "evolver")
    }
    experiment["output_path"] = "experiments/runs/offline-native-alternating-smoke"
    experiment["checkpoint_path"] = "experiments/checkpoints/offline-native-alternating-smoke.json"
    config_path = config_dir / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(root)

    calls: list[str] = []
    role_counts = {f"offline-{role}": 0 for role in ("agent", "customer")}

    def completion(*, model: str, messages: list[dict], **kwargs):
        assert kwargs.get("num_retries") == 0
        calls.append(model)
        if model == "offline-evolver":
            system = messages[0]["content"]
            if "evolve the Customer" in system:
                content = json.dumps({
                    "candidates": ["Ask for a clear update, then follow the answer with a related request."]
                })
            else:
                content = json.dumps({
                    "analysis": "Offline wiring fixture; inspect the order before explaining the next step.",
                    "strategy": "Inspect the requested order before explaining the next step.",
                })
        elif model == "offline-evaluator":
            system = messages[0]["content"]
            if "Compare the incumbent Customer" in system:
                content = json.dumps({"choice": "incumbent", "reason": "Fixture trajectories are identical."})
            else:
                content = json.dumps({"improved": False, "reason": "Fixture trajectories are identical."})
        elif model == "offline-customer":
            role_counts[model] += 1
            content = (
                "I want help with my recent order. My email is fatima.wilson5721@example.com."
                if role_counts[model] % 2 == 1 else "###STOP###"
            )
        elif model == "offline-agent":
            role_counts[model] += 1
            tool_call = {
                "id": f"offline-user-lookup-{role_counts[model]}",
                "type": "function",
                "function": {
                    "name": "find_user_id_by_email",
                    "arguments": json.dumps({"email": "fatima.wilson5721@example.com"}),
                },
            }
            content = None
            response_message = {"role": "assistant", "content": content, "tool_calls": [tool_call]}
            return ModelResponse(
                model=model,
                choices=[{"index": 0, "finish_reason": "tool_calls", "message": response_message}],
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
        elif model == "offline-reviewer":
            system = messages[0]["content"]
            content = (
                json.dumps({"status": "succeeded", "reasoning": "Offline fixture."})
                if "Classify the user authentication outcome" in system
                else json.dumps({"errors": [], "summary": "Offline native review fixture."})
            )
        else:
            raise AssertionError(f"unexpected model call: {model}")

        return ModelResponse(
            model=model,
            choices=[{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _response: 0.0)

    output, result = run_from_config(config_path, tau2_data_dir=data_dir)

    generation = result["generations"][0]
    assert result["status"] == "complete"
    assert generation["customer_phase"]["selection"]["choice"] == "incumbent"
    assert generation["service_phase"]["accepted"] is False
    assert generation["service_phase"]["clean_panel"]["evaluated"] is False
    assert generation["service_phase"]["proposed_strategy"]["text"].startswith("Inspect the requested order")
    records = list((output / "episodes").glob("*/episode-record.json"))
    simulations = list((output / "episodes").glob("*/native-simulation.json"))
    assert len(records) == len(simulations) == 8
    assert any(
        call.get("name") == "find_user_id_by_email"
        for path in simulations
        for message in json.loads(path.read_text(encoding="utf-8"))["messages"]
        for call in (message.get("tool_calls") or ())
    )
    assert result["provider_usage"]["attempts"] == len(calls)
    assert result["provider_usage"]["attempts"] > 0
    assert (output / "alternating-result.json").is_file()
