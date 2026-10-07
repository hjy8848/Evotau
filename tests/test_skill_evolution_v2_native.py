"""Real pinned text runtime/backend/evaluator with a deterministic local completion boundary."""

import json
import os
from pathlib import Path

import pytest
import yaml


def test_v2_native_rollout_activator_provider_chain_and_resume(tmp_path, monkeypatch):
    data = Path(
        os.environ.get(
            "EVOTAU_TAU2_DATA_DIR", "/Users/spring/.cache/evotau/tau2-data-b7ea9074"
        )
    )
    if not data.is_dir():
        pytest.skip("pinned tau data unavailable")
    monkeypatch.setenv("TAU2_DATA_DIR", str(data))
    from litellm import ModelResponse
    from tau2.utils import llm_utils

    from evotau.alternating_run import run_from_config
    from evotau.web.artifact_reader import ArtifactReader

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (
            root
            / "configs/alternating-skill-memory-v2-qwen3-7-plus-retail-mechanism-smoke.yaml"
        ).read_text()
    )
    exp = config["experiment"]
    exp["id"] = "offline-v2-native"
    exp["max_steps"] = 2
    exp["generations"] = 1
    exp["task_selection"]["evolution"] = ["66"]
    exp["max_parallel_episodes"] = 1
    exp["output_path"] = "experiments/runs/offline-v2-native"
    exp["checkpoint_path"] = "experiments/checkpoints/offline-v2-native.json"
    exp["models"] = {
        r: "offline-" + r for r in ("agent", "customer", "evaluator", "evolver")
    }
    exp["model_args"] = {r: {"temperature": 0} for r in exp["models"]}
    policy = exp["skill_evolution_v2"]
    policy["activator"] = {
        "model": "offline-activator",
        "model_args": {"temperature": 0},
    }
    policy["service_evolution"]["candidates_per_cluster"] = 1
    policy["service_evolution"]["crossover"] = False
    directory = tmp_path / "configs"
    directory.mkdir()
    file = directory / "v2.yaml"
    file.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    calls = []

    def completion(*, model, messages, **kwargs):
        assert kwargs.get("num_retries") == 0
        calls.append(model)
        system = messages[0]["content"]
        if model == "offline-activator":
            context = json.loads(messages[-1]["content"])
            assert set(context) == {
                "messages",
                "catalog",
            } and "guidance" not in json.dumps(context["catalog"])
            assert not any(
                k in context
                for k in (
                    "task_id",
                    "task_success",
                    "user_scenario",
                    "evaluation_criteria",
                )
            )
            content = json.dumps(
                {
                    "active_skill_ids": [context["catalog"][0]["skill_id"]],
                    "reason": "Specific observed pre-confirmation condition.",
                    "confidence": 0.9,
                }
            )
        elif model == "offline-evolver":
            if "check Customer interaction" in system:
                content = json.dumps(
                    dict.fromkeys(
                        (
                            "preserves_facts",
                            "preserves_objective",
                            "interaction_only",
                            "no_benchmark_leakage",
                        ),
                        True,
                    )
                )
            elif "You evolve the Customer" in system:
                content = json.dumps(
                    {
                        "candidates": [
                            {
                                "strategy": "Ask for clarification without changing facts or objective.",
                                "semantic_family": "clarification",
                                "target_weakness_family": "scope",
                                "substantive_delta_from_prior": "",
                            }
                        ]
                    }
                )
            elif "Failure Diagnoser" in system:
                content = json.dumps(
                    {
                        "clusters": [
                            {
                                "cluster_id": "scope",
                                "root_cause": "Scope ambiguity.",
                                "evidence_task_ids": ["66"],
                                "protected_success_task_ids": [],
                                "recommended_surface": "skill",
                                "recommended_mutation_types": ["add"],
                                "risk": "Overhead.",
                            }
                        ]
                    }
                )
            elif "Validate only whether" in system:
                content = json.dumps(
                    {
                        "reusable": True,
                        "policy_subordinate": True,
                        "no_task_entities": True,
                    }
                )
            else:
                content = json.dumps(
                    {
                        "analysis": "One local repair; fix count 1; risk count 0.",
                        "semantic_family": "scope",
                        "target_cluster_id": "scope",
                        "operation": "add",
                        "target_skill_id": None,
                        "skill": {
                            "trigger": "Ambiguous scope before confirmation.",
                            "guidance": "Ask one concise question only when necessary.",
                            "activation_signature": {
                                "positive_conditions": ["Ambiguous scope is visible."],
                                "negative_conditions": ["Scope is clear."],
                                "interaction_phase": ["before_confirmation"],
                            },
                        },
                        "expected_fixes": ["66"],
                        "protected_cases_at_risk": [],
                        "substantive_delta_from_prior": "",
                    }
                )
        elif model == "offline-evaluator":
            content = json.dumps({"results": []})
        elif model == "offline-customer":
            content = "My email is fatima.wilson5721@example.com. Please help."
        elif model == "offline-agent":
            content = None
            response = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "lookup",
                        "type": "function",
                        "function": {
                            "name": "find_user_id_by_email",
                            "arguments": json.dumps(
                                {"email": "fatima.wilson5721@example.com"}
                            ),
                        },
                    }
                ],
            }
            return ModelResponse(
                model=model,
                choices=[
                    {"index": 0, "finish_reason": "tool_calls", "message": response}
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )
        else:
            raise AssertionError(model)
        return ModelResponse(
            model=model,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _: 0)
    output, result = run_from_config(file, tau2_data_dir=data)
    assert result["status"] == "complete" and result["schema_version"] == 3
    assert result["api_usage_by_role"]["skill_activator"]["calls"] > 0
    assert result["api_usage_by_role"]["reviewer"]["calls"] == 0
    assert result["provider_usage"]["attempts"] == len(calls)
    assert result["api_usage_by_role"]["customer_evolver"]["calls"] == 1
    assert result["api_usage_by_role"]["service_evolver"]["calls"] == 1
    assert result["api_usage_by_role"]["service_diagnoser"]["calls"] == 1
    assert result["api_usage_by_role"]["semantic_validator"]["calls"] == 2
    summary = result["generations"][-1]["activation_summary"]
    assert summary["available"] and summary["activated_turns"] > 0
    assert summary["rendered_skill_tokens"] > 0
    traces = list(output.glob("episodes/*/skill-activation-trace.json"))
    assert traces
    activations = [json.loads(p.read_text()) for p in traces]
    assert any(
        row["decisions"][0]["decision"]["active_skill_ids"]
        for row in activations
        if row["decisions"]
    )
    native = list(output.glob("episodes/*/native-simulation.json"))
    assert any(
        any(m["role"] == "tool" for m in json.loads(p.read_text())["messages"])
        for p in native
    )
    assert not result["heldout_evaluated"]
    before = len(calls)
    _, resumed = run_from_config(file, tau2_data_dir=data)
    assert (
        len(calls) == before and resumed["provider_usage"] == result["provider_usage"]
    )
    reader = ArtifactReader(tmp_path / "experiments/runs", project_root=tmp_path)
    assert reader.get_run("offline-v2-native")["status"] == "complete"
    assert reader.get_run("offline-v2-native")["heldout_sealed"]
    export = os.environ.get("EVOTAU_V2_OFFLINE_EXPORT")
    if export:
        import shutil

        destination = Path(export)
        if destination.exists():
            raise ValueError("offline smoke export must use a new directory")
        shutil.copytree(tmp_path, destination)
        (destination / "offline-evidence.json").write_text(
            json.dumps(
                {
                    "kind": "deterministic-native-integration",
                    "network_provider_calls": 0,
                    "completion_boundary": "local scripted ModelResponse; not live InferAI",
                    "native_layers": [
                        "UserSimulator",
                        "LLMAgent",
                        "Retail tools/backend",
                        "native evaluator",
                    ],
                    "episode_count": result["completed_episodes"],
                    "instrumented_local_calls": len(calls),
                    "resume_additional_calls": len(calls) - before,
                    "heldout_evaluated": False,
                    "effectiveness_claim": False,
                },
                indent=2,
            )
            + "\n"
        )


def test_v2_evolver_real_generate_boundary_honors_request_cap(monkeypatch):
    from litellm import ModelResponse
    from tau2.utils import llm_utils

    from evotau.alternating import LLMAlternatingEvolvers
    from evotau.budget import ProviderBudgetExceeded, RequestBudget
    from evotau.evolution_candidates import V2Providers

    dispatches = []

    def completion(**kwargs):
        dispatches.append(kwargs["model"])
        return ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ready":true}'},
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _: 0)
    budget = RequestBudget(1)
    provider = V2Providers(
        LLMAlternatingEvolvers(model="offline", model_args={}, request_budget=budget)
    )
    assert provider.call("tiny", {}, "evotau_service_diagnoser") == {"ready": True}
    with pytest.raises(ProviderBudgetExceeded):
        provider.call("another", {}, "evotau_service_skill_mutator")
    assert len(dispatches) == 1 and budget.snapshot().attempts == 1
