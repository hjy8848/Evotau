import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evotau import inferai_responses
from evotau.alternating import LLMAlternatingEvolvers
from evotau.budget import RequestBudget
from evotau.evolution_candidates import V2Providers
from evotau.provider_stream import IncompleteProviderStream, collect_completion
from evotau.tau_provenance import EVOLUTION_ROLE_NAMES, freeze_role_model_args


def chunk(text=None, finish=None, usage=None):
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}], "usage": usage}


def test_stream_requires_terminal_completion_and_uses_reported_usage():
    usage = {"prompt_tokens": 12, "completion_tokens": 5}
    result = collect_completion(iter([chunk('{"ok":', None), chunk('true}', "stop"), {"choices": [], "usage": usage}]), lambda _: {"usage": {"prompt_tokens": 999}})
    assert result["usage"] == usage
    assert collect_completion(iter([chunk('{}', "stop")]), lambda _: {"usage": {"prompt_tokens": 999}})["usage"] is None


@pytest.mark.parametrize("chunks", [
    [chunk('{}')], [chunk('{}', 'length')], [chunk('{}', 'content_filter')],
    [chunk('{}', 'stop'), chunk('extra')], [chunk('', 'stop')],
    [chunk('{'), {"error": {"message": "upstream error"}}],
])
def test_partial_or_failed_stream_is_never_accepted(chunks):
    with pytest.raises(IncompleteProviderStream):
        collect_completion(iter(chunks), lambda _: pytest.fail("must not build a failed stream"))


def test_midstream_error_preserves_visible_partial_evidence_without_reasoning():
    def broken():
        yield chunk('{"ok":')
        yield {"choices": [{"delta": {"reasoning_content": "SECRET REASONING"}}]}
        raise RuntimeError("connection lost")
    with pytest.raises(IncompleteProviderStream) as raised:
        collect_completion(broken(), lambda _: None)
    assert raised.value.raw_response == '{"ok":'


def test_budget_persists_pending_call_before_provider_and_refuses_uncertain_resume(tmp_path):
    path = tmp_path / "usage.json"
    budget = RequestBudget(1)
    budget.enable_live_usage(path)

    def dispatch():
        snapshot = json.loads(path.read_text())["provider_usage"]
        assert snapshot["in_flight"] == 1 and snapshot["attempts"] == 0
        # A different process cannot discard a possibly billed interrupted call.
        with pytest.raises(ValueError, match="inconsistent"):
            RequestBudget(1).enable_live_usage(path)
        return {"usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    budget.dispatch_external_call(model="test/model", call_name="diagnosis", dispatch=dispatch)
    resumed = RequestBudget(1)
    resumed.enable_live_usage(path)
    assert resumed.snapshot().attempts == 1 and resumed.snapshot().remaining == 0


@pytest.mark.parametrize("validator,flags", [
    ("validate_skill", ("reusable", "policy_subordinate", "no_task_entities")),
    ("validate_customer", ("preserves_facts", "preserves_objective", "interaction_only", "no_benchmark_leakage")),
])
@pytest.mark.parametrize("bad", ["true", 1, None])
def test_validator_invalid_types_fail_closed(monkeypatch, validator, flags, bad):
    provider = V2Providers(None)
    result = {**dict.fromkeys(flags, True), "reason": "test"}
    result[flags[0]] = bad
    monkeypatch.setattr(provider, "call", lambda *a: result)
    with pytest.raises(ValueError, match="structured schema"):
        getattr(provider, validator)({})
    result[flags[0]] = False
    assert getattr(provider, validator)({})[flags[0]] is False  # semantic rejection is valid evidence


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_responses_incomplete_status_rejected_even_with_valid_json(status):
    with pytest.raises(RuntimeError, match="completed"):
        inferai_responses._response_text({"status": status, "output_text": '{}'})


def test_stream_config_is_explicit_and_evolver_only():
    args = {role: {"temperature": 0.0} for role in EVOLUTION_ROLE_NAMES}
    args["evolver"]["stream"] = True
    frozen = dict(freeze_role_model_args(args, roles=EVOLUTION_ROLE_NAMES))
    assert dict(frozen["evolver"])["stream"] is True
    args["agent"]["stream"] = True
    with pytest.raises(ValueError, match="evolver only"):
        freeze_role_model_args(args, roles=EVOLUTION_ROLE_NAMES)


def test_formal_launcher_requires_enabled_provider_and_confirmed_finite_cap():
    path = Path(__file__).resolve().parents[1] / "experiments/execution/run-v2-live-evolution.py"
    spec = importlib.util.spec_from_file_location("readiness_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for enabled, cap, approved in [(False, 10, 10), (True, None, None), (True, 10, None), (True, 10, 11)]:
        with pytest.raises((ValueError, RuntimeError)):
            module.validate_launch_authorization(SimpleNamespace(real_provider_enabled=enabled, request_budget_cap=cap), approved)
    module.validate_launch_authorization(SimpleNamespace(real_provider_enabled=True, request_budget_cap=10), 10)


def test_json_fence_normalization_is_explicit_and_visible_output_retained(tmp_path, monkeypatch):
    from evotau import alternating

    monkeypatch.setattr(LLMAlternatingEvolvers, "_json_call", staticmethod(lambda *a, **kw: alternating._parse_evolver_json('```json\n{"clusters":[]}\n```', call_name=kw["call_name"])))
    provider = V2Providers(LLMAlternatingEvolvers(model="test/model", model_args={"temperature": 0}, output_directory=tmp_path))
    provider.diagnose({"task_interactions": []})
    file = next((tmp_path / "evolver-calls").glob("*/visible-completion.json"))
    record = json.loads(file.read_text())
    assert record["visible_text"].startswith('```json')
    assert record["format_normalization"] == "complete_markdown_fence_only"
    assert not record["semantic_repair"]


def test_real_tau_generate_collects_stream_inside_one_budgeted_request(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    llm_utils = pytest.importorskip("tau2.utils.llm_utils")
    observed = []

    def reply(request):
        observed.append(json.loads(request.content))
        chunks = [
            {"id": "test-stream", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-v4-pro", "choices": [{"index": 0, "delta": {"role": "assistant", "content": '{"clusters":[]}'}, "finish_reason": None}]},
            {"id": "test-stream", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-v4-pro", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "test-stream", "object": "chat.completion.chunk", "created": 1, "model": "deepseek-v4-pro", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}},
        ]
        body = "".join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
        return httpx.Response(200, text=body, headers={"Content-Type": "text/event-stream"})

    client = openai.OpenAI(api_key="offline", base_url="https://inferai.invalid/v1", http_client=httpx.Client(transport=httpx.MockTransport(reply)))
    original = llm_utils.completion
    monkeypatch.setattr(llm_utils, "completion", lambda *a, **kw: original(*a, client=client, **kw))
    budget = RequestBudget(1)
    provider = V2Providers(LLMAlternatingEvolvers(model="openai/deepseek-v4-pro", model_args={"api_base": "https://inferai.invalid/v1", "stream": True}, request_budget=budget, output_directory=tmp_path))
    assert provider.diagnose({"task_interactions": []}) == {"clusters": []}
    assert observed[0]["stream"] is True
    assert observed[0]["stream_options"] == {"include_usage": True}
    assert budget.snapshot().attempts == 1
    assert budget.snapshot().prompt_tokens == 7
    assert budget.snapshot().completion_tokens == 4


def test_readiness_import_contract_allows_only_E_expansion():
    from copy import deepcopy

    import yaml

    from evotau.alternating_manifest import AlternatingManifest

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "readiness_baseline_import", root / "experiments/execution/import-v2-readiness-baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = AlternatingManifest.from_mapping(yaml.safe_load(
        (root / "configs/v2-readiness-pro-rehearsal.yaml").read_text())).to_document()
    child = AlternatingManifest.from_mapping(yaml.safe_load(
        (root / "configs/v2-readiness-pro-e5-rehearsal.yaml").read_text())).to_document()
    assert module.contract(parent) == module.contract(child)
    for key, value in (("request_budget_cap", 999), ("max_steps", 31),
                       ("initial_customer_strategy_sha256", "different")):
        changed = deepcopy(child)
        changed[key] = value
        assert module.contract(parent) != module.contract(changed)
    changed = deepcopy(child)
    changed["evotau"]["source_sha256"] = "different"
    assert module.contract(parent) != module.contract(changed)
    changed = deepcopy(child)
    changed["role_model_args"]["agent"]["temperature"] = .5
    assert module.contract(parent) != module.contract(changed)
