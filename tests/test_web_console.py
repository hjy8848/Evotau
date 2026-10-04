from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
from fastapi.testclient import TestClient

from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import PromptStrategy
from evotau.tau_provenance import sha256_json
from evotau.web.app import (
    _compatibility_row,
    _human_events,
    create_app,
)
from evotau.web.artifact_reader import ArtifactReader
from evotau.web.event_stream import EventJournal
from evotau.web.run_manager import RunManager, RunManagerError
from evotau.web.view_models import (
    budget_view,
    customer_strategy_view,
    generation_view,
    strategy_diff_rows,
)


def _write_json(path: Path, payload: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def _write_phase0_fixture(
    root: Path,
    *,
    secret: bool = True,
    run_id: str = "fixture",
    task_id: str = "73",
    seed: int = 42,
    protocol: bool | None = None,
    model_id: str | None = None,
) -> Path:
    run_dir = root / "experiments" / "runs" / run_id
    manifest = {
        "experiment_id": f"offline {run_id}",
        "phase": "0-integration-proof",
        "upstream": {"commit": "a" * 40},
        "real_provider_enabled": False,
        "request_budget_cap": 70,
        "seed": seed,
        "task_selection": {"evolution": [task_id], "validation": ["93"], "heldout": []},
    }
    if protocol is not None:
        manifest["enforce_communication_protocol"] = protocol
    if model_id is not None:
        manifest["role_models"] = {"agent": model_id}
        manifest["role_model_args"] = {"agent": {"temperature": 0}}
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    strategy = {
        "customer": {"text": "Pursue the task scenario and respond naturally."},
        "service": {"text": "Use the τ-bench policy and tools carefully."},
    }
    trajectory = {
        "id": f"episode-{task_id}",
        "task_id": task_id,
        "seed": seed,
        "messages": [
            {"role": "user", "turn_idx": 0, "content": "Please update my order."},
            {
                "role": "assistant",
                "turn_idx": 1,
                "content": "I can check that.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "get_order",
                        "arguments": {"order_id": "12345"},
                    }
                ],
            },
            {"role": "tool", "turn_idx": 1, "id": "call-1", "content": "status=paid"},
            {"role": "assistant", "turn_idx": 2, "content": "Done."},
        ],
    }
    if secret:
        trajectory["messages"].append(
            {
                "role": "assistant",
                "turn_idx": 3,
                "content": "Bearer secret-token-value",
                "metadata": {"api_key": "top-secret-api-key"},
            }
        )
    _write_json(run_dir / "trajectory.json", trajectory)
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": f"offline {run_id}",
        "manifest_sha256": manifest["manifest_sha256"],
        "task_id": task_id,
        "simulation_id": f"episode-{task_id}",
        "simulation_file": "trajectory.json",
        "native_reward": 1.0,
        "strategy": strategy,
        "provider_budget": {
            "attempts": 4,
            "cap": 70,
            "prompt_tokens": None,
            "completion_tokens": None,
            "usage_unavailable": 1,
            "model_usage": [],
        },
    }
    _write_json(run_dir / "phase0-result.json", result)
    return run_dir


def _write_alternating_heldout_fixture(root: Path) -> Path:
    run_dir = root / "experiments" / "runs" / "alternating-fixture"
    manifest = {
        "experiment_id": "alternating fixture",
        "phase": "alternating-self-evolution",
        "real_provider_enabled": False,
        "request_budget_cap": None,
        "seed": 12,
        "task_panels": {"E": ["E"], "V": ["V"], "H": ["H_SECRET_TASK"]},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    _write_json(run_dir / "manifest.json", manifest)
    episode_dir = run_dir / "episodes" / "attempt-H"
    customer, service = PromptStrategy("adaptive"), PromptStrategy("careful")
    record = EpisodeRecord(
        episode_id="heldout-episode-secret",
        task_id="H_SECRET_TASK",
        seed=12,
        customer_strategy_id=customer_strategy_id(customer),
        service_strategy_id=service_strategy_id(service),
        status=EpisodeStatus.COMPLETE,
        task_success=True,
        native_reward=1.0,
        trajectory_ref="episodes/attempt-H/native-simulation.json",
    )
    _write_json(episode_dir / "episode-record.json", record.to_dict())
    _write_json(
        episode_dir / "run-telemetry.json",
        {"panel_name": "heldout-native_customer-S0", "simulation_id": record.episode_id},
    )
    _write_json(
        episode_dir / "native-simulation.json",
        {
            "id": record.episode_id,
            "task_id": record.task_id,
            "seed": record.seed,
            "messages": [{"role": "user", "content": "HIDDEN_H_TRAJECTORY_SENTINEL"}],
        },
    )
    return run_dir


def _complete_alternating_heldout_fixture(root: Path) -> Path:
    run_dir = _write_alternating_heldout_fixture(root)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _write_json(run_dir / "alternating-result.json", {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "alternating fixture",
        "manifest_sha256": manifest["manifest_sha256"],
        "initial_customer": {"text": "initial"},
        "initial_service": {"text": "initial"},
        "final_customer": {"text": "adaptive"},
        "final_service": {"text": "careful"},
        "generations": [{
            "generation": 0,
            "customer_before": {"strategy_id": "c0", "strategy": "base customer"},
            "customer_after": {"strategy_id": "c1", "strategy": "adaptive customer strategy"},
            "service_before": {"strategy_id": "s0", "strategy": "base service"},
            "service_after": {"strategy_id": "s0", "strategy": "base service"},
            "customer_phase": {
                "candidates": [],
                "selection": {"choice": "incumbent", "reason": "Incumbent showed the clearest weakness."},
            },
            "service_phase": {
                "accepted": False,
                "proposed_strategy": {"text": "Inspect the order before replying."},
                "selection": {"improved": False, "reason": "The replay remained unsuccessful."},
                "clean_panel": {"evaluated": False, "catastrophic_regression": None},
            },
        }],
        "provider_usage": {"attempts": 2, "cap": None, "prompt_tokens": 10,
                            "completion_tokens": 5, "usage_unavailable": 0, "model_usage": []},
    })
    return run_dir


def test_home_does_not_start_a_run_and_preview_blocks_provider_off(tmp_path: Path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    config_source = Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"
    config = config_source.read_text(encoding="utf-8")
    (project / "configs" / "mvp.yaml").write_text(config, encoding="utf-8")
    calls = []

    class NeverSpawn:
        def __call__(self, *_args, **_kwargs):
            calls.append(True)
            raise AssertionError("provider run was unexpectedly spawned")

    output = project / "experiments" / "runs"
    manager = RunManager(project, output, popen=NeverSpawn())
    client = TestClient(
        create_app(project_root=project, runs_root=output, manager=manager)
    )
    home = client.get("/")
    assert home.status_code == 200
    assert "运行不会自动开始" in home.text
    assert not calls

    token = re.search(r'name="csrf_token" value="([^"]+)"', home.text).group(1)
    preview = client.post(
        "/preview",
        data={
            "csrf_token": token,
            "phase": "0-integration-proof",
            "config_path": "configs/mvp.yaml",
        },
    )
    assert preview.status_code == 200
    assert "真实 Provider 为 OFF" in preview.text
    assert 'type="submit" disabled' in preview.text
    assert not calls


def test_human_timeline_sorts_full_timestamp_and_formats_local_time():
    run = {
        "run_id": "fixture",
        "events": [
            {
                "event_id": "later",
                "event_type": "run_finished",
                "timestamp": "2026-09-29T00:05:00+00:00",
            },
            {
                "event_id": "earlier",
                "event_type": "run_started",
                "timestamp": "2026-09-28T23:55:00+00:00",
            },
        ],
    }
    events = _human_events(run)
    assert [event["event_id"] for event in events] == ["earlier", "later"]
    expected = [
        datetime.fromisoformat(event["timestamp"]).astimezone().strftime("%H:%M")
        for event in [run["events"][1], run["events"][0]]
    ]
    assert [event["time"] for event in events] == expected


def test_run_comparison_formats_conditions_for_humans_but_compares_full_values():
    models = {"agent": "provider/model-a", "customer": "provider/model-a"}
    args = {
        role: {"temperature": 0.0, "max_tokens": 512, "thinking_mode": "disabled"}
        for role in models
    }
    row = _compatibility_row("Model configuration", (models, args), (models, args))
    assert row["same"] is True
    assert (
        row["left"]
        == "provider/model-a (all roles) · temperature 0.0 · max tokens 512 · thinking_mode disabled"
    )
    changed = _compatibility_row(
        "Task panel",
        {"evolution": ["73"], "validation": ["93"], "heldout": []},
        {"evolution": ["74"], "validation": ["93"], "heldout": []},
    )
    assert changed["same"] is False
    assert changed["left"] == "E: 1 task(s) (73) · V: 1 task(s) (93) · H: 0 task(s)"


def test_run_and_episode_pages_render_normalized_trajectory_redact_secrets_and_append_events(
    tmp_path: Path,
):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project)
    result_before = (run_dir / "phase0-result.json").read_bytes()
    trajectory_before = (run_dir / "trajectory.json").read_bytes()
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )

    run_page = client.get("/runs/fixture")
    assert run_page.status_code == 200
    assert "运行不会" not in run_page.text
    assert "Unavailable" in run_page.text
    assert "Bearer secret-token-value" not in run_page.text
    assert "top-secret-api-key" not in run_page.text
    assert "events.jsonl" not in run_page.text

    episode_page = client.get("/runs/fixture/episodes/episode-73")
    assert episode_page.status_code == 200
    assert "Customer" in episode_page.text
    assert "Please update my order." in episode_page.text
    assert episode_page.text.count("TOOL CALL · get_order") == 1
    assert "Tool Result" in episode_page.text
    assert "DB state trace unavailable." in episode_page.text
    assert "按任务给定顺序" not in episode_page.text
    assert "Bearer secret-token-value" not in episode_page.text
    assert "top-secret-api-key" not in episode_page.text

    events_path = run_dir / "events.jsonl"
    first = events_path.read_bytes()
    client.get("/runs/fixture")
    assert events_path.read_bytes() == first
    assert (run_dir / "phase0-result.json").read_bytes() == result_before
    assert (run_dir / "trajectory.json").read_bytes() == trajectory_before
    assert b"Bearer secret-token-value" not in first
    assert b"top-secret-api-key" not in first
    events = client.get("/runs/fixture/events.jsonl").json()
    assert len({item["event_id"] for item in events}) == len(events)


def test_episode_console_places_verified_db_change_after_its_tool_call_in_both_modes(
    tmp_path: Path,
):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project, secret=False)
    trajectory_path = run_dir / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["messages"][1]["tool_calls"][0]["name"] = "update_order_status"
    trajectory["messages"][2]["name"] = "update_order_status"
    trajectory["messages"][3] = {
        "role": "assistant",
        "turn_idx": 2,
        "tool_calls": [
            {
                "id": "call-2",
                "name": "return_delivered_order_items",
                "arguments": {"order_id": "12345", "item_ids": ["item-1"]},
            }
        ],
    }
    trajectory["messages"].extend(
        [
            {
            "role": "tool",
            "turn_idx": 2,
            "id": "call-2",
            "name": "return_delivered_order_items",
            "content": "status=return requested",
            },
            {"role": "assistant", "turn_idx": 3, "content": "Done."},
        ]
    )
    _write_json(trajectory_path, trajectory)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    trace = {
        "schema_version": 1,
        "task_id": "73",
        "episode_id": "episode-73",
        "trajectory_ref": "native-simulation.json",
        "source": "deterministic_replay",
        "status": "complete",
        "provenance": {
            "manifest_sha256": manifest["manifest_sha256"],
            "tau_bench_commit": "b" * 40,
            "trajectory_sha256": hashlib.sha256(
                trajectory_path.read_bytes()
            ).hexdigest(),
        },
        "summary": {
            "initial_state_hash": "c" * 64,
            "actual_final_state_hash": "d" * 64,
            "db_match": True,
            "write_tool_call_count": 1,
            "mutation_event_count": 1,
            "field_change_count": 1,
            "final_comparison": {"db_match": True, "differences": None},
        },
        "events": [
            {
                "event_index": 0,
                "turn_idx": 1,
                "tool_call_id": "call-1",
                "requestor": "assistant",
                "tool_name": "get_order",
                "arguments": {"order_id": "12345"},
                "changes": [
                    {
                        "entity_type": "order",
                        "entity_id": "12345",
                        "field_path": "status",
                        "before": "paid",
                        "after": "shipped",
                    }
                ],
            }
        ],
    }
    trace["trace_sha256"] = sha256_json(trace)
    _write_json(run_dir / "db-state-trace.json", trace)
    client = TestClient(
        create_app(
            project_root=project,
            runs_root=project / "experiments/runs",
        )
    )
    page = client.get("/runs/fixture/episodes/episode-73")

    assert page.status_code == 200
    assert "DB match ✓" in page.text
    assert "Environment outcome" in page.text
    assert "Order" in page.text
    assert "Status" in page.text
    assert "paid → shipped" in page.text
    assert "call-1" in page.text
    assert page.text.count('aria-label="Environment change"') == 1
    assert 'class="simple-only"' in page.text
    assert 'class="research-only"' in page.text
    artifact_page = client.get("/runs/fixture/artifacts")
    assert "db-state-trace.json" in artifact_page.text

    trace["events"].append(
        {
            "event_index": 1,
            "turn_idx": 2,
            "tool_call_id": "call-2",
            "requestor": "assistant",
            "tool_name": "return_delivered_order_items",
            "arguments": {"order_id": "12345", "item_ids": ["item-1"]},
            "changes": [
                {
                    "entity_type": "order",
                    "entity_id": "12345",
                    "field_path": "return_items",
                    "before": None,
                    "after": ["item-1"],
                }
            ],
        }
    )
    trace["summary"].update(
        {"write_tool_call_count": 2, "mutation_event_count": 2, "field_change_count": 2}
    )
    trace.pop("trace_sha256")
    trace["trace_sha256"] = sha256_json(trace)
    _write_json(run_dir / "db-state-trace.json", trace)
    multiple_page = client.get("/runs/fixture/episodes/episode-73")
    assert multiple_page.text.count('aria-label="Environment change"') == 2
    first_tool = multiple_page.text.index("TOOL CALL · update_order_status")
    first_change = multiple_page.text.index('aria-label="Environment change"')
    first_result = multiple_page.text.index("TOOL RESULT · update_order_status")
    second_tool = multiple_page.text.index("TOOL CALL · return_delivered_order_items")
    second_change = multiple_page.text.rindex('aria-label="Environment change"')
    second_result = multiple_page.text.index("TOOL RESULT · return_delivered_order_items")
    assert first_tool < first_change < first_result < second_tool < second_change < second_result

    trace["events"] = []
    trace["summary"].update(
        {"write_tool_call_count": 2, "mutation_event_count": 0, "field_change_count": 0}
    )
    trace.pop("trace_sha256")
    trace["trace_sha256"] = sha256_json(trace)
    _write_json(run_dir / "db-state-trace.json", trace)
    zero_page = client.get("/runs/fixture/episodes/episode-73")
    assert "Mutation events</small><strong>0</strong>" in zero_page.text
    assert 'aria-label="Environment change"' not in zero_page.text

    trace["summary"]["db_match"] = False
    trace["summary"]["final_comparison"] = {
        "db_match": False,
        "gold_state_available": True,
        "differences": [
            {
                "entity_type": "order",
                "entity_id": "12345",
                "field_path": "status",
                "actual": "shipped",
                "expected": "return requested",
            }
        ],
    }
    trace.pop("trace_sha256")
    trace["trace_sha256"] = sha256_json(trace)
    _write_json(run_dir / "db-state-trace.json", trace)
    mismatch_page = client.get("/runs/fixture/episodes/episode-73")
    assert "Actual vs expected" in mismatch_page.text
    assert "Actual: shipped" in mismatch_page.text
    assert "Expected: return requested" in mismatch_page.text
    assert "native review to understand the outcome" in mismatch_page.text


def test_missing_and_malformed_runs_fail_closed_with_friendly_pages(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project, secret=False)
    app = create_app(project_root=project, runs_root=project / "experiments/runs")
    client = TestClient(app)
    assert client.get("/runs/missing").status_code == 404
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_id"] = "tampered"
    _write_json(manifest_path, manifest)
    response = client.get("/runs/fixture")
    assert response.status_code == 422
    assert "Run 已隐藏" in response.text
    assert "Traceback" not in response.text

    manifest["experiment_id"] = "offline fixture"
    _write_json(manifest_path, manifest)
    result_path = run_dir / "phase0-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["strategy"] = ["malformed"]
    _write_json(result_path, result)
    malformed = client.get("/runs/fixture")
    assert malformed.status_code == 422
    assert "Traceback" not in malformed.text


def test_heldout_is_sealed_from_current_console_until_run_completion(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_alternating_heldout_fixture(project)
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )
    response = client.get("/runs/alternating-fixture")
    assert response.status_code == 200
    assert "Heldout panel sealed" in response.text
    assert "H_SECRET_TASK" not in response.text
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in response.text
    assert "heldout-episode-secret" not in response.text
    assert (
        "H_SECRET_TASK"
        not in response.text.split('id="command-index">', 1)[-1].split("</script>", 1)[
            0
        ]
    )
    assert client.get("/runs/alternating-fixture/heldout").status_code == 200
    assert (
        client.get("/runs/alternating-fixture/episodes/heldout-episode-secret").status_code
        == 404
    )
    assert client.get("/runs/alternating-fixture/task-review").status_code == 404
    assert client.get("/runs/alternating-fixture/cross-play").status_code == 404
    events = client.get("/runs/alternating-fixture/events.jsonl").text
    assert "H_SECRET_TASK" not in events
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in events
    assert (run_dir / "events.jsonl").exists()
    run = client.app.state.reader.get_run("alternating-fixture")
    assert run["result"] is None
    assert run["raw_artifacts"]["result"] is None


def test_heldout_is_revealed_only_after_complete_alternating_result(
    tmp_path: Path,
):
    project = tmp_path / "project"
    _complete_alternating_heldout_fixture(project)
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )
    response = client.get("/runs/alternating-fixture")
    assert response.status_code == 200
    assert "H_SECRET_TASK" in response.text
    assert "adaptive customer strategy" in response.text
    assert "Inspect the order before replying." in response.text
    episode_page = client.get("/runs/alternating-fixture/episodes/heldout-episode-secret")
    assert episode_page.status_code == 200
    assert "H_SECRET_TASK" in episode_page.text
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" in episode_page.text


def test_event_journal_is_derived_append_only_and_resumable(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project, secret=False)
    reader = ArtifactReader(project / "experiments/runs", project_root=project)
    run = reader.get_run("fixture")
    journal = EventJournal()
    first = journal.sync(run)
    before = (run_dir / "events.jsonl").read_bytes()
    second = journal.sync(reader.get_run("fixture"))
    assert first == second
    assert (run_dir / "events.jsonl").read_bytes() == before
    assert {row["event_type"] for row in first} >= {
        "run_started",
        "episode_finished",
        "tool_call",
        "tool_result",
    }


def test_view_models_show_alternating_selection_and_open_service_proposal():
    view = generation_view({
        "generation": 0,
        "customer_before": {"strategy_id": "c0", "strategy": "old customer"},
        "customer_after": {"strategy_id": "c1", "strategy": "new customer"},
        "service_before": {"strategy_id": "s0", "strategy": "old service"},
        "service_after": {"strategy_id": "s0", "strategy": "old service"},
        "customer_phase": {"selection": {"choice": 0, "reason": "More revealing trajectory."}},
        "service_phase": {
            "accepted": False,
            "proposed_strategy": {"text": "A natural-language Service proposal."},
            "selection": {"improved": False, "reason": "Same native outcome."},
        },
    })
    assert view["customer_evolved"] is True
    assert view["service_accepted"] is False
    assert "Same native outcome" in view["service_reason"]
    assert view["proposed_service_text"] == "A natural-language Service proposal."
    summary = budget_view(
        {
            "attempts": 2,
            "cap": 10,
            "prompt_tokens": None,
            "completion_tokens": None,
            "cost": "Cost unavailable",
        }
    )
    assert summary["attempt_label"] == "2 / 10"
    assert summary["token_label"] == "Unavailable"
    assert summary["total_tokens"] is None
    assert summary["cost_label"] == "Cost unavailable"


def test_episode_filters_inspector_and_artifact_index_are_read_only(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project, secret=False)
    before = {
        name: (run_dir / name).read_bytes()
        for name in ("manifest.json", "phase0-result.json", "trajectory.json")
    }
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )
    response = client.get("/runs/fixture/episodes?task=73")
    assert response.status_code == 200
    assert "episode-73" in response.text
    assert "data-inspector-target" in response.text
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in response.text
    assert client.get("/runs/fixture/episodes?task=999").status_code == 200
    assert "No episodes match" in client.get("/runs/fixture/episodes?task=999").text
    artifacts = client.get("/runs/fixture/artifacts")
    assert artifacts.status_code == 200
    assert "manifest.json" in artifacts.text and "phase0-result.json" in artifacts.text
    assert "SHA-256" in artifacts.text
    for name, payload in before.items():
        assert (run_dir / name).read_bytes() == payload
    assert client.get("/static/js/console.js").status_code == 200
    assert "evotau-display-mode" in client.get("/static/js/console.js").text


def test_protocol_badges_preserve_strict_default_and_unknown_legacy_modes(
    tmp_path: Path,
):
    project = tmp_path / "project"
    _write_phase0_fixture(project, run_id="default", protocol=False)
    _write_phase0_fixture(project, run_id="strict", task_id="74", protocol=True)
    _write_phase0_fixture(project, run_id="legacy", task_id="75", protocol=None)
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )
    assert "Upstream default" in client.get("/runs/default").text
    assert "Strict diagnostic" in client.get("/runs/strict").text
    assert "Unknown (legacy artifact)" in client.get("/runs/legacy").text


def test_run_comparison_requires_known_matching_frozen_conditions(tmp_path: Path):
    project = tmp_path / "project"
    _write_phase0_fixture(
        project, run_id="run-a", task_id="73", protocol=False, model_id="model-x"
    )
    _write_phase0_fixture(
        project, run_id="run-b", task_id="74", protocol=False, model_id="model-x"
    )
    _write_phase0_fixture(
        project, run_id="run-c", task_id="73", protocol=True, model_id="model-x"
    )
    _write_phase0_fixture(
        project, run_id="run-d", task_id="73", protocol=False, model_id="model-x"
    )
    client = TestClient(
        create_app(project_root=project, runs_root=project / "experiments/runs")
    )
    different_panel = client.get("/compare?run_a=run-a&run_b=run-b")
    assert different_panel.status_code == 200
    assert (
        "NOT COMPARABLE" in different_panel.text
        and "Task panel" in different_panel.text
    )
    different_protocol = client.get("/compare?run_a=run-a&run_b=run-c")
    assert "NOT COMPARABLE" in different_protocol.text
    assert "Communication mode" in different_protocol.text
    matched = client.get("/compare?run_a=run-a&run_b=run-d")
    assert "Frozen comparison conditions match" in matched.text
    assert "Native task success" in matched.text


def test_strategy_diff_is_display_only_and_marks_open_text_changes():
    diff = strategy_diff_rows(
        {"text": "Ask one clear question."},
        {"text": "Clarify the request, then ask one clear question."},
        "customer",
    )
    assert diff["available"] is True
    assert len(diff["rows"]) == 1
    assert diff["rows"][0]["changed"] is True


def test_open_text_customer_strategy_is_displayed_verbatim():
    view = customer_strategy_view({"text": "Use the scenario and adapt to the service."})
    assert view["available"] is True
    assert view["version"] == "open-text"
    assert view["text"] == "Use the scenario and adapt to the service."


def test_run_manager_refuses_provider_off_and_binds_launch_inputs_without_spawning(
    tmp_path: Path,
):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs/mvp.yaml").write_text(
        (Path(__file__).resolve().parents[1] / "configs/mvp.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    data_dir = project / "tau-data"
    data_dir.mkdir()
    spawns = []
    manager = RunManager(
        project, project / "experiments/runs", popen=lambda *a, **k: spawns.append(a)
    )
    preview = manager.preview(
        phase="0-integration-proof",
        config_path="configs/mvp.yaml",
        tau2_data_dir=data_dir,
    )
    with pytest.raises(RunManagerError, match="禁用真实 provider"):
        manager.start(
            phase=preview.phase,
            config_path="configs/mvp.yaml",
            confirmed_manifest_sha256=preview.manifest_sha256,
            confirmed_launch_sha256=preview.launch_sha256,
            tau2_data_dir=data_dir,
        )
    assert spawns == []

    config_path = project / "configs/mvp-enabled.yaml"
    text = (project / "configs/mvp.yaml").read_text(encoding="utf-8")
    text = text.replace("real_provider_enabled: false", "real_provider_enabled: true")
    text = text.replace("agent: null", "agent: mock-agent").replace(
        "customer: null", "customer: mock-customer"
    )
    text = text.replace("reviewer: null", "reviewer: mock-reviewer").replace(
        "evaluator: null", "evaluator: mock-evaluator"
    )
    config_path.write_text(text, encoding="utf-8")
    enabled = manager.preview(
        phase="0-integration-proof", config_path=config_path, tau2_data_dir=data_dir
    )
    alternate_data = project / "other-data"
    alternate_data.mkdir()
    with pytest.raises(RunManagerError, match="启动参数在预览后发生变化"):
        manager.start(
            phase=enabled.phase,
            config_path=config_path,
            confirmed_manifest_sha256=enabled.manifest_sha256,
            confirmed_launch_sha256=enabled.launch_sha256,
            tau2_data_dir=alternate_data,
        )
    assert spawns == []


def test_run_manager_spawns_only_after_matching_preview_and_rejects_duplicate(
    tmp_path: Path,
):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    config_path = project / "configs/mvp-enabled.yaml"
    text = (Path(__file__).resolve().parents[1] / "configs/mvp.yaml").read_text(
        encoding="utf-8"
    )
    text = text.replace("real_provider_enabled: false", "real_provider_enabled: true")
    for role in ("agent", "customer", "reviewer", "evaluator"):
        text = text.replace(f"{role}: null", f"{role}: mock-{role}")
    config_path.write_text(text, encoding="utf-8")
    data_dir = project / "tau-data"
    data_dir.mkdir()

    class RunningProcess:
        def poll(self):
            return None

    spawned = []

    def fake_popen(command, **kwargs):
        spawned.append((command, kwargs))
        return RunningProcess()

    manager = RunManager(project, project / "experiments/runs", popen=fake_popen)
    preview = manager.preview(
        phase="0-integration-proof",
        config_path=config_path,
        tau2_data_dir=data_dir,
    )
    started = manager.start(
        phase=preview.phase,
        config_path=config_path,
        confirmed_manifest_sha256=preview.manifest_sha256,
        confirmed_launch_sha256=preview.launch_sha256,
        tau2_data_dir=data_dir,
    )
    assert started["status"] == "running"
    assert spawned[0][0][1:3] == ["-m", "evotau.phase0_run"]
    assert spawned[0][1]["stdin"] == subprocess.DEVNULL
    assert manager.status_for_run(started["run_id"]) == "running"
    with pytest.raises(RunManagerError, match="已经在运行"):
        manager.start(
            phase=preview.phase,
            config_path=config_path,
            confirmed_manifest_sha256=preview.manifest_sha256,
            confirmed_launch_sha256=preview.launch_sha256,
            tau2_data_dir=data_dir,
        )
    assert len(spawned) == 1


def test_event_journal_failure_is_display_only(tmp_path: Path):
    project = tmp_path / "project"
    _write_phase0_fixture(project, secret=False)
    app = create_app(project_root=project, runs_root=project / "experiments/runs")
    app.state.event_journal.sync = lambda _run: (_ for _ in ()).throw(
        OSError("disk full")
    )
    response = TestClient(app).get("/runs/fixture")
    assert response.status_code == 200
    assert "事件日志损坏或不可写" in response.text
    assert "offline fixture" in response.text
