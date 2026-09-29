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

from evotau.crossplay import build_crossplay_matrix
from evotau.manifest import sha256_json
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import CustomerStrategy, ServiceStrategy
from evotau.web.app import (
    _compatibility_row,
    _crossplay_cells,
    _human_events,
    create_app,
)
from evotau.web.artifact_reader import ArtifactReader
from evotau.web.event_stream import EventJournal
from evotau.web.run_manager import RunManager, RunManagerError
from evotau.web.view_models import (
    budget_view,
    failure_status,
    gate_status,
    strategy_diff_rows,
)


def _write_json(path: Path, payload: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def _write_phase0_fixture(
    root: Path, *, secret: bool = True, run_id: str = "fixture", task_id: str = "73",
    seed: int = 42, protocol: bool | None = None, model_id: str | None = None,
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
        "customer": {"disclosure": "minimal_on_request", "request_order": "reverse_independent",
                     "challenge_style": "ask_reason", "challenge_budget": 1},
        "service": {"rules": []},
    }
    trajectory = {
        "id": f"episode-{task_id}",
        "task_id": task_id,
        "seed": seed,
        "messages": [
            {"role": "user", "turn_idx": 0, "content": "Please update my order."},
            {"role": "assistant", "turn_idx": 1, "content": "I can check that.",
             "tool_calls": [{"id": "call-1", "name": "get_order", "arguments": {"order_id": "12345"}}]},
            {"role": "tool", "turn_idx": 1, "id": "call-1", "content": "status=paid"},
            {"role": "assistant", "turn_idx": 2, "content": "Done."},
        ],
    }
    if secret:
        trajectory["messages"].append({
            "role": "assistant", "turn_idx": 3,
            "content": "Bearer secret-token-value",
            "metadata": {"api_key": "top-secret-api-key"},
        })
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
            "attempts": 4, "cap": 70, "prompt_tokens": None, "completion_tokens": None,
            "usage_unavailable": 1, "model_usage": [],
        },
    }
    _write_json(run_dir / "phase0-result.json", result)
    return run_dir


def _write_pilot_heldout_fixture(root: Path) -> Path:
    run_dir = root / "experiments" / "runs" / "pilot-fixture"
    manifest = {
        "experiment_id": "pilot fixture",
        "phase": "4-pilot",
        "real_provider_enabled": False,
        "request_budget_cap": 100,
        "evolution_seeds": [12],
        "task_selection": {"evolution": ["E"], "validation": ["V"], "heldout": ["H_SECRET_TASK"]},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    _write_json(run_dir / "pilot-manifest.json", manifest)
    episode_dir = run_dir / "seed-blocks" / "seed-12" / "episodes" / "attempt-H"
    customer, service = CustomerStrategy(), ServiceStrategy()
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
    _write_json(episode_dir / "run-telemetry.json", {"panel_name": "heldout", "simulation_id": record.episode_id})
    _write_json(episode_dir / "native-simulation.json", {
        "id": record.episode_id, "task_id": record.task_id, "seed": record.seed,
        "messages": [{"role": "user", "content": "HIDDEN_H_TRAJECTORY_SENTINEL"}],
    })
    matrix = build_crossplay_matrix(
        [record], [], customer_strategies=[customer], service_strategies=[service],
        task_ids=[record.task_id], seeds=[record.seed],
    )
    _write_json(run_dir / "crossplay-matrix.json", matrix.to_dict())
    return run_dir


def _write_completed_pilot_fixture(root: Path) -> Path:
    run_dir = _write_pilot_heldout_fixture(root)
    manifest_path = run_dir / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_dir = run_dir / "seed-blocks/seed-12"
    episode_dir = seed_dir / "episodes/attempt-H"
    trajectory_path = episode_dir / "native-simulation.json"
    telemetry_path = episode_dir / "run-telemetry.json"
    record = json.loads((episode_dir / "episode-record.json").read_text(encoding="utf-8"))
    seed_result = {
        "schema_version": 1,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "evolution_seed": 12,
        "episodes": [record],
        "provider_budget": {"attempts": 2, "cap": 100, "prompt_tokens": 12,
                            "completion_tokens": 6, "usage_unavailable": 0, "model_usage": []},
        "artifacts": [
            {"path": trajectory_path.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest()},
            {"path": telemetry_path.relative_to(root).as_posix(),
             "sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest()},
        ],
    }
    seed_result_path = seed_dir / "pilot-seed-result.json"
    seed_raw = _write_json(seed_result_path, seed_result)
    root_result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "pilot fixture",
        "manifest_sha256": manifest["manifest_sha256"],
        "seed_blocks": [{
            "evolution_seed": 12,
            "result_path": seed_result_path.relative_to(root).as_posix(),
            "result_sha256": hashlib.sha256(seed_raw).hexdigest(),
            "episode_count": 1,
            "provider_budget": seed_result["provider_budget"],
        }],
    }
    _write_json(run_dir / "pilot-result.json", root_result)
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
    client = TestClient(create_app(project_root=project, runs_root=output, manager=manager))
    home = client.get("/")
    assert home.status_code == 200
    assert "运行不会自动开始" in home.text
    assert not calls

    token = re.search(r'name="csrf_token" value="([^"]+)"', home.text).group(1)
    preview = client.post("/preview", data={
        "csrf_token": token, "phase": "0-integration-proof", "config_path": "configs/mvp.yaml",
    })
    assert preview.status_code == 200
    assert "真实 Provider 为 OFF" in preview.text
    assert 'type="submit" disabled' in preview.text
    assert not calls


def test_human_timeline_sorts_full_timestamp_and_formats_local_time():
    run = {
        "run_id": "fixture",
        "events": [
            {"event_id": "later", "event_type": "run_finished", "timestamp": "2026-09-29T00:05:00+00:00"},
            {"event_id": "earlier", "event_type": "run_started", "timestamp": "2026-09-28T23:55:00+00:00"},
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
    args = {role: {"temperature": 0.0, "max_tokens": 512, "thinking_mode": "disabled"}
            for role in models}
    row = _compatibility_row("Model configuration", (models, args), (models, args))
    assert row["same"] is True
    assert row["left"] == "provider/model-a (all roles) · temperature 0.0 · max tokens 512 · thinking_mode disabled"
    changed = _compatibility_row("Task panel", {"evolution": ["73"], "validation": ["93"], "heldout": []},
                                 {"evolution": ["74"], "validation": ["93"], "heldout": []})
    assert changed["same"] is False
    assert changed["left"] == "E: 1 task(s) (73) · V: 1 task(s) (93) · H: 0 task(s)"


def test_run_and_episode_pages_render_normalized_trajectory_redact_secrets_and_append_events(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_phase0_fixture(project)
    result_before = (run_dir / "phase0-result.json").read_bytes()
    trajectory_before = (run_dir / "trajectory.json").read_bytes()
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))

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


def test_heldout_is_not_rendered_in_pages_raw_manifest_or_crossplay(tmp_path: Path):
    project = tmp_path / "project"
    run_dir = _write_pilot_heldout_fixture(project)
    manifest_path = run_dir / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_semantic_review"] = {
        "reviewer_id": "reviewer-1", "reviewed_at": "2026-09-29T00:00:00Z",
        "task_reviews": [{"task_id": "H_SECRET_TASK", "rationale": "H_REVIEW_SENTINEL"}],
        "pairwise_reviews": [{"left_task_id": "H_SECRET_TASK", "right_task_id": "E", "rationale": "H_PAIR_SENTINEL"}],
    }
    manifest["task_semantic_review_sha256"] = "f" * 64
    manifest["manifest_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    _write_json(manifest_path, manifest)
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    response = client.get("/runs/pilot-fixture")
    assert response.status_code == 200
    assert "Heldout panel sealed" in response.text
    assert "H_SECRET_TASK" not in response.text
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in response.text
    assert "heldout-episode-secret" not in response.text
    assert "H_REVIEW_SENTINEL" not in response.text
    assert "H_PAIR_SENTINEL" not in response.text
    assert "H_SECRET_TASK" not in response.text.split('id="command-index">', 1)[-1].split("</script>", 1)[0]
    assert client.get("/runs/pilot-fixture/heldout").status_code == 200
    review = client.get("/runs/pilot-fixture/task-review")
    assert review.status_code == 200 and "H_SECRET_TASK" not in review.text
    assert "H_REVIEW_SENTINEL" not in review.text
    assert client.get("/runs/pilot-fixture/episodes/heldout-episode-secret").status_code == 404
    events = client.get("/runs/pilot-fixture/events.jsonl").text
    assert "H_SECRET_TASK" not in events
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in events
    assert (run_dir / "events.jsonl").exists()
    crossplay = client.get("/runs/pilot-fixture/cross-play")
    assert crossplay.status_code == 200
    assert "H_SECRET_TASK" not in crossplay.text
    run = client.app.state.reader.get_run("pilot-fixture")
    assert run["result"] is None
    assert run["raw_artifacts"]["result"] is None


def test_heldout_can_only_be_revealed_after_verified_complete_root_index(tmp_path: Path):
    project = tmp_path / "project"
    _write_completed_pilot_fixture(project)
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    response = client.get("/runs/pilot-fixture")
    assert response.status_code == 200
    assert "H_SECRET_TASK" in response.text
    episode_page = client.get("/runs/pilot-fixture/episodes/heldout-episode-secret")
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
    assert {row["event_type"] for row in first} >= {"run_started", "episode_finished", "tool_call", "tool_result"}


def test_view_models_do_not_conflate_failure_gate_or_unavailable_cost():
    assert failure_status({"verified": True})["label"] == "已验证的 Service 失败"
    assert failure_status({"provisional": True})["label"] == "尚未验证，不计入 fitness"
    assert gate_status({"accepted": True})["label"] == "已接受"
    assert gate_status({"accepted": False})["label"] == "已拒绝"
    assert gate_status({"inconclusive": True, "accepted": False})["label"] == "无法判断"
    summary = budget_view({"attempts": 2, "cap": 10, "prompt_tokens": None,
                           "completion_tokens": None, "cost": "Cost unavailable"})
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
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    response = client.get("/runs/fixture/episodes?task=73&failure=none")
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
    assert 'evotau-display-mode' in client.get("/static/js/console.js").text


def test_protocol_badges_preserve_strict_default_and_unknown_legacy_modes(tmp_path: Path):
    project = tmp_path / "project"
    _write_phase0_fixture(project, run_id="default", protocol=False)
    _write_phase0_fixture(project, run_id="strict", task_id="74", protocol=True)
    _write_phase0_fixture(project, run_id="legacy", task_id="75", protocol=None)
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    assert "Upstream default" in client.get("/runs/default").text
    assert "Strict diagnostic" in client.get("/runs/strict").text
    assert "Unknown (legacy artifact)" in client.get("/runs/legacy").text


def test_run_comparison_requires_known_matching_frozen_conditions(tmp_path: Path):
    project = tmp_path / "project"
    _write_phase0_fixture(project, run_id="run-a", task_id="73", protocol=False, model_id="model-x")
    _write_phase0_fixture(project, run_id="run-b", task_id="74", protocol=False, model_id="model-x")
    _write_phase0_fixture(project, run_id="run-c", task_id="73", protocol=True, model_id="model-x")
    _write_phase0_fixture(project, run_id="run-d", task_id="73", protocol=False, model_id="model-x")
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    different_panel = client.get("/compare?run_a=run-a&run_b=run-b")
    assert different_panel.status_code == 200
    assert "NOT COMPARABLE" in different_panel.text and "Task panel" in different_panel.text
    different_protocol = client.get("/compare?run_a=run-a&run_b=run-c")
    assert "NOT COMPARABLE" in different_protocol.text
    assert "Communication mode" in different_protocol.text
    matched = client.get("/compare?run_a=run-a&run_b=run-d")
    assert "Frozen comparison conditions match" in matched.text
    assert "Native task success" in matched.text


def test_failure_candidate_route_only_accepts_provisional_episode(tmp_path: Path):
    project = tmp_path / "project"
    _write_phase0_fixture(project, secret=False)
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    assert client.get("/runs/fixture/failures/candidate-episode-73").status_code == 404


def test_crossplay_uses_neutral_magnitude_scale_and_metric_specific_denominators():
    cell = {
        "customer_strategy_id": "c", "service_strategy_id": "s",
        "native_success_rate": 0.9, "verified_failure_rate": 0.9,
        "successful_episodes": 9, "valid_episodes": 10,
        "verified_failure_episodes": 3, "strategy_adherent_episodes": 4,
    }
    success = _crossplay_cells({"cells": [cell]}, "native_success_rate")[0]
    failure = _crossplay_cells({"cells": [cell]}, "verified_failure_rate")[0]
    assert success["style"] == failure["style"] == "high"
    assert (success["metric_numerator"], success["metric_denominator"]) == (9, 10)
    assert (failure["metric_numerator"], failure["metric_denominator"]) == (3, 4)
    assert _crossplay_cells({"cells": [{**cell, "verified_failure_rate": None}]}, "verified_failure_rate")[0]["style"] == "unknown"


def test_strategy_diff_is_display_only_and_marks_changed_fields():
    diff = strategy_diff_rows(
        {"disclosure": "minimal_on_request", "request_order": "scenario_order",
         "challenge_style": "none", "challenge_budget": 0},
        {"disclosure": "related_on_request", "request_order": "scenario_order",
         "challenge_style": "ask_reason", "challenge_budget": 1},
        "customer",
    )
    assert diff["available"] is True
    assert {row["field"] for row in diff["rows"] if row["changed"]} == {"信息披露", "挑战方式", "挑战次数"}


def test_run_manager_refuses_provider_off_and_binds_launch_inputs_without_spawning(tmp_path: Path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs/mvp.yaml").write_text(
        (Path(__file__).resolve().parents[1] / "configs/mvp.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    data_dir = project / "tau-data"
    data_dir.mkdir()
    spawns = []
    manager = RunManager(project, project / "experiments/runs", popen=lambda *a, **k: spawns.append(a))
    preview = manager.preview(phase="0-integration-proof", config_path="configs/mvp.yaml", tau2_data_dir=data_dir)
    with pytest.raises(RunManagerError, match="禁用真实 provider"):
        manager.start(
            phase=preview.phase, config_path="configs/mvp.yaml",
            confirmed_manifest_sha256=preview.manifest_sha256,
            confirmed_launch_sha256=preview.launch_sha256,
            tau2_data_dir=data_dir,
        )
    assert spawns == []

    config_path = project / "configs/mvp-enabled.yaml"
    text = (project / "configs/mvp.yaml").read_text(encoding="utf-8")
    text = text.replace("real_provider_enabled: false", "real_provider_enabled: true")
    text = text.replace("agent: null", "agent: mock-agent").replace("customer: null", "customer: mock-customer")
    text = text.replace("reviewer: null", "reviewer: mock-reviewer").replace("evaluator: null", "evaluator: mock-evaluator")
    config_path.write_text(text, encoding="utf-8")
    enabled = manager.preview(phase="0-integration-proof", config_path=config_path, tau2_data_dir=data_dir)
    alternate_data = project / "other-data"
    alternate_data.mkdir()
    with pytest.raises(RunManagerError, match="启动参数在预览后发生变化"):
        manager.start(
            phase=enabled.phase, config_path=config_path,
            confirmed_manifest_sha256=enabled.manifest_sha256,
            confirmed_launch_sha256=enabled.launch_sha256,
            tau2_data_dir=alternate_data,
        )
    assert spawns == []


def test_run_manager_spawns_only_after_matching_preview_and_rejects_duplicate(tmp_path: Path):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    config_path = project / "configs/mvp-enabled.yaml"
    text = (Path(__file__).resolve().parents[1] / "configs/mvp.yaml").read_text(encoding="utf-8")
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
        phase="0-integration-proof", config_path=config_path, tau2_data_dir=data_dir,
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
    app.state.event_journal.sync = lambda _run: (_ for _ in ()).throw(OSError("disk full"))
    response = TestClient(app).get("/runs/fixture")
    assert response.status_code == 200
    assert "事件日志损坏或不可写" in response.text
    assert "offline fixture" in response.text
