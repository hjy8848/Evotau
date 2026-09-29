from __future__ import annotations

import hashlib
import json
import re
import subprocess
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
from evotau.web.app import _crossplay_cells, create_app
from evotau.web.artifact_reader import ArtifactReader
from evotau.web.event_stream import EventJournal
from evotau.web.run_manager import RunManager, RunManagerError
from evotau.web.view_models import budget_view, failure_status, gate_status


def _write_json(path: Path, payload: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def _write_phase0_fixture(root: Path, *, secret: bool = True) -> Path:
    run_dir = root / "experiments" / "runs" / "fixture"
    manifest = {
        "experiment_id": "offline fixture",
        "phase": "0-integration-proof",
        "upstream": {"commit": "a" * 40},
        "real_provider_enabled": False,
        "request_budget_cap": 70,
        "task_selection": {"evolution": ["73"], "validation": ["93"], "heldout": []},
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    strategy = {
        "customer": {"disclosure": "minimal_on_request", "request_order": "reverse_independent",
                     "challenge_style": "ask_reason", "challenge_budget": 1},
        "service": {"rules": []},
    }
    trajectory = {
        "id": "episode-73",
        "task_id": "73",
        "seed": 42,
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
        "experiment_id": "offline fixture",
        "manifest_sha256": manifest["manifest_sha256"],
        "task_id": "73",
        "simulation_id": "episode-73",
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
    assert "Tool call · get_order" in episode_page.text
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
    client = TestClient(create_app(project_root=project, runs_root=project / "experiments/runs"))
    response = client.get("/runs/pilot-fixture")
    assert response.status_code == 200
    assert "Heldout panel sealed" in response.text
    assert "H_SECRET_TASK" not in response.text
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in response.text
    assert "heldout-episode-secret" not in response.text
    assert client.get("/runs/pilot-fixture/episodes/heldout-episode-secret").status_code == 404
    events = client.get("/runs/pilot-fixture/events.jsonl").text
    assert "H_SECRET_TASK" not in events
    assert "HIDDEN_H_TRAJECTORY_SENTINEL" not in events
    assert (run_dir / "events.jsonl").exists()
    crossplay = client.get("/runs/pilot-fixture/cross-play")
    assert crossplay.status_code == 200
    assert "H_SECRET_TASK" not in crossplay.text


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


def test_crossplay_color_direction_respects_success_vs_failure_metrics():
    cell = {
        "customer_strategy_id": "c", "service_strategy_id": "s",
        "native_success_rate": 0.9, "verified_failure_rate": 0.9,
    }
    assert _crossplay_cells({"cells": [cell]}, "native_success_rate")[0]["style"] == "good"
    assert _crossplay_cells({"cells": [cell]}, "verified_failure_rate")[0]["style"] == "bad"


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
