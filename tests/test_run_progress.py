from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evotau.records import EpisodeRecord, EpisodeStatus
from evotau.tau_provenance import sha256_json
from evotau.web.app import create_app


def put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def monitor(tmp_path):
    run = tmp_path / "experiments/runs/monitor"
    manifest = {"experiment_id": "monitor", "phase": "alternating-self-evolution", "domain": "retail",
                "generations": 2, "max_parallel_episodes": 1, "real_provider_enabled": False,
                "task_panels": {"E": ["66", "92", "29"], "V": ["44"], "H": ["H_SECRET_TASK"]},
                "checkpoint_path": "experiments/checkpoints/monitor.json", "request_budget_cap": None,
                "evotau": {"git_commit": "a" * 40}}
    digest = sha256_json(manifest); manifest["manifest_sha256"] = digest; put(run / "manifest.json", manifest)
    c0 = {"strategy_id": "c0", "strategy": ""}; c1 = {"strategy_id": "c1", "strategy": "original challenge"}
    s0 = {"strategy_id": "s0", "strategy": {"skills": []}, "carrier": "skill_memory_v1"}
    s1 = {"strategy_id": "s1", "strategy": {"skills": [{"skill_id": "skill-0001", "trigger": "mixed order actions", "guidance": "exchange before return"}]}, "carrier": "skill_memory_v1"}
    generation = {"generation": 0, "customer_before": c0, "customer_after": c1, "service_before": s0, "service_after": s1, "customer_phase": {"candidates": []}, "service_phase": {"accepted": True}}
    put(tmp_path / manifest["checkpoint_path"], {"manifest_sha256": digest, "completed_generation": 0, "generations": [generation]})
    stage = {"manifest_sha256": digest, "generation": 1, "stage": "customer_proposals_ready"}
    put(run / "generation-0001-stage.json", stage)
    put(run / "generation-0001-customer-proposals.json", {"manifest_sha256": digest, "customer_candidates": [{"strategy": "new challenge", "strategy_id": "c2"}]})
    put(run / "continuation-provenance.json", {"source_experiment_id": "parent", "imported_complete_episodes": 4})
    refs = {}

    def episode(task: str, panel: str, success: bool, reused: bool = False):
        eid = f"{panel}-{task}"; directory = run / "episodes" / eid
        record = EpisodeRecord(episode_id=eid, task_id=task, seed=1,
                               customer_strategy_id="c2" if "candidate" in panel else "c1",
                               service_strategy_id="s1", status=EpisodeStatus.COMPLETE,
                               task_success=success, native_reward=float(success),
                               trajectory_ref=f"episodes/{eid}/native-simulation.json")
        put(directory / "episode-record.json", record.to_dict())
        put(directory / "run-telemetry.json", {"simulation_id": eid, "panel_name": panel, "budget_after": {"attempts": 1}})
        put(directory / "native-simulation.json", {"id": eid, "task_id": task, "seed": 1, "messages": []})
        refs[eid] = {"panel_name": panel, "task_id": task, "seed": 1, "episode_id": eid, "reused": reused}
        put(run / "episode-panel-references.json", {"manifest_sha256": digest, "references": refs})
    for task in ("66", "92", "29"):
        episode(task, "generation-1-customer-incumbent", task != "92", True)
    episode("66", "generation-1-customer-candidate-0", False)
    diag = "episodes/failure/incomplete-run.json"
    put(run / diag, {"attempt_id": "failure", "task_id": "29", "seed": 1, "panel_name": "generation-1-customer-candidate-0", "failure_message": "Upstream unavailable", "partial_trajectory_ref": "episodes/failure/partial-simulation.json"})
    put(run / "episodes/failure/partial-simulation.json", {"attempt_id": "failure", "task_id": "29", "seed": 1, "status": "incomplete", "messages": [{"role": "user", "turn_idx": 7, "content": "hello"}]})
    failure = {"stage": "evolution", "task_id": "29", "panel_name": "generation-1-customer-candidate-0", "diagnostics_ref": diag, "failure_message": "502 sk-testsecretabcdefghijk unavailable"}
    execution = {"manifest_sha256": digest, "status": "failed", "failure": failure}
    put(run / "run-execution-state.json", execution)
    put(run / "api-usage-live.json", {"provider_usage": {"attempts": 3, "successes": 2, "failures": 1, "cap": None, "usage_unavailable": 1}})
    wire = []
    for i, code in enumerate((200, 200, 502)):
        wire.extend([{"event": "request", "event_id": str(i), "body_parameters": {"model": "deepseek-v4-flash", "thinking": {"type": "disabled"}}},
                     {"event": "response", "event_id": str(i), "http_status": code, "elapsed_seconds": 57.7 if code == 502 else 4.2}])
    (run / "actual-provider-http.jsonl").write_text("".join(json.dumps(row) + "\n" for row in wire))
    client = TestClient(create_app(project_root=tmp_path))
    return {"client": client, "root": tmp_path, "run": run, "digest": digest, "execution": execution, "stage": stage, "episode": episode}


def progress(monitor):
    response = monitor["client"].get("/runs/monitor/progress")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def test_failed_monitor_complete_fitness_partial_panel_and_active_state(monitor):
    p = progress(monitor)
    assert p["status"] == "failed" and p["generation"] == 1 and p["generations"] == 2
    assert p["stage"] == "Customer candidate evaluation"
    inc, evolver, candidate, service, replay = p["rows"]
    assert (inc["count"], inc["total"], inc["reused"], inc["successes"]) == (3, 3, 3, 2)
    assert inc["state"] == "REUSED" and inc["accuracy"] == pytest.approx(2 / 3)
    assert evolver["state"] == "COMPLETE" and evolver["condition"] == "C2 generated"
    assert candidate["count"] == 1 and candidate["state"] == "FAILED" and candidate["accuracy"] is None
    assert service["state"] == replay["state"] == "NOT_STARTED"
    assert p["active_customer"]["label"] == "C1"
    assert p["active_service"]["skills"][0]["skill_id"] == "skill-0001"
    assert p["current_episode"]["task_id"] == "29" and p["current_episode"]["turn"] == 7
    assert p["provider"]["calls"] == 3  # Live global ledger, not stale episode budget=1.
    assert p["provider"]["statuses"] == {"200": 2, "502": 1}
    assert p["provider"]["latest_call"]["http_status"] == 502
    assert p["provider"]["flash_disabled"] == p["provider"]["flash_requests"] == 3
    assert p["provider"]["peak_requests"] == 1
    assert p["provider"]["coverage_complete"]
    assert p["source_commit"] == "a" * 40 and p["source_parent"] == "parent"
    assert "sk-testsecret" not in json.dumps(p) and "H_SECRET_TASK" not in json.dumps(p)


def test_rendered_page_and_get_polling_do_not_mutate_core_or_launch(monitor, monkeypatch):
    reader = monitor["client"].app.state.reader
    manager = monitor["client"].app.state.run_manager
    def forbidden(*args, **kwargs):
        raise AssertionError("monitor must never launch or resume a run")
    monkeypatch.setattr(manager, "start", forbidden)
    watched = {f: f.read_bytes() for f in monitor["root"].rglob("*.json")}
    response = monitor["client"].get("/runs/monitor")
    assert response.status_code == 200
    for text in ("Experiment monitor", "Retail E3 · G2 · P1", "FAILED", "Latest provider call", "HTTP 502", "Service SkillMemory", "run-progress.js", "57.7s"):
        assert text in response.text
    progress(monitor); progress(monitor)
    assert all(f.read_bytes() == data for f, data in watched.items())
    assert reader.get_run("monitor", live_status="running")["status"] == "failed"


def test_external_running_and_pending_call_are_real_not_stale_200(monitor):
    execution = {**monitor["execution"], "status": "running", "failure": None}
    put(monitor["run"] / "run-execution-state.json", execution)
    (monitor["run"] / "episodes/current-attempt").mkdir()
    wire = monitor["run"] / "actual-provider-http.jsonl"
    with wire.open("a") as f:
        f.write(json.dumps({"event": "request", "event_id": "pending", "body_parameters": {"model": "deepseek-v4-flash", "thinking": {"type": "disabled"}}}) + "\n")
        f.write('{"event":"request"')  # A torn final append is retried on the next poll.
    p = progress(monitor)
    assert p["status"] == "running"
    assert p["rows"][2]["state"] == "RUNNING" and p["rows"][3]["state"] == "WAITING"
    assert p["provider"]["latest_call"]["state"] == "pending"
    assert p["provider"]["latest_call"]["http_status"] is None
    assert p["current_episode"]["attempt_id"] == "current-attempt"
    assert p["current_episode"]["turn"] is None and p["current_episode"]["task_id"] is None


def test_sparse_observations_stay_unavailable(monitor):
    (monitor["run"] / "actual-provider-http.jsonl").unlink()
    p = progress(monitor)
    assert p["provider"]["calls"] == 3
    assert p["provider"]["statuses"] == {}
    assert p["provider"]["latest_call"] is None
    assert p["provider"]["flash_requests"] is None
    assert p["provider"]["peak_requests"] is None
    assert not p["provider"]["coverage_complete"]


def test_no_op_skips_replay_and_keeps_active_memory(monitor):
    run, digest = monitor["run"], monitor["digest"]
    for task in ("92", "29"):
        monitor["episode"](task, "generation-1-customer-candidate-0", False)
    put(run / "generation-0001-stage.json", {"manifest_sha256": digest, "generation": 1, "stage": "service_proposal_ready"})
    put(run / "generation-0001-service-proposal.json", {"manifest_sha256": digest, "strategy_id": "s1", "frozen_service": {"strategy_id": "s1"}, "mutation": {"operation": "no_op"}})
    put(run / "run-execution-state.json", {**monitor["execution"], "status": "running", "failure": None})
    p = progress(monitor)
    assert p["rows"][-1]["state"] == "SKIPPED"
    assert p["rows"][-2]["state"] == "COMPLETE"
    assert p["active_service"]["label"] == "S1"


def test_sealed_heldout_never_reads_diagnostics_or_wire(monitor):
    run = monitor["run"]
    execution = {**monitor["execution"], "failure": {"stage": "heldout", "task_id": "H_SECRET_TASK", "diagnostics_ref": "H_SECRET_TASK/does-not-exist.json", "failure_message": "HIDDEN_H_TRAJECTORY"}}
    put(run / "run-execution-state.json", execution)
    (run / "actual-provider-http.jsonl").write_text("DO_NOT_READ_SEALED_WIRE\n")
    p = progress(monitor)
    assert p["current_episode"] is None and p["rows"] == []
    assert p["provider"]["latest_call"] is None
    assert "H_SECRET" not in json.dumps(p) and "HIDDEN_H" not in json.dumps(p)


@pytest.mark.parametrize("malformed", [
    {"event": "request", "event_id": "bad", "body_parameters": ["invalid"]},
    {"event": "response", "event_id": "bad", "elapsed_seconds": "invalid"},
    {"event": "response", "event_id": "bad", "elapsed_seconds": float("inf")},
])
def test_malformed_wire_values_fail_closed(monitor, malformed):
    with (monitor["run"] / "actual-provider-http.jsonl").open("a") as stream:
        stream.write(json.dumps(malformed) + "\n")
    assert monitor["client"].get("/runs/monitor/progress").status_code == 422


def test_stage_generation_and_checkpoint_manifest_must_match(monitor):
    path = monitor["run"] / "generation-0001-stage.json"
    put(path, {**monitor["stage"], "generation": 0})
    assert monitor["client"].get("/runs/monitor/progress").status_code == 422
    put(path, monitor["stage"])
    checkpoint = monitor["root"] / "experiments/checkpoints/monitor.json"
    value = json.loads(checkpoint.read_text()); value["manifest_sha256"] = "wrong"
    put(checkpoint, value)
    assert monitor["client"].get("/runs/monitor/progress").status_code == 422


def test_complete_monitor_stays_on_last_generation(monitor):
    checkpoint = json.loads((monitor["root"] / "experiments/checkpoints/monitor.json").read_text())
    last = {**checkpoint["generations"][0], "generation": 1}
    put(monitor["run"] / "alternating-result.json", {
        "schema_version": 2, "manifest_sha256": monitor["digest"], "experiment_id": "monitor", "status": "complete",
        "generations": [checkpoint["generations"][0], last],
    })
    put(monitor["run"] / "run-execution-state.json", {**monitor["execution"], "status": "complete", "failure": None})
    p = progress(monitor)
    assert p["status"] == "complete" and p["stage"] == "Experiment complete"
    assert p["generation_number"] == p["generations"] == p["completed_generations"] == 2


def test_initial_checkpoint_is_active_before_first_commit(monitor):
    path = monitor["root"] / "experiments/checkpoints/monitor.json"
    value = json.loads(path.read_text()); initial = value["generations"][0]
    value.update(generations=[], initial_customer=initial["customer_before"], initial_service=initial["service_before"])
    put(path, value)
    put(monitor["run"] / "run-execution-state.json", {**monitor["execution"], "status": "paused", "failure": None})
    p = progress(monitor)
    assert p["status"] == "paused" and p["generation"] == 0
    assert p["active_customer"] == {"label": "C0", "strategy_id": "c0"}
    assert p["active_service"] == {"label": "S0", "strategy_id": "s0", "skills": []}


@pytest.mark.parametrize("kind", ["stage", "proposal", "wire", "symlink"])
def test_corrupt_or_unsafe_monitoring_artifact_is_hidden(monitor, kind):
    run = monitor["run"]
    if kind in ("stage", "proposal"):
        name = "generation-0001-stage.json" if kind == "stage" else "generation-0001-customer-proposals.json"
        value = json.loads((run / name).read_text()); value["manifest_sha256"] = "wrong"; put(run / name, value)
    elif kind == "wire":
        (run / "actual-provider-http.jsonl").write_text('not JSON\n{}\n')
    else:
        (run / "actual-provider-http.jsonl").unlink()
        secret = monitor["root"] / "outside.jsonl"; secret.write_text('PRIVATE_OUTSIDE\n')
        (run / "actual-provider-http.jsonl").symlink_to(secret)
    r = monitor["client"].get("/runs/monitor/progress")
    assert r.status_code == 422 and "PRIVATE_OUTSIDE" not in r.text
