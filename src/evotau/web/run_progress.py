"""Read-only, bounded monitoring of alternating-run artifacts.

This projection never launches/resumes a run or supplies evidence to evolution.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from .artifact_reader import ArtifactReader, ArtifactReadError, redact_secrets

_STAGE_NAMES = {
    "customer_incumbent_complete": "Customer Evolver",
    "customer_proposals_ready": "Customer candidate evaluation",
    "customer_candidates_complete": "Customer selection",
    "customer_selected": "Service Evolver",
    "service_proposal_ready": "Candidate Service evaluation",
    "service_E_complete": "Service acceptance / Validation",
    "validation_complete": "Generation checkpoint",
    "generation_complete": "Generation complete",
}
_STATUS_NAMES = {
    "running": "运行中", "failed": "已停止 · 失败", "paused": "已暂停",
    "complete": "已完成", "incomplete": "未完成", "not_started": "尚未开始",
}


def _optional(reader: ArtifactReader, root: Any, name: str) -> dict[str, Any] | None:
    path = reader._contained_path(root, name, require_exists=False)
    return reader._read_json(path) if path.exists() else None


def _bound(document: dict[str, Any] | None, digest: str) -> dict[str, Any] | None:
    if document is not None and document.get("manifest_sha256") != digest:
        raise ArtifactReadError("monitoring document differs from its frozen manifest")
    return document


def _jsonl(reader: ArtifactReader, root: Any, name: str) -> list[dict[str, Any]]:
    path = reader._contained_path(root, name, require_exists=False)
    if not path.exists():
        return []
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ArtifactReadError("provider observation file is unsafe or too large")
    raw = path.read_text(encoding="utf-8")
    rows = []
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                break  # One append may still be in flight; retry on next refresh.
            raise ArtifactReadError("provider observation file is malformed") from exc
        if not isinstance(value, dict):
            raise ArtifactReadError("provider observation row is malformed")
        rows.append(value)
    return rows


def _provider_health(rows: list[dict[str, Any]], budget: dict[str, Any]) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}
    settled: dict[str, dict[str, Any]] = {}
    active: set[str] = set()
    statuses: Counter[str] = Counter()
    flash = disabled = peak = 0
    for row in rows:
        event, key = row.get("event"), row.get("event_id")
        if not isinstance(key, str):
            continue
        if event == "request":
            requests[key] = row
            active.add(key)
            peak = max(peak, len(active))
            args = row.get("body_parameters") or {}
            if not isinstance(args, dict) or not isinstance(args.get("thinking") or {}, dict):
                raise ArtifactReadError("provider request parameters are malformed")
            if str(args.get("model", "")).removeprefix("openai/") == "deepseek-v4-flash":
                flash += 1
                disabled += (args.get("thinking") or {}).get("type") == "disabled"
        elif event in {"response", "transport_failure"}:
            elapsed = row.get("elapsed_seconds")
            if elapsed is not None and (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
                raise ArtifactReadError("provider latency is malformed")
            active.discard(key)
            settled[key] = row
            statuses[str(row.get("http_status", "transport error"))] += 1
    latest = None
    if requests:
        key = next(reversed(requests))
        req, response = requests[key], settled.get(key)
        latest = {
            "state": "pending" if response is None else "settled",
            "model": (req.get("body_parameters") or {}).get("model"),
            "http_status": None if response is None else response.get("http_status"),
            "elapsed_seconds": None if response is None else response.get("elapsed_seconds"),
            "at": req.get("at") if response is None else response.get("at"),
            "transport_error": response is not None and response.get("event") == "transport_failure",
        }
    calls = budget.get("attempts")
    observed = len(requests)
    return {
        "calls": calls, "statuses": dict(statuses), "observed_requests": observed,
        "coverage_complete": bool(rows) and observed == calls,
        "flash_requests": flash if rows else None,
        "flash_disabled": disabled if rows else None,
        "peak_requests": peak if rows else None, "latest_call": latest,
    }


def _version(generations: list[dict[str, Any]], side: str) -> str:
    version = 0
    for item in generations:
        before, after = item.get(f"{side}_before") or {}, item.get(f"{side}_after") or {}
        if before.get("strategy_id") != after.get("strategy_id"):
            version = int(item["generation"]) + 1
    return ("C" if side == "customer" else "S") + str(version)


def read_run_progress(reader: ArtifactReader, run: dict[str, Any]) -> dict[str, Any] | None:
    if run["phase"] != "alternating-self-evolution":
        return None
    root, manifest, digest = run["path"], run["manifest"], run["manifest_sha256"]
    checkpoint = _bound(reader._load_checkpoint(manifest), digest) or {}
    generations = run["generation_commits"]
    planned = manifest.get("generations")
    planned = planned if type(planned) is int and planned > 0 else None
    completed = len(generations)
    evolution_finished = bool(generations) and (run["status"] == "complete" or (planned is not None and completed >= planned))
    generation = max(0, completed - 1) if evolution_finished else completed
    stage = None if evolution_finished else _bound(
        _optional(reader, root, f"generation-{generation:04d}-stage.json"), digest,
    )
    if stage is not None and stage.get("generation") != generation:
        raise ArtifactReadError("monitoring stage differs from its generation")
    failure = run.get("execution_failure") or {}
    # Do not project sealed H task identities, traces or provider observations.
    execution = _bound(_optional(reader, root, "run-execution-state.json"), digest)
    execution_stage = ((execution or {}).get("failure") or {}).get("stage")
    hidden_phase = run["heldout_sealed"] and (execution_stage in {"heldout", "fresh_customer"} or (evolution_finished and run["status"] != "complete"))
    stage_name = (stage or {}).get("stage")
    stage_label = _STAGE_NAMES.get(stage_name, "Customer incumbent evaluation")
    if run["status"] == "complete":
        stage_label = "Experiment complete"
    elif hidden_phase:
        stage_label = "Evolution complete · Heldout sealed"
    panel_ids = manifest.get("task_panels", {}).get("E", ())
    if not panel_ids:
        panel_ids = manifest.get("task_selection", {}).get("evolution", ())
    task_ids = {str(value) for value in panel_ids}
    total = len(task_ids)
    refs_doc = _bound(_optional(reader, root, "episode-panel-references.json"), digest)
    refs = [] if refs_doc is None else list(refs_doc.get("references", {}).values())
    by_id = {episode["episode_id"]: episode for episode in run["episodes"]}
    active_c, active_s = _version(generations, "customer"), _version(generations, "service")
    # Completed rows describe the last committed generation's before/after states.
    committed = generations[-1] if evolution_finished else None
    before_c = _version(generations[:-1], "customer") if committed else active_c
    before_s = _version(generations[:-1], "service") if committed else active_s
    customer_phase = (committed or {}).get("customer_phase") or {}
    service_phase = (committed or {}).get("service_phase") or {}
    proposal = None if committed or hidden_phase else _bound(
        _optional(reader, root, f"generation-{generation:04d}-customer-proposals.json"), digest,
    )
    service_proposal = None if committed or hidden_phase else _bound(
        _optional(reader, root, f"generation-{generation:04d}-service-proposal.json"), digest,
    )
    candidates = customer_phase.get("candidates", ()) if committed else (proposal or {}).get("customer_candidates", ())
    pending = "WAITING" if run["status"] == "running" else "NOT_STARTED"

    def panel(label: str, name: str, condition: str, *, waiting: bool = False) -> dict[str, Any]:
        panel_refs = [ref for ref in refs if ref.get("panel_name") == name and str(ref.get("task_id")) in task_ids]
        records: dict[str, dict[str, Any]] = {}
        reused = 0
        for ref in panel_refs:
            record = by_id.get(ref.get("episode_id"))
            if record is None or record["status"] != "complete" or type(record["task_success"]) is not bool:
                continue
            task = str(ref["task_id"])
            if task in records:
                raise ArtifactReadError("panel has duplicate task observations")
            records[task] = record
            reused += ref.get("reused") is True
        # Legacy artifacts may not have a reference index. Keep their real panel labels.
        if refs_doc is None:
            records = {str(ep["task_id"]): ep for ep in run["episodes"] if ep.get("panel_name") == name and str(ep["task_id"]) in task_ids and ep["status"] == "complete" and type(ep["task_success"]) is bool}
        count = len(records)
        full = total > 0 and count == total
        state = "REUSED" if full and reused == total else "COMPLETE" if full else pending if waiting and not count else "RUNNING" if run["status"] == "running" else "STOPPED"
        if failure.get("panel_name") == name:
            state = "FAILED"
        return {"label": label, "condition": condition, "count": count, "total": total,
                "reused": reused, "state": state,
                "successes": sum(ep["task_success"] for ep in records.values()) if full else None,
                "accuracy": sum(ep["task_success"] for ep in records.values()) / total if full else None}

    rows = [panel("Incumbent", f"generation-{generation}-customer-incumbent", f"{before_c} × {before_s}")]
    rows.append({"label": "Customer Evolver", "condition": f"C{generation + 1} generated" if candidates else "", "state": "COMPLETE" if candidates else "RUNNING" if stage_name == "customer_incumbent_complete" and run["status"] == "running" else pending})
    for index, candidate in enumerate(candidates):
        rows.append(panel("Customer candidate" + (f" {index}" if len(candidates) > 1 else ""), f"generation-{generation}-customer-candidate-{index}", f"C{generation + 1} × {before_s}"))
    if not candidates:
        rows.append({"label": "Customer candidate", "condition": "", "state": pending})
    service_ready = bool(service_proposal or committed)
    rows.append({"label": "Service Evolver", "condition": "", "state": "COMPLETE" if service_ready else "RUNNING" if stage_name == "customer_selected" and run["status"] == "running" else pending})
    operation = service_phase.get("operation") if committed else ((service_proposal or {}).get("mutation") or {}).get("operation")
    changed = not service_proposal or service_proposal.get("strategy_id") != (service_proposal.get("frozen_service") or {}).get("strategy_id")
    if operation == "no_op" or (service_proposal and not changed):
        rows.append({"label": "Candidate Service", "condition": "NO_OP · replay skipped", "state": "SKIPPED"})
    else:
        selected_c = before_c if customer_phase.get("selected_customer") == "incumbent" else f"C{generation + 1}"
        if stage and stage.get("selected_customer") == "incumbent":
            selected_c = before_c
        rows.append(panel("Candidate Service", f"generation-{generation}-service-candidate", f"{selected_c} × S{generation + 1}" if service_ready else "", waiting=not service_ready))
    failed_episode = None
    if failure.get("diagnostics_ref") and not hidden_phase:
        diagnostic = _optional(reader, root, str(failure["diagnostics_ref"]))
        if diagnostic and str(diagnostic.get("task_id")) in task_ids:
            failed_episode = by_id.get(str(diagnostic.get("attempt_id")))
    current = None
    if failed_episode:
        native_messages = failed_episode["trajectory"].get("messages", [])
        turn = next((msg.get("turn_idx") for msg in reversed(native_messages) if type(msg.get("turn_idx")) is int), None)
        current = {"label": "失败现场", "task_id": failed_episode["task_id"],
                   "episode_id": failed_episode["episode_id"], "panel_name": failed_episode["panel_name"],
                   "turn": turn, "message_count": len(native_messages),
                   "failure_message": failed_episode.get("failure_message")}
    elif run["status"] == "running":
        # Native runner persists full trajectories only at completion/failure.
        active = reader._active_episode(root, "running", False)
        if active:
            current = {"label": "当前 episode", "attempt_id": active["attempt_id"],
                       "task_id": None, "panel_name": None, "turn": None, "message_count": None}
    health = _provider_health([] if hidden_phase else _jsonl(reader, root, "actual-provider-http.jsonl"), run["budget"])
    origin = _optional(reader, root, "continuation-provenance.json")
    initial_c = checkpoint.get("initial_customer") or {}
    initial_s = checkpoint.get("initial_service") or {}
    initial_strategies = reader._load_strategies({"initial_service": initial_s}, [])
    service = run.get("current_service_strategy") or initial_strategies["service"].get(str(initial_s.get("strategy_id"))) or {}
    skills = service.get("skills", ())
    result = {
        "status": run["status"], "status_label": _STATUS_NAMES.get(run["status"], run["status"]),
        "domain": manifest.get("domain", "retail").title(), "e_size": total,
        "generations": planned, "generation": generation,
        "generation_number": generation + 1, "completed_generations": completed,
        "parallelism": run["max_concurrency"], "stage": stage_label,
        "rows": [] if hidden_phase else rows, "current_episode": current,
        "source_commit": (manifest.get("evotau") or {}).get("git_commit"),
        "source_parent": None if origin is None else origin.get("source_experiment_id"),
        "imported_episodes": None if origin is None else origin.get("imported_complete_episodes"),
        "active_customer": {"label": active_c, "strategy_id": run.get("current_customer_id") or initial_c.get("strategy_id")},
        "active_service": {"label": active_s, "strategy_id": run.get("current_service_id") or initial_s.get("strategy_id"), "skills": skills},
        "provider": health, "failure_message": failure.get("failure_message"),
        "as_of": datetime.now(UTC).isoformat(),
    }
    return redact_secrets(result)
