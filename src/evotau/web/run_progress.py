"""Read-only, bounded monitoring of alternating-run artifacts.

This projection never launches/resumes a run or supplies evidence to evolution.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from evotau.tau_provenance import capture_code_provenance

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


def _strategy_text(candidate: dict[str, Any]) -> str:
    value = candidate.get("strategy", candidate.get("text", ""))
    if isinstance(value, dict):
        value = value.get("text", "")
    if not isinstance(value, str):
        raise ArtifactReadError("Customer proposal text is malformed")
    return value


def _live_episodes(reader: ArtifactReader, root: Any, digest: str, allowed: set[str]) -> list[dict[str, Any]]:
    values = []
    for path in sorted((root / "episodes").glob("*/active-episode.json")):
        doc = _bound(_optional(reader, root, path.relative_to(root).as_posix()), digest)
        if not doc or doc.get("status") != "running" or str(doc.get("task_id")) not in allowed:
            continue
        if any((path.parent / name).exists() for name in ("episode-record.json", "incomplete-run.json")):
            continue  # Terminal research evidence overrides an observer's stale write.
        if doc.get("attempt_id") != path.parent.name or type(doc.get("turn_index")) not in (int, type(None)):
            raise ArtifactReadError("live episode identity or turn is inconsistent")
        if (doc.get("role") not in (None, "user", "assistant", "tool")
                or doc.get("next_role") not in (None, "user", "agent", "env")
                or any(value is not None and not isinstance(value, str) for value in
                       (doc.get("tool_name"), doc.get("panel_name"), doc.get("activity"), doc.get("updated_at")))):
            raise ArtifactReadError("live episode metadata is malformed")
        # Whitelist even if a malformed observer writes additional hidden fields.
        values.append({"label": "当前 episode", "attempt_id": doc["attempt_id"],
                       "task_id": str(doc["task_id"]), "panel_name": doc.get("panel_name"),
                       "turn": doc.get("turn_index"), "role": doc.get("role"),
                       "tool_name": doc.get("tool_name"), "next_role": doc.get("next_role"),
                       "activity": doc.get("activity"), "updated_at": doc.get("updated_at")})
    return sorted(values, key=lambda value: value.get("updated_at") or "", reverse=True)


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
    for document in (proposal, service_proposal):
        if document is not None and document.get("generation", generation) != generation:
            raise ArtifactReadError("monitoring proposal differs from its generation")
    candidates = customer_phase.get("candidates", ()) if committed else (proposal or {}).get("customer_candidates", ())
    pending = "WAITING" if run["status"] == "running" else "NOT_STARTED"
    panel_records: dict[str, dict[str, dict[str, Any]]] = {}

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
        panel_records[name] = records
        full = total > 0 and count == total
        state = "REUSED" if full and reused == total else "COMPLETE" if full else pending if waiting and not count else "RUNNING" if run["status"] == "running" else "STOPPED"
        if failure.get("panel_name") == name:
            state = "FAILED"
        return {"label": label, "condition": condition, "count": count, "total": total,
                "reused": reused, "state": state,
                "observed_successes": sum(ep["task_success"] for ep in records.values()),
                "observed_accuracy": sum(ep["task_success"] for ep in records.values()) / count if count else None,
                "provisional": not full and count > 0,
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
    diagnostic = None
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
                   "role": native_messages[-1].get("role") if native_messages else None,
                   "tool_name": None, "activity": "native_episode_failed",
                   "updated_at": (execution or {}).get("last_completed_at"),
                   "failure_message": failed_episode.get("failure_message")}
    live_episodes = []
    if run["status"] == "running" and not hidden_phase:
        allowed = task_ids | {str(value) for value in manifest.get("task_panels", {}).get("V", ())}
        live_episodes = _live_episodes(reader, root, digest, allowed)
    if live_episodes:
        current = live_episodes[0]
    elif current is None and run["status"] == "running" and not hidden_phase:
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
    prefix = f"generation-{generation}-"

    def compare(label: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        paired_tasks = [str(task) for task in panel_ids if str(task) in before and str(task) in after]
        transitions = {name: [] for name in ("P→F", "F→P", "P→P", "F→F")}
        for task in paired_tasks:
            if before[task].get("seed") != after[task].get("seed"):
                raise ArtifactReadError("paired monitor observations have different seeds")
            name = ("P" if before[task]["task_success"] else "F") + "→" + ("P" if after[task]["task_success"] else "F")
            transitions[name].append(task)
        return {"label": label, "paired_count": len(paired_tasks), "total": total,
                "provisional": len(paired_tasks) != total, "task_ids": transitions,
                "counts": {name: len(tasks) for name, tasks in transitions.items()}}

    def paired(label: str, before_name: str, after_name: str) -> dict[str, Any]:
        return compare(label, panel_records.get(before_name, {}), panel_records.get(after_name, {}))

    pairs = [paired(f"Customer candidate {index}", prefix + "customer-incumbent", prefix + f"customer-candidate-{index}") for index in range(len(candidates))]
    selected = customer_phase.get("selected_customer") if committed else (stage or {}).get("selected_customer")
    selected_panel = prefix + "customer-incumbent" if selected == "incumbent" else prefix + f"customer-candidate-{selected}" if type(selected) is int else None
    if selected_panel and operation != "no_op" and changed:
        pairs.append(paired("Service repair", selected_panel, prefix + "service-candidate"))
    def referenced_records(references: list[dict[str, Any]]) -> dict[str, Any]:
        records = {}
        for ref in references:
            ep = by_id.get(ref.get("episode_id"))
            if ep is None or str(ref.get("task_id")) not in task_ids or ep["status"] != "complete":
                continue
            if (ep["task_id"] != str(ref["task_id"]) or ep["seed"] != ref.get("seed")
                    or type(ep["task_success"]) is not bool or ep["task_id"] in records):
                raise ArtifactReadError("committed paired observations differ from native episodes")
            records[ep["task_id"]] = ep
        return records

    for previous in generations:
        if committed and previous["generation"] == generation:
            continue
        c_phase, s_phase = previous.get("customer_phase", {}), previous.get("service_phase", {})
        before = referenced_records(c_phase.get("incumbent_episodes", []))
        candidate_records = [referenced_records(value.get("episodes", [])) for value in c_phase.get("candidates", ())]
        for index, after in enumerate(candidate_records):
            pairs.append(compare(f"Gen {previous['generation']} Customer candidate {index}", before, after))
        chosen = c_phase.get("selected_customer")
        if s_phase.get("operation") != "no_op" and (chosen == "incumbent" or type(chosen) is int and 0 <= chosen < len(candidate_records)):
            selected_records = before if chosen == "incumbent" else candidate_records[chosen]
            pairs.append(compare(f"Gen {previous['generation']} Service repair", selected_records, referenced_records(s_phase.get("challenge_episodes", []))))
    mutation = service_phase if committed else (service_proposal or {}).get("mutation") or {}
    pending_service = None
    if service_ready and mutation.get("operation"):
        proposed = service_phase.get("proposed_service_memory", {}) if committed else (service_proposal or {}).get("proposed_service_memory", {})
        target = mutation.get("target_skill_id")
        skill = mutation.get("skill") or next((value for value in proposed.get("skills", ()) if value.get("skill_id") == target), None)
        if skill is None and mutation.get("operation") == "add":
            existing = {value.get("skill_id") for value in service_phase.get("input_service_memory", {}).get("skills", ())} if committed else {value.get("skill_id") for value in skills}
            skill = next((value for value in proposed.get("skills", ()) if value.get("skill_id") not in existing), None)
        pending_service = {"operation": mutation["operation"].upper(), "target_skill_id": target,
                           "skill": skill, "analysis": mutation.get("analysis", service_phase.get("analysis", "")),
                           "committed": bool(committed), "accepted": service_phase.get("accepted") if committed else None}
    imported_paths = {} if origin is None else origin.get("imported_file_sha256", {})
    imported_attempts = {name.split("/")[1] for name in imported_paths if name.startswith("episodes/")}
    visible_complete = [ep for ep in run["episodes"] if ep["status"] == "complete"]
    visible_incomplete = [ep for ep in run["episodes"] if ep["status"] != "complete"]
    def attempt_id(ep: dict[str, Any]) -> str:
        parts = ((ep.get("raw_record") or {}).get("trajectory_ref") or "").split("/")
        return parts[1] if len(parts) > 1 and parts[0] == "episodes" else ep["episode_id"]

    imported = sum(attempt_id(ep) in imported_attempts for ep in visible_complete) if imported_paths else None if origin is None else origin.get("imported_complete_episodes")
    new_complete = sum(attempt_id(ep) not in imported_attempts for ep in visible_complete) if imported_paths else len(visible_complete) if origin is None else None
    failure_card = None
    if failure and not hidden_phase:
        source = manifest.get("evotau") or {}
        actual = capture_code_provenance(runtime_only=source.get("source_scope") == "runtime-v2")
        matches = source.get("source_sha256") == actual.source_sha256
        failure_card = {"stage": failure.get("stage"), "type": (diagnostic or {}).get("failure_type", failure.get("failure_type")),
                        "task_id": failure.get("task_id"), "panel_name": failure.get("panel_name"),
                        "message": failure.get("failure_message"), "runtime_matches": matches,
                        "resume_safe": "当前源码匹配；需显式恢复，完成条件复用，失败条件不计 fitness" if matches else "需使用冻结 runtime 或修复后创建新条件；不要用当前源码直接恢复",
                        "checkpoint_preserved": bool(checkpoint)}
    result = {
        "status": run["status"], "status_label": _STATUS_NAMES.get(run["status"], run["status"]),
        "domain": manifest.get("domain", "retail").title(), "e_size": total,
        "generations": planned, "generation": generation,
        "generation_number": generation + 1, "completed_generations": completed,
        "parallelism": run["max_concurrency"], "stage": stage_label,
        "rows": [] if hidden_phase else rows, "current_episode": current,
        "active_episodes": live_episodes,
        "customer_proposals": [] if hidden_phase else [{"strategy_id": candidate.get("strategy_id"), "text": _strategy_text(candidate), "index": index, "committed": bool(committed)} for index, candidate in enumerate(candidates)],
        "service_mutation": None if hidden_phase else pending_service,
        "paired_transitions": [] if hidden_phase else pairs,
        "continuation": {"parent": None if origin is None else origin.get("source_experiment_id"),
                         "imported_complete": imported, "reused_references": sum(ref.get("reused") is True for ref in refs if str(ref.get("task_id")) in task_ids),
                         "new_complete": new_complete, "incomplete_attempts": len(visible_incomplete),
                         "imported_incomplete": sum(attempt_id(ep) in imported_attempts for ep in visible_incomplete) if imported_paths or origin is None else None,
                         "new_incomplete": sum(attempt_id(ep) not in imported_attempts for ep in visible_incomplete) if imported_paths or origin is None else None},
        "failure": failure_card,
        "source_commit": (manifest.get("evotau") or {}).get("git_commit"),
        "source_parent": None if origin is None else origin.get("source_experiment_id"),
        "imported_episodes": None if origin is None else origin.get("imported_complete_episodes"),
        "active_customer": {"label": active_c, "strategy_id": run.get("current_customer_id") or initial_c.get("strategy_id")},
        "active_service": {"label": active_s, "strategy_id": run.get("current_service_id") or initial_s.get("strategy_id"), "skills": skills},
        "provider": health, "failure_message": failure.get("failure_message"),
        "as_of": datetime.now(UTC).isoformat(),
    }
    return redact_secrets(result)
