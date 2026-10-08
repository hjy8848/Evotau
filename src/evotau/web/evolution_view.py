"""Artifact-only V2 projections. No metric inference, provider calls or Console statistics."""

from ..tau_provenance import sha256_json
from .artifact_reader import ArtifactReadError, redact_secrets


def evolution_view(reader, run):
    policy = run["manifest"].get("skill_evolution_v2")
    generations = run["generation_commits"]
    boards = [
        candidate
        for g in generations
        for candidate in g.get("service_phase", {}).get("candidates", [])
    ]
    matrix = [cell for g in generations for cell in g.get("failure_matrix", [])]
    allowed = set(run["manifest"].get("task_panels", {}).get("E", [])) | set(
        run["manifest"].get("task_panels", {}).get("V", [])
    )

    def verify_evidence(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "task_id" and str(item) not in allowed:
                    raise ArtifactReadError("V2 evidence contains out-of-panel task ID")
                if (
                    key
                    in (
                        "evidence_task_ids",
                        "protected_success_task_ids",
                        "expected_fixes",
                        "protected_cases_at_risk",
                        "fail_to_pass",
                        "pass_to_fail",
                        "fail_to_fail",
                        "pass_to_pass",
                        "source_task_ids",
                        "known_fixes",
                        "known_regressions",
                    )
                    and isinstance(item, (list, tuple))
                    and any(str(task) not in allowed for task in item)
                ):
                    raise ArtifactReadError(
                        "V2 evidence contains sealed/out-of-panel task IDs"
                    )
                verify_evidence(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                verify_evidence(item)

    if policy:
        verify_evidence(generations)
    for cell in matrix:
        if str(cell.get("task_id")) not in allowed:
            raise ArtifactReadError(
                "evolution evidence references sealed/out-of-panel content"
            )
    state = {}
    checkpoint = reader._load_checkpoint(run["manifest"])
    if checkpoint and checkpoint.get("schema_version") == 3:
        if checkpoint.get("checkpoint_sha256") != sha256_json(
            {k: v for k, v in checkpoint.items() if k != "checkpoint_sha256"}
        ):
            raise ArtifactReadError("V2 checkpoint digest mismatch")
        state = checkpoint.get("state", {})
        verify_evidence(state)
    journal_root = run["path"] / "evolution-v2"
    live_artifacts = []
    if policy and journal_root.exists():
        for path in sorted(journal_root.glob("*.json")):
            doc = reader._read_json(
                reader._contained_path(
                    run["path"], path.relative_to(run["path"]).as_posix()
                )
            )
            if (
                doc.get("manifest_sha256") != run["manifest_sha256"]
                or doc.get("payload_sha256") != sha256_json(doc.get("payload"))
                or doc.get("envelope_sha256")
                != sha256_json({k: v for k, v in doc.items() if k != "envelope_sha256"})
            ):
                raise ArtifactReadError("V2 stage digest mismatch")
            verify_evidence(doc["payload"])
            if doc["stage"].endswith("generation_complete"):
                generation = doc["payload"]["generation"]
                if not any(
                    sha256_json(g) == sha256_json(generation) for g in generations
                ):
                    raise ArtifactReadError(
                        "V2 generation differs from immutable committed journal"
                    )
                state = doc["payload"]["state"]
            # A live board shows frozen raw proposals/screens/gates before generation commit.
            if any(
                label in doc["stage"]
                for label in (
                    "service_proposals",
                    "service_screen",
                    "service_full_gate",
                    "service_diagnosis",  # Historical read-only artifact compatibility.
                    "service_crossover",
                )
            ):
                live_artifacts.append(
                    {"stage": doc["stage"], "evidence": doc["payload"]}
                )
    latest = generations[-1] if generations else {}
    archive = state.get("evolution_archive", {}).get("entries", [])
    customers = state.get("customer_archive", [])
    active = run.get("current_service_strategy") or state.get("service") or {}
    if not isinstance(active, dict):
        active = {}
    active_ids = {skill["skill_id"] for skill in active.get("skills", [])}
    activation_summary = state.get(
        "activation_summary", latest.get("activation_summary", {})
    )
    skill_stats = activation_summary.get("skills", {})
    return redact_secrets(
        {
            "available": bool(policy),
            "policy": policy,
            "activation_mode": policy["service_skill_runtime"]
            if policy
            else "render-all / legacy",
            "mechanism_smoke": not run["manifest"].get("run_validation", True),
            "generations": generations,
            "candidates": boards,
            "failure_matrix": matrix,
            "archive": archive,
            "customer_archive": customers,
            "runtime_active_skill_ids": sorted(active_ids),
            "activation_summary": activation_summary,
            "skill_stats": skill_stats,
            "service_provenance": state.get("service_provenance", []),
            "history_summary": state.get("history_summary", []),
            "live_artifacts": live_artifacts,
            "health": latest.get("evolution_health", {}),
            "statistical_gate_state": state.get("statistical_gate_state", {}),
            "api_usage": (run.get("result") or {}).get("api_usage_by_role", {}),
        }
    )
