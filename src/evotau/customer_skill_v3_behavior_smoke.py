"""Tiny native τ-bench smoke comparing the V2 Customer baseline with one V3 skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .budget import RequestBudget
from .customer_skill_v3 import (
    CustomerSkill,
    extract_forbidden_literals,
    render_customer_skill_v3,
    validate_reflection_seed_signals,
    validate_skill,
)
from .manifest import ActivationSmokeManifest, sha256_json
from .native_runner import TauBenchEpisodeRunner, _provider_provenance_document
from .phase0 import load_config
from .phase0_run import _load_pinned_tasks, _write_json_once
from .phase3_run import load_provider_bundle
from .records import EpisodeRecord
from .strategies import CustomerStrategy, ServiceStrategy


def _project_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("behavior-smoke file path must stay inside the project")
    current = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError("behavior-smoke paths cannot traverse symlinks")
    resolved = current.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("behavior-smoke input must be a regular project file")
    return resolved


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path,
    provider_plugin: str,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    project_root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    raw = load_config(config_file)
    experiment = raw["experiment"]
    manifest = ActivationSmokeManifest.from_mapping(
        raw,
        task_review_document=json.loads(
            _project_file(project_root, str(experiment["task_review_path"])).read_text(encoding="utf-8")
        ),
    )
    if (manifest.condition != "customer_representation_comparison"
            or manifest.customer_evolver_schema != "skill_v3"):
        raise ValueError("V3 behavior smoke requires the frozen skill_v3 representation-comparison config")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required; inject it from Keychain for this command")

    proposal_path = _project_file(project_root, str(experiment["proposal_smoke_artifact"]))
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    if proposal.get("status") != "complete" or len(proposal.get("candidates", ())) != 2:
        raise ValueError("V3 behavior smoke requires a completed exact-K proposal artifact")
    selected = proposal["candidates"][0]
    if selected.get("validation") != "passed" or selected.get("task_literal_leakage_detected") is not False:
        raise ValueError("selected V3 skill did not pass the provider-only leakage checks")
    candidate = selected["candidate"]
    baseline_payload = experiment.get("customer_strategy")
    if not isinstance(baseline_payload, dict):
        raise TypeError("behavior smoke requires the matched V2 baseline")
    baseline = CustomerStrategy(**baseline_payload)
    if not baseline.is_v2:
        raise ValueError("behavior smoke baseline is not Customer Evolver V2")
    skill_value = candidate.get("strategy", {}).get("skill")
    if not isinstance(skill_value, dict):
        raise TypeError("proposal artifact contains no V3 skill object")
    skill = CustomerSkill(
        name=skill_value["name"], trigger=skill_value["trigger"],
        procedure=tuple(skill_value["procedure"]),
        stop_conditions=skill_value["stop_conditions"], hypothesis=skill_value["hypothesis"],
        evidence_refs=tuple(skill_value["evidence_refs"]),
    )
    data_root = Path(tau2_data_dir).expanduser().resolve()
    tasks = _load_pinned_tasks(
        manifest, data_dir=data_root,
        task_selection=experiment["task_selection"],
        task_ids=manifest.evolution_task_ids + manifest.validation_task_ids,
    )
    literals = extract_forbidden_literals(tasks)
    normalized = validate_skill(
        skill.to_dict()["skill"], incumbent=baseline, operation=candidate["operator"],
        verified_failure_ids=skill.evidence_refs, forbidden_literals=literals,
    )
    if normalized != skill or normalized.strategy_id != candidate.get("strategy_id"):
        raise ValueError("V3 behavior skill failed deterministic replay validation")
    _ = validate_reflection_seed_signals(json.loads(
        _project_file(project_root, str(experiment["reflection_seed_path"])).read_text(encoding="utf-8")
    ))
    if (sha256_json(validate_reflection_seed_signals(json.loads(
            _project_file(project_root, str(experiment["reflection_seed_path"])).read_text(encoding="utf-8")
        ))) != manifest.reflection_seed_sha256):
        raise ValueError("behavior-smoke reflection seed differs from the frozen manifest")
    expected_frozen = {
        "role_models": dict(manifest.role_models),
        "role_model_args": {role: dict(args) for role, args in manifest.role_model_args},
        "evolution_task_ids": list(manifest.evolution_task_ids),
        "validation_task_ids": list(manifest.validation_task_ids),
        "customer_strategy_sha256": manifest.customer_strategy_sha256,
        "service_strategy_sha256": manifest.service_strategy_sha256,
        "task_review_sha256": manifest.task_semantic_review_sha256,
        "reflection_seed_sha256": manifest.reflection_seed_sha256,
        "seed": manifest.seed,
        "customer_candidates": manifest.customer_candidates,
        "generations": manifest.generations,
        "max_episodes": manifest.max_episodes,
        "request_budget_cap": manifest.request_budget_cap,
        "max_concurrency": manifest.max_concurrency,
        "provider_retries": manifest.provider_retries,
    }
    if proposal.get("comparison_frozen_inputs") != expected_frozen:
        raise ValueError("provider-only V3 proposal smoke did not use the behavior-smoke frozen comparison inputs")

    bundle = load_provider_bundle(
        provider_plugin, config=raw, manifest=manifest, allow_frozen_service=True,
    )
    output_dir = (project_root / manifest.output_path).resolve()
    if not output_dir.is_relative_to(project_root):
        raise ValueError("behavior-smoke output path escapes the project")
    context = {
        "schema_version": 1,
        "activation_manifest_sha256": manifest.sha256,
        "proposal_artifact_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
        "customer_skill_id": skill.strategy_id,
        "service_strategy_sha256": manifest.service_strategy_sha256,
        "task_review_sha256": manifest.task_semantic_review_sha256,
        "provider_provenance": _provider_provenance_document(
            bundle.provenance, {"audit_provider": bundle.callbacks["audit_provider"]},
        ),
        "behavior_smoke_task_id": "22",
        "seed": 1,
        "paired_baseline_and_skill": True,
        "service_frozen": True,
    }
    context_path = output_dir / "run-context.json"
    if (context_path.exists()
            and json.loads(context_path.read_text(encoding="utf-8")) != context):
        raise ValueError("existing behavior-smoke run context differs from its frozen inputs")
    budget = RequestBudget(cap=manifest.request_budget_cap)
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=raw,
        data_dir=data_root,
        request_budget=budget,
        audit_provider=bundle.callbacks["audit_provider"],
        output_directory=output_dir,
    )
    if not context_path.exists():
        _write_json_once(context_path, context)
    service = ServiceStrategy()
    episodes = []
    for strategy, label in ((baseline, "behavior-smoke-v2-baseline"), (skill, "behavior-smoke-v3-skill")):
        episodes.append(runner(
            task_id="22", seed=1, customer=strategy, service=service, panel_name=label,
        ))

    def telemetry_for(record: EpisodeRecord) -> dict[str, Any]:
        for path in (output_dir / "episodes").glob("*/episode-record.json"):
            saved = EpisodeRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if saved.episode_id == record.episode_id:
                return json.loads((path.parent / "run-telemetry.json").read_text(encoding="utf-8"))
        raise FileNotFoundError("native EpisodeRecord has no matching prompt telemetry")

    telemetry = [telemetry_for(record) for record in episodes]
    turn_hashes = []
    customer_turn_counts = []
    for info in telemetry:
        # Episode folder names are random attempt IDs; find the exact telemetry row.
        matching_path = next(
            path for path in (output_dir / "episodes").glob("*/run-telemetry.json")
            if json.loads(path.read_text(encoding="utf-8"))["episode_key_sha256"] == info["episode_key_sha256"]
        )
        simulation = json.loads((matching_path.parent / "native-simulation.json").read_text(encoding="utf-8"))
        customer_turns = [
            message.get("content", "") for message in simulation.get("messages", ())
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        customer_turn_counts.append(len(customer_turns))
        turn_hashes.append(hashlib.sha256(
            json.dumps(customer_turns, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest())
        prompt_hashes = info["rendered_prompt_sha256"]
        if prompt_hashes.get("customer_scenario_source") != prompt_hashes.get("customer_scenario_runtime"):
            raise ValueError("native Customer scenario differs from the reviewed task scenario")
        if not prompt_hashes.get("customer_native_system_prompt"):
            raise ValueError("native τ-bench Customer prompt integrity was not verified")

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": manifest.experiment_id,
        "behavior_smoke_task": "22",
        "seed": 1,
        "service_frozen": True,
        "customer_skill_id": skill.strategy_id,
        "rendered_skill_sha256": hashlib.sha256(render_customer_skill_v3(skill).encode("utf-8")).hexdigest(),
        "episodes": [
            {
                "arm": label,
                "episode": record.to_dict(),
                "prompt_hashes": telemetry[index]["rendered_prompt_sha256"],
                "customer_turn_sha256": turn_hashes[index],
                "customer_turn_count": customer_turn_counts[index],
            }
            for index, (label, record) in enumerate(zip(
                ("V2-baseline", "V3-skill"), episodes, strict=True,
            ))
        ],
        "customer_utterances_changed": turn_hashes[0] != turn_hashes[1],
        "customer_system_prompt_changed": (
            telemetry[0]["rendered_prompt_sha256"].get("customer")
            != telemetry[1]["rendered_prompt_sha256"].get("customer")
        ),
        "skill_applicable": episodes[1].strategy_applicable,
        "skill_adherent": episodes[1].customer_strategy_adherent,
        "customer_valid": episodes[1].customer_valid,
        "customer_invalidity_category": episodes[1].customer_invalidity_category,
        "native_scenario_preserved": all(
            item["rendered_prompt_sha256"].get("customer_scenario_source")
            == item["rendered_prompt_sha256"].get("customer_scenario_runtime") for item in telemetry
        ),
        "native_system_prompt_extension_verified": all(
            bool(item["rendered_prompt_sha256"].get("customer_native_system_prompt"))
            for item in telemetry
        ),
        "provider_budget": budget.snapshot().to_dict(),
        "run_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
    }
    result_path = output_dir / "behavior-smoke-result.json"
    if result_path.exists():
        raise FileExistsError(f"behavior-smoke result already exists: {result_path}")
    _write_json_once(result_path, result)
    return {**result, "artifact_path": str(result_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--provider-plugin", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_from_config(
            args.config, tau2_data_dir=args.tau2_data_dir,
            provider_plugin=args.provider_plugin,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"V3 behavior smoke failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - provider failures must not expose credentials
        print(f"V3 behavior smoke failed ({type(exc).__name__}); inspect episode artifacts", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
