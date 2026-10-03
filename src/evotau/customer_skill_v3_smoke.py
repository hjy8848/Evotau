"""Provider-only smoke for the bounded open-ended Customer skill Evolver v3."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .budget import RequestBudget
from .customer_skill_v3 import (
    CustomerSkillProviderResponse,
    LLMCustomerSkillEvolver,
    extract_forbidden_literals,
    propose_customer_skills,
    sanitized_reflection_context,
    validate_reflection_seed_signals,
)
from .manifest import (
    ActivationSmokeManifest,
    capture_code_provenance,
    role_model_args_for_runtime,
    sha256_json,
)
from .phase0 import load_config
from .phase0_run import _load_pinned_tasks
from .records import CandidateEvaluation, customer_strategy_id
from .strategies import CustomerStrategy


def _resolve_project_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (relative_path.is_absolute() or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).is_absolute()
            or ".." in relative_path.parts):
        raise ValueError("V3 proposal-smoke input paths must stay inside the project")
    current = root
    for part in relative_path.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError("V3 proposal-smoke input paths cannot traverse symlinks")
    resolved = current.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("V3 proposal-smoke input must be a regular project file")
    return resolved


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def run_from_config(
    config_path: str | Path,
    *,
    tau2_data_dir: str | Path,
    proposal_provider: Any | None = None,
) -> dict[str, Any]:
    """Make one provider request, validate exactly K skills, and save an immutable result."""

    config_file = Path(config_path).expanduser().resolve()
    project_root = config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    raw = load_config(config_file)
    if not isinstance(raw, Mapping) or set(raw) != {"skill_v3_proposal_smoke"}:
        raise ValueError("V3 proposal smoke config must contain only skill_v3_proposal_smoke")
    cfg = raw["skill_v3_proposal_smoke"]
    expected_fields = {
        "schema_version", "experiment_id", "activation_config_path", "candidate_count",
        "proposal_seed", "request_attempt_cap", "output_path",
    }
    if not isinstance(cfg, Mapping) or set(cfg) != expected_fields:
        raise ValueError("V3 proposal-smoke config has missing or unsupported fields")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("unsupported V3 proposal-smoke config schema")
    if not isinstance(cfg["experiment_id"], str) or not cfg["experiment_id"].strip():
        raise ValueError("V3 proposal-smoke experiment_id must be non-empty")
    if cfg["candidate_count"] != 2 or type(cfg["candidate_count"]) is not int:
        raise ValueError("V3 proposal smoke freezes K=2")
    if type(cfg["proposal_seed"]) is not int or cfg["proposal_seed"] < 0:
        raise ValueError("V3 proposal smoke seed must be a non-negative integer")
    if (type(cfg["request_attempt_cap"]) is not int
            or not 1 <= cfg["request_attempt_cap"] <= 8):
        raise ValueError("provider-only V3 smoke request cap must be 1..8")
    output_relative = cfg["output_path"]
    if (not isinstance(output_relative, str) or not output_relative.strip()
            or PurePosixPath(output_relative).is_absolute()
            or PureWindowsPath(output_relative).is_absolute()
            or ".." in PurePosixPath(output_relative).parts
            or ".." in PureWindowsPath(output_relative).parts):
        raise ValueError("V3 proposal-smoke output path must stay inside the project")
    artifact_dir = (project_root / output_relative).resolve()
    if not artifact_dir.is_relative_to(project_root):
        raise ValueError("V3 proposal-smoke output path escapes the project")
    output_walk = project_root
    for part in Path(output_relative).parts:
        if part in {"", "."}:
            continue
        output_walk = output_walk / part
        if output_walk.is_symlink():
            raise ValueError("V3 proposal-smoke output path cannot traverse symlinks")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"V3 proposal-smoke output directory is not empty: {artifact_dir}")
    artifact_path = artifact_dir / "proposal-smoke.json"
    if artifact_path.exists():
        raise FileExistsError(f"V3 proposal-smoke artifact already exists: {artifact_path}")

    activation_file = _resolve_project_file(project_root, str(cfg["activation_config_path"]))
    activation_raw = load_config(activation_file)
    review_path = _resolve_project_file(project_root, str(activation_raw["experiment"]["task_review_path"]))
    review_doc = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = ActivationSmokeManifest.from_mapping(activation_raw, task_review_document=review_doc)
    if (manifest.condition != "customer_representation_comparison"
            or manifest.customer_evolver_schema != "skill_v3"):
        raise ValueError("V3 proposal smoke must use the frozen skill_v3 comparison config")
    baseline_data = activation_raw["experiment"].get("customer_strategy")
    if not isinstance(baseline_data, Mapping):
        raise TypeError("V3 proposal smoke requires the matched V2 incumbent")
    incumbent = CustomerStrategy(**baseline_data)
    if not incumbent.is_v2:
        raise ValueError("V3 proposal smoke incumbent must be the seven-axis V2 baseline")

    signal_path = _resolve_project_file(
        project_root, str(activation_raw["experiment"]["reflection_seed_path"]),
    )
    reflection_seed = validate_reflection_seed_signals(
        json.loads(signal_path.read_text(encoding="utf-8")),
    )
    if sha256_json(reflection_seed) != manifest.reflection_seed_sha256:
        raise ValueError("V3 proposal smoke reflection seed differs from activation manifest")
    aggregate = CandidateEvaluation(
        strategy_id=customer_strategy_id(incumbent), episodes=(), panel_name="discovery",
    )
    reflection = sanitized_reflection_context(aggregate, prior_signals=reflection_seed)
    data_root = Path(tau2_data_dir).expanduser().resolve()
    tasks = _load_pinned_tasks(
        manifest, data_dir=data_root,
        task_selection=activation_raw["experiment"]["task_selection"],
        task_ids=manifest.evolution_task_ids + manifest.validation_task_ids,
    )
    literals = extract_forbidden_literals(tasks)

    if proposal_provider is None and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required; inject it from Keychain for this command")
    model = dict(manifest.role_models)["evolver"]
    model_args = role_model_args_for_runtime(manifest.role_model_args)["evolver"]
    provider = proposal_provider or LLMCustomerSkillEvolver(model=model, model_args=model_args)
    budget = RequestBudget(cap=cfg["request_attempt_cap"])
    captured: dict[str, Any] = {"proposal_context": None, "response_sha256": None}

    def capture(context: Any) -> Any:
        captured["proposal_context"] = context.to_dict()
        response = provider(context)
        if isinstance(response, CustomerSkillProviderResponse):
            captured["response_sha256"] = response.response_sha256
        else:
            captured["response_sha256"] = sha256_json(response)
        return response

    proposals: tuple[Any, ...] = ()
    failure_type: str | None = None
    validation_error: str | None = None
    try:
        if proposal_provider is None:
            from tau2.utils import llm_utils

            with budget.instrument_tau_llm_utils(llm_utils, retry_empty_responses=False):
                proposals = propose_customer_skills(
                    incumbent, 2, generation=0, seed=int(cfg["proposal_seed"]),
                    recent_failures=(), already_seen=(), already_tested_skills=(),
                    reflection_feedback=reflection, forbidden_literals=literals,
                    proposal_provider=capture,
                )
        else:
            proposals = propose_customer_skills(
                incumbent, 2, generation=0, seed=int(cfg["proposal_seed"]),
                recent_failures=(), already_seen=(), already_tested_skills=(),
                reflection_feedback=reflection, forbidden_literals=literals,
                proposal_provider=capture,
            )
    except Exception as exc:  # noqa: BLE001 - persist safe diagnostics without raw provider text
        failure_type = type(exc).__name__
        validation_error = "provider or candidate validation failed; inspect response fingerprint and run telemetry"

    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_copy = artifact_dir / "run-config.yaml"
    if not config_copy.exists():
        shutil.copyfile(config_file, config_copy)
    result = {
        "schema_version": 1,
        "experiment_id": str(cfg["experiment_id"]),
        "status": "complete" if failure_type is None else "failed",
        "failure_type": failure_type,
        "validation_error": validation_error,
        "provider_calls_are_episode_free": True,
        "proposal_context": captured["proposal_context"],
        "proposal_context_sha256": (
            None if captured["proposal_context"] is None
            else sha256_json(captured["proposal_context"])
        ),
        "proposal_response_sha256": captured["response_sha256"],
        "candidates": [
            {
                "candidate": item.to_dict(),
                "validation": "passed",
                "task_literal_leakage_detected": False,
            }
            for item in proposals
        ],
        "candidate_count_expected": 2,
        "candidate_count_validated": len(proposals),
        "forbidden_literal_check_count": len(literals),
        "reflection_seed_sha256": manifest.reflection_seed_sha256,
        "comparison_frozen_inputs": {
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
        },
        "activation_manifest_sha256": manifest.sha256,
        "task_review_sha256": manifest.task_semantic_review_sha256,
        "proposal_config_sha256": sha256_json(raw),
        "provider_model": model,
        "provider_model_args": dict(manifest.role_model_args)["evolver"],
        "provider_usage": budget.snapshot().to_dict(),
        "code_provenance": capture_code_provenance().to_dict(),
    }
    _write_once(artifact_path, result)
    return {**result, "artifact_path": str(artifact_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_from_config(args.config, tau2_data_dir=args.tau2_data_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"V3 proposal smoke failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - provider errors must not leak credentials
        print(f"V3 proposal smoke failed ({type(exc).__name__}); inspect its immutable artifact", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
