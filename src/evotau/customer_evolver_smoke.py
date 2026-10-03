"""Provider-only smoke for the direct Customer Evolver; runs no τ-bench episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .budget import RequestBudget
from .customer_evolver_v2 import (
    SYSTEM_PROMPT,
    CustomerEvolverResponseError,
    LLMCustomerStrategyEvolver,
    propose_customer_strategies,
)
from .manifest import (
    capture_code_provenance,
    freeze_role_model_args,
    role_model_args_for_runtime,
    sha256_json,
)
from .phase0 import load_config
from .strategies import CustomerStrategy

_CONFIG_FIELDS = frozenset({
    "schema_version", "experiment_id", "real_provider_enabled", "model",
    "model_args", "generation", "candidate_count", "proposal_seed",
    "incumbent_strategy", "output_path",
})


@dataclass(frozen=True, slots=True)
class CustomerEvolverSmokeConfig:
    experiment_id: str
    real_provider_enabled: bool
    model: str
    model_args: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]
    generation: int
    candidate_count: int
    proposal_seed: int
    incumbent_strategy: CustomerStrategy
    output_path: str

    @classmethod
    def from_mapping(cls, raw: Any) -> CustomerEvolverSmokeConfig:
        if not isinstance(raw, dict) or set(raw) != {"proposal_smoke"}:
            raise ValueError("proposal-smoke config must contain only proposal_smoke")
        values = raw["proposal_smoke"]
        if not isinstance(values, dict) or set(values) != _CONFIG_FIELDS:
            raise ValueError("proposal-smoke config has missing or unsupported fields")
        if type(values["schema_version"]) is not int or values["schema_version"] != 1:
            raise ValueError("unsupported Customer Evolver smoke config version")
        experiment_id = values["experiment_id"]
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("proposal-smoke experiment_id must be non-empty")
        if type(values["real_provider_enabled"]) is not bool or not values["real_provider_enabled"]:
            raise ValueError("proposal-only provider smoke requires real_provider_enabled: true")
        model = values["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("proposal-smoke model must be a frozen model ID")
        if (type(values["generation"]) is not int or values["generation"] < 0
                or type(values["candidate_count"]) is not int or values["candidate_count"] < 1
                or type(values["proposal_seed"]) is not int or values["proposal_seed"] < 0):
            raise ValueError("proposal-smoke generation, K, and seed must be valid integers")
        frozen_args = freeze_role_model_args(
            {"evolver": values["model_args"]}, roles=("evolver",),
        )
        args = dict(dict(frozen_args)["evolver"])
        if "temperature" not in args or args.get("thinking_mode") != "disabled":
            raise ValueError("proposal-smoke must freeze temperature and disable model thinking")
        customer = values["incumbent_strategy"]
        if not isinstance(customer, dict):
            raise TypeError("proposal-smoke incumbent_strategy must be a seven-field mapping")
        incumbent = CustomerStrategy(**customer)
        if not incumbent.is_v2 or set(customer) != set(incumbent.to_dict()):
            raise ValueError("proposal-smoke requires an exact seven-field v2 incumbent")
        output_path = values["output_path"]
        if (not isinstance(output_path, str) or not output_path.strip()
                or PurePosixPath(output_path).is_absolute()
                or PureWindowsPath(output_path).is_absolute()
                or ".." in PurePosixPath(output_path).parts
                or ".." in PureWindowsPath(output_path).parts):
            raise ValueError("proposal-smoke output_path must stay inside the project")
        return cls(
            experiment_id=experiment_id,
            real_provider_enabled=True,
            model=model,
            model_args=frozen_args,
            generation=values["generation"],
            candidate_count=values["candidate_count"],
            proposal_seed=values["proposal_seed"],
            incumbent_strategy=incumbent,
            output_path=output_path.replace("\\", "/"),
        )


class CustomerEvolverSmokeFailed(RuntimeError):
    """A failed provider proposal with its immutable diagnostic artifact saved."""


class _ProviderCallFailed(RuntimeError):
    def __init__(self, error_type: str):
        super().__init__(error_type)
        self.error_type = error_type


def run_from_config(
    config_path: str | Path,
    *,
    proposal_provider: Any | None = None,
) -> dict[str, Any]:
    """Make one completion request, validate K strategies, and save a write-once artifact."""

    config_file = Path(config_path).expanduser().resolve()
    project_root = (
        config_file.parent.parent if config_file.parent.name == "configs" else config_file.parent
    )
    raw_config = load_config(config_file)
    config = CustomerEvolverSmokeConfig.from_mapping(raw_config)
    model_args = role_model_args_for_runtime(config.model_args)["evolver"]
    config_sha256 = sha256_json(raw_config)
    artifact_dir = _resolve_output_dir(project_root, config.output_path)
    artifact_path = artifact_dir / "proposal-smoke.json"
    if artifact_path.exists():
        raise FileExistsError(f"proposal-smoke artifact already exists: {artifact_path}")
    if proposal_provider is None and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required; inject it from Keychain for this command")

    provider = proposal_provider or LLMCustomerStrategyEvolver(
        model=config.model,
        model_args=model_args,
    )
    budget = RequestBudget(cap=None)
    provenance = capture_code_provenance().to_dict()
    response_metadata: dict[str, Any] = {}

    def capture_response(context: Any) -> Any:
        try:
            response = provider(context)
        except CustomerEvolverResponseError as exc:
            response_metadata["response_sha256"] = exc.response_sha256
            raise
        except Exception as exc:  # noqa: BLE001 - never persist provider exception text
            raise _ProviderCallFailed(type(exc).__name__) from None
        response_metadata.update(_response_summary(response))
        return response

    try:
        if proposal_provider is None:
            from tau2.utils import llm_utils

            with budget.instrument_tau_llm_utils(llm_utils, retry_empty_responses=False):
                candidates = _propose(config, capture_response)
        else:
            candidates = _propose(config, capture_response)
    except _ProviderCallFailed as exc:
        result = _base_result(config, raw_config, config_sha256, provenance, budget)
        result.update({
            "status": "failed",
            "failure_stage": "provider_call",
            "error_type": exc.error_type,
            "candidate_proposals": [],
        })
        result.update(response_metadata)
        _write_json_once(artifact_path, result)
        raise CustomerEvolverSmokeFailed(
            f"Customer Evolver provider call failed ({exc.error_type}); artifact: {artifact_path}"
        ) from None
    except CustomerEvolverResponseError as exc:
        result = _base_result(config, raw_config, config_sha256, provenance, budget)
        result.update({
            "status": "failed",
            "failure_stage": "response_format",
            "error_type": type(exc).__name__,
            "validation_error": str(exc),
            "candidate_proposals": [],
            "response_sha256": exc.response_sha256,
        })
        _write_json_once(artifact_path, result)
        raise CustomerEvolverSmokeFailed(
            f"Customer Evolver response was not valid JSON; artifact: {artifact_path}"
        ) from None
    except Exception as exc:  # noqa: BLE001 - persist provider diagnostics without raw exception text
        result = _base_result(config, raw_config, config_sha256, provenance, budget)
        result.update({
            "status": "failed",
            "failure_stage": "candidate_validation",
            "error_type": type(exc).__name__,
            "validation_error": _safe_diagnostic(str(exc)),
            "candidate_proposals": [],
        })
        result.update(response_metadata)
        _write_json_once(artifact_path, result)
        raise CustomerEvolverSmokeFailed(
            f"Customer Evolver proposal failed ({type(exc).__name__}); artifact: {artifact_path}"
        ) from None

    result = _base_result(config, raw_config, config_sha256, provenance, budget)
    result.update({
        "status": "complete",
        "candidate_proposals": [
            {
                "strategy_id": item.strategy_id,
                "strategy": item.strategy.to_dict(),
                "changed_fields": list(item.changed_fields),
                "hypothesis": item.expected_behavioral_effect,
                "evidence_refs": list(item.supporting_failure_ids),
            }
            for item in candidates
        ],
        "proposal_context_sha256": candidates[0].proposal_context_sha256,
    })
    result.update(response_metadata)
    _write_json_once(artifact_path, result)
    return {**result, "artifact_path": str(artifact_path)}


def _propose(
    config: CustomerEvolverSmokeConfig,
    provider: Any,
) -> tuple:
    return propose_customer_strategies(
        config.incumbent_strategy,
        config.candidate_count,
        generation=config.generation,
        seed=config.proposal_seed,
        recent_failures=(),
        already_seen=(),
        already_tested_strategies=(),
        evolution_task_ids=(),
        proposal_provider=provider,
    )


def _response_summary(value: Any) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = repr(type(value).__name__)
    result: dict[str, Any] = {
        "response_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "response_type": type(value).__name__,
    }
    if isinstance(value, dict):
        rows = value.get("candidates")
        result["response_top_level_keys"] = sorted(str(key) for key in value)
        result["response_candidate_count"] = len(rows) if isinstance(rows, list) else None
        if isinstance(rows, list):
            result["response_candidate_shapes"] = [
                {
                    "candidate_keys": sorted(str(key) for key in row) if isinstance(row, dict) else None,
                    "strategy_keys": sorted(str(key) for key in row.get("strategy", {}))
                    if isinstance(row, dict) and isinstance(row.get("strategy"), dict) else None,
                    "changed_fields": row.get("changed_fields") if isinstance(row, dict) else None,
                    "evidence_ref_count": len(row.get("evidence_refs", []))
                    if isinstance(row, dict) and isinstance(row.get("evidence_refs"), list) else None,
                    "hypothesis_length": len(row.get("hypothesis", ""))
                    if isinstance(row, dict) and isinstance(row.get("hypothesis"), str) else None,
                    "hypothesis_preview": _safe_diagnostic(row.get("hypothesis", ""))
                    if isinstance(row, dict) and isinstance(row.get("hypothesis"), str) else None,
                }
                for row in rows[:8]
            ]
    return result


def _safe_diagnostic(value: str) -> str:
    # Validator messages are useful, but strip credential-shaped substrings defensively.
    import re

    cleaned = re.sub(r"(?i)bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    return cleaned[:400]


def _base_result(
    config: CustomerEvolverSmokeConfig,
    raw_config: dict[str, Any],
    config_sha256: str,
    provenance: dict[str, Any],
    budget: RequestBudget,
) -> dict[str, Any]:
    context = {
        "schema_version": 2,
        "generation": config.generation,
        "K": config.candidate_count,
        "proposal_seed": config.proposal_seed,
        "incumbent_strategy": config.incumbent_strategy.to_dict(),
        "already_tested_strategies": [],
        "already_tested_strategy_ids": [],
        "verified_failure_summaries": [],
    }
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha256,
        "config": raw_config["proposal_smoke"],
        "code_provenance": provenance,
        "model": config.model,
        "runtime_model_args": role_model_args_for_runtime(config.model_args)["evolver"],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "proposal_context_sha256": sha256_json(context),
        "request_scope": {
            "task_ids_provided": [],
            "verified_failure_summaries_provided": 0,
            "tau_bench_episodes_run": 0,
        },
        "provider_budget": budget.snapshot().to_dict(),
    }


def _resolve_output_dir(project_root: Path, relative: str) -> Path:
    target = project_root / relative
    current = project_root
    for part in PurePosixPath(relative).parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise ValueError("proposal-smoke output path cannot traverse symlinks")
    resolved_root = project_root.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError("proposal-smoke output directory must stay inside the project")
    return target


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="one-time provider and output configuration; use a fresh output path for each attempt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_from_config(args.config)
    except CustomerEvolverSmokeFailed as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Customer Evolver proposal smoke failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - provider exceptions may contain credentials
        print(
            f"Customer Evolver proposal smoke failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
