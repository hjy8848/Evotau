"""Immutable Phase 0 manifest construction and provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping


TAU_BENCH_REPOSITORY = "sierra-research/tau2-bench"
TAU_BENCH_COMMIT = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
TAU2_PACKAGE_VERSION = "1.0.1"

REQUIRED_SOURCE_PATHS = frozenset(
    {
        "data/tau2/domains/retail/tasks.json",
        "data/tau2/domains/retail/split_tasks.json",
        "data/tau2/domains/retail/db.json",
        "data/tau2/domains/retail/policy.md",
        "data/tau2/user_simulator/simulation_guidelines.md",
        "src/tau2/domains/retail/tools.py",
        "src/tau2/environment/environment.py",
        "src/tau2/registry.py",
        "src/tau2/orchestrator/orchestrator.py",
        "src/tau2/user/user_simulator.py",
        "src/tau2/agent/llm_agent.py",
        "src/tau2/runner/build.py",
        "src/tau2/runner/simulation.py",
        "src/tau2/data_model/simulation.py",
        "src/tau2/evaluator/evaluator.py",
        "src/tau2/evaluator/evaluator_nl_assertions.py",
        "src/tau2/evaluator/review_llm_judge.py",
        "src/tau2/evaluator/reviewer.py",
        "src/tau2/utils/llm_utils.py",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """Return Git's SHA-1 for a blob, including its object header."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_git_blob_sha1(path: str | Path, expected_sha1: str) -> None:
    actual = git_blob_sha1(Path(path).read_bytes())
    if actual != expected_sha1:
        raise ValueError(
            f"upstream file fingerprint mismatch for {path}: "
            f"expected {expected_sha1}, got {actual}"
        )


def _relative_path(value: str, field: str) -> str:
    if not value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"{field} must be a non-empty relative path")
    if ".." in PurePosixPath(value).parts or ".." in PureWindowsPath(value).parts:
        raise ValueError(f"{field} cannot escape the project directory")
    return value.replace("\\", "/")


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    phase: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    domain: str
    communication_mode: str
    evaluation_type: str
    split_name: str
    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    excluded_task_ids: tuple[str, ...]
    seed: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    output_path: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if self.upstream_repository != TAU_BENCH_REPOSITORY:
            raise ValueError("upstream repository must match the pinned tau-bench repository")
        if self.upstream_commit != TAU_BENCH_COMMIT:
            raise ValueError("upstream commit must match the audited tau-bench commit")
        if self.upstream_package_version != TAU2_PACKAGE_VERSION:
            raise ValueError("upstream package version does not match the pinned commit")
        if self.domain != "retail" or self.communication_mode != "half_duplex_text":
            raise ValueError("Phase 0 is fixed to tau-bench Retail half-duplex text mode")
        if self.evaluation_type != "all":
            raise ValueError("Phase 0 must explicitly use EvaluationType.ALL")
        if self.split_name != "train":
            raise ValueError("smoke E and V tasks must come from the official train split")
        if len(self.evolution_task_ids) != 1 or len(self.validation_task_ids) != 1:
            raise ValueError("Phase 0 smoke requires exactly one E task and one V task")
        selected = (
            self.evolution_task_ids + self.validation_task_ids + self.heldout_task_ids
        )
        if len(selected) != len(set(selected)):
            raise ValueError("E, V, and H task IDs must be disjoint")
        if self.heldout_task_ids:
            raise ValueError("the minimal smoke uses H=0; final heldout tasks stay sealed")
        if set(selected) & set(self.excluded_task_ids):
            raise ValueError("an excluded task cannot be selected for E, V, or H")
        if self.max_episodes != 1:
            raise ValueError("Phase 0 permits only one real episode")
        if self.max_steps != 64:
            raise ValueError("Phase 0 max_steps must be 64")
        if not 1 <= self.request_budget_cap <= 70:
            raise ValueError("Phase 0 request budget must be in the range 1..70")
        if self.provider_retries != 0:
            raise ValueError("provider retries must be disabled for Phase 0")
        if self.max_concurrency != 1:
            raise ValueError("Phase 0 runs with concurrency=1")
        if self.real_provider_enabled:
            missing = [name for name, model in self.role_models if not model]
            if missing:
                raise ValueError(f"live runs require frozen model IDs for: {', '.join(missing)}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.customer_strategy_sha256):
            raise ValueError("customer strategy hash must be a SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.service_strategy_sha256):
            raise ValueError("service strategy hash must be a SHA-256 hex digest")
        blobs = dict(self.source_blob_sha1)
        missing_sources = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing_sources:
            raise ValueError(f"missing upstream source fingerprints: {sorted(missing_sources)}")
        invalid = [
            path for path, sha in blobs.items() if not re.fullmatch(r"[0-9a-f]{40}", sha)
        ]
        if invalid:
            raise ValueError(f"invalid Git blob SHA-1 values for: {sorted(invalid)}")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentManifest":
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        models = experiment.get("models", {})
        source_blobs = experiment["source_blob_sha1"]
        customer = experiment.get("customer_strategy")
        service = experiment.get("service_strategy")
        return cls(
            experiment_id=str(experiment["id"]),
            phase=str(experiment["phase"]),
            upstream_repository=str(experiment["upstream"]["repository"]),
            upstream_commit=str(experiment["upstream"]["commit"]),
            upstream_package_version=str(experiment["upstream"]["package_version"]),
            domain=str(experiment["domain"]),
            communication_mode=str(experiment["communication_mode"]),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            evolution_task_ids=tuple(str(x) for x in selection["evolution"]),
            validation_task_ids=tuple(str(x) for x in selection["validation"]),
            heldout_task_ids=tuple(str(x) for x in selection.get("heldout", ())),
            excluded_task_ids=tuple(str(x) for x in selection.get("excluded", ())),
            seed=int(experiment["seed"]),
            max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=bool(experiment["real_provider_enabled"]),
            role_models=tuple(
                (name, None if models.get(name) is None else str(models[name]))
                for name in ("agent", "customer", "reviewer")
            ),
            source_blob_sha1=tuple(
                sorted((str(path), str(sha).lower()) for path, sha in source_blobs.items())
            ),
            customer_strategy_sha256=sha256_json(
                {"mode": "native_no_overlay"} if customer is None else customer
            ),
            service_strategy_sha256=sha256_json(
                {"rules": []} if service is None else service
            ),
            output_path=_relative_path(
                str(experiment["output_path"]), "output_path"
            ),
            checkpoint_path=_relative_path(
                str(experiment["checkpoint_path"]), "checkpoint_path"
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "upstream": {
                "repository": self.upstream_repository,
                "commit": self.upstream_commit,
                "package_version": self.upstream_package_version,
            },
            "domain": self.domain,
            "communication_mode": self.communication_mode,
            "evaluation_type": self.evaluation_type,
            "task_selection": {
                "source_split": self.split_name,
                "evolution": list(self.evolution_task_ids),
                "validation": list(self.validation_task_ids),
                "heldout": list(self.heldout_task_ids),
                "excluded": list(self.excluded_task_ids),
            },
            "seed": self.seed,
            "max_steps": self.max_steps,
            "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap,
            "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency,
            "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "strategy_sha256": {
                "customer": self.customer_strategy_sha256,
                "service": self.service_strategy_sha256,
            },
            "paths": {
                "output": self.output_path,
                "checkpoint": self.checkpoint_path,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


def write_manifest_once(path: str | Path, manifest: ExperimentManifest) -> Path:
    """Write an immutable JSON manifest; never replace an existing artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_document(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target
