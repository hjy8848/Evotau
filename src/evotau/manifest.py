"""Immutable Phase 0 manifest construction and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

TAU_BENCH_REPOSITORY = "sierra-research/tau2-bench"
TAU_BENCH_COMMIT = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
TAU2_PACKAGE_VERSION = "1.0.1"
ROLE_NAMES = ("agent", "customer", "reviewer", "evaluator")
MODEL_ARGUMENT_NAMES = frozenset({
    "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty",
})
DEFAULT_ROLE_MODEL_ARGS = {role: {"temperature": 0.0} for role in ROLE_NAMES}
MVP_FAILURE_TAXONOMY = (
    ("identity_verification", "retail.policy:identity_verification", "missing_identity_verification"),
    ("pre_write", "retail.policy:explicit_confirmation", "missing_explicit_confirmation"),
    ("pre_write", "retail.policy:complete_change_scope", "incomplete_write_scope"),
)

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


DEFAULT_CUSTOMER_STRATEGY_HASH = sha256_json({
    "disclosure": "minimal_on_request", "request_order": "scenario_order",
    "challenge_style": "none", "challenge_budget": 0,
})
DEFAULT_SERVICE_STRATEGY_HASH = sha256_json({"rules": []})


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    git_commit: str | None
    working_tree_clean: bool | None
    source_sha256: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "git_commit": self.git_commit,
            "working_tree_clean": self.working_tree_clean,
            "source_sha256": self.source_sha256,
        }


def capture_code_provenance() -> CodeProvenance:
    """Fingerprint EvoTau sources and, when available, the containing Git state."""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    in_project = (project_root / "pyproject.toml").is_file()
    if in_project:
        source_paths = sorted(
            [*package_dir.rglob("*.py"), project_root / "pyproject.toml", *project_root.glob("configs/**/*.yaml")]
        )
        relative = lambda path: path.relative_to(project_root).as_posix()
    else:
        source_paths = sorted(package_dir.rglob("*.py"))
        relative = lambda path: f"evotau/{path.relative_to(package_dir).as_posix()}"
    digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            continue
        name = relative(path).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    commit = None
    clean = None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=False, capture_output=True, text=True, timeout=2,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root,
            check=False, capture_output=True, text=True, timeout=2,
        )
        if commit_result.returncode == 0 and status_result.returncode == 0:
            commit = commit_result.stdout.strip()
            clean = not status_result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return CodeProvenance(commit, clean, digest.hexdigest())


def _validate_code_provenance(git_commit: str | None, source_sha256: str) -> None:
    if git_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("EvoTau Git commit must be a full lowercase SHA-1")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("EvoTau source fingerprint must be a SHA-256 hex digest")


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


def freeze_role_model_args(
    raw: Mapping[str, Any] | None,
) -> tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...]:
    """Validate and freeze the generation arguments applied to each model role."""

    if raw is None:
        raw = DEFAULT_ROLE_MODEL_ARGS
    if not isinstance(raw, Mapping) or set(raw) != set(ROLE_NAMES):
        raise ValueError(f"model_args must contain exactly {sorted(ROLE_NAMES)}")
    frozen = []
    for role in ROLE_NAMES:
        params = raw[role]
        if not isinstance(params, Mapping) or not params:
            raise ValueError(f"model_args.{role} must be a non-empty mapping")
        if set(params) - MODEL_ARGUMENT_NAMES:
            raise ValueError(
                f"model_args.{role} contains unsupported arguments: "
                f"{sorted(set(params) - MODEL_ARGUMENT_NAMES)}"
            )
        normalized: list[tuple[str, float | int]] = []
        for name, value in sorted(params.items()):
            if name == "max_tokens":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"model_args.{role}.max_tokens must be a positive integer")
                normalized.append((name, value))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"model_args.{role}.{name} must be numeric")
            low, high = (0.0, 2.0) if name == "temperature" else (
                (0.0, 1.0) if name == "top_p" else (-2.0, 2.0)
            )
            if not low <= value <= high:
                raise ValueError(f"model_args.{role}.{name} must be in {low}..{high}")
            normalized.append((name, value))
        frozen.append((role, tuple(normalized)))
    return tuple(frozen)


def _freeze_role_models(raw: Mapping[str, Any]) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(ROLE_NAMES):
        raise ValueError(f"models must contain exactly {sorted(ROLE_NAMES)}")
    frozen = []
    for role in ROLE_NAMES:
        value = raw[role]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"models.{role} must be a non-empty model ID or null")
        frozen.append((role, value))
    return tuple(frozen)


def _role_model_args_payload(
    values: tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...],
) -> dict[str, dict[str, float | int]]:
    return {role: dict(params) for role, params in values}


def _role_runtime_arguments(
    values: tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...],
) -> dict[str, Any]:
    params = _role_model_args_payload(values)
    return {
        "agent_llm_args": {**params["agent"], "num_retries": 0},
        "customer_llm_args": {**params["customer"], "num_retries": 0},
        "reviewer_llm_args": params["reviewer"],
        "evaluator_llm_args": params["evaluator"],
        "provider_default_max_retries": 0,
        "cache_enabled": False,
        "review_mode": "full",
    }


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    phase: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
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
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    customer_strategy_sha256: str
    service_strategy_sha256: str
    output_path: str
    checkpoint_path: str

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
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
        if tuple(name for name, _ in self.role_models) != ROLE_NAMES:
            raise ValueError(f"role_models must freeze exactly {sorted(ROLE_NAMES)}")
        freeze_role_model_args(_role_model_args_payload(self.role_model_args))
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
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExperimentManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        models = experiment.get("models", {})
        source_blobs = experiment["source_blob_sha1"]
        customer = experiment.get("customer_strategy")
        service = experiment.get("service_strategy")
        code = capture_code_provenance()
        return cls(
            experiment_id=str(experiment["id"]),
            phase=str(experiment["phase"]),
            upstream_repository=str(experiment["upstream"]["repository"]),
            upstream_commit=str(experiment["upstream"]["commit"]),
            upstream_package_version=str(experiment["upstream"]["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
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
            role_models=_freeze_role_models(models),
            role_model_args=freeze_role_model_args(experiment.get("model_args")),
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
            "evotau": {
                "git_commit": self.evotau_git_commit,
                "working_tree_clean": self.evotau_working_tree_clean,
                "source_sha256": self.evotau_source_sha256,
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
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
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


@dataclass(frozen=True, slots=True)
class MechanismManifest:
    """Frozen protocol for the bounded two-generation Phase 3 smoke."""

    experiment_id: str
    upstream_repository: str
    upstream_commit: str
    upstream_package_version: str
    evotau_git_commit: str | None
    evotau_working_tree_clean: bool | None
    evotau_source_sha256: str
    evolution_task_id: str
    validation_task_id: str
    seed: int
    max_steps: int
    max_episodes: int
    request_budget_cap: int
    provider_retries: int
    max_concurrency: int
    real_provider_enabled: bool
    role_models: tuple[tuple[str, str | None], ...]
    role_model_args: tuple[tuple[str, tuple[tuple[str, float | int], ...]], ...]
    source_blob_sha1: tuple[tuple[str, str], ...]
    output_path: str
    checkpoint_path: str
    customer_candidates: int = 2
    generations: int = 2
    domain: str = "retail"
    communication_mode: str = "half_duplex_text"
    evaluation_type: str = "all"
    split_name: str = "train"
    excluded_task_ids: tuple[str, ...] = ()
    customer_strategy_sha256: str = DEFAULT_CUSTOMER_STRATEGY_HASH
    service_strategy_sha256: str = DEFAULT_SERVICE_STRATEGY_HASH

    def __post_init__(self) -> None:
        _validate_code_provenance(self.evotau_git_commit, self.evotau_source_sha256)
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if (self.upstream_repository, self.upstream_commit, self.upstream_package_version) != (
            TAU_BENCH_REPOSITORY, TAU_BENCH_COMMIT, TAU2_PACKAGE_VERSION
        ):
            raise ValueError("mechanism smoke must use the audited tau-bench pin")
        if self.domain != "retail" or self.communication_mode != "half_duplex_text" or self.evaluation_type != "all":
            raise ValueError("mechanism smoke is fixed to Retail half-duplex text with evaluation_type=all")
        if self.split_name != "train":
            raise ValueError("E and V tasks must come from the official train split")
        if not self.evolution_task_id or not self.validation_task_id or self.evolution_task_id == self.validation_task_id:
            raise ValueError("E and V task IDs must be distinct and non-empty")
        if {self.evolution_task_id, self.validation_task_id} & set(self.excluded_task_ids):
            raise ValueError("excluded tasks cannot be selected for E or V")
        if self.seed < 0 or self.max_steps != 64:
            raise ValueError("mechanism smoke requires a non-negative seed and max_steps=64")
        if self.customer_candidates != 2 or self.generations != 2:
            raise ValueError("minimal mechanism smoke freezes K=2 and exactly two generations")
        if not 1 <= self.max_episodes <= 23:
            raise ValueError("Phase 3 permits at most 23 episodes including Phase 0 integration")
        if not 1 <= self.request_budget_cap <= 1800:
            raise ValueError("Phase 3 provider-attempt cap must be in the range 1..1800")
        if self.provider_retries != 0 or self.max_concurrency != 1:
            raise ValueError("mechanism smoke requires provider retries=0 and concurrency=1")
        models = dict(self.role_models)
        if set(models) != set(ROLE_NAMES):
            raise ValueError(f"role_models must freeze exactly {sorted(ROLE_NAMES)}")
        freeze_role_model_args(_role_model_args_payload(self.role_model_args))
        if self.real_provider_enabled and any(not models[name] for name in models):
            raise ValueError("live smoke requires frozen model IDs for every role")
        blobs = dict(self.source_blob_sha1)
        missing = REQUIRED_SOURCE_PATHS - set(blobs)
        if missing:
            raise ValueError(f"missing upstream source fingerprints: {sorted(missing)}")
        invalid = [path for path, digest in blobs.items() if not re.fullmatch(r"[0-9a-f]{40}", digest)]
        if invalid:
            raise ValueError(f"invalid Git blob SHA-1 values for: {sorted(invalid)}")
        for name, digest in (("customer", self.customer_strategy_sha256), ("service", self.service_strategy_sha256)):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{name} strategy hash must be a SHA-256 hex digest")
        _relative_path(self.output_path, "output_path")
        _relative_path(self.checkpoint_path, "checkpoint_path")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MechanismManifest:
        experiment = raw["experiment"]
        selection = experiment["task_selection"]
        models = experiment.get("models", {})
        upstream = experiment["upstream"]
        if experiment.get("phase") != "3-two-generation-smoke":
            raise ValueError("MechanismManifest requires phase='3-two-generation-smoke'")
        if len(selection.get("evolution", ())) != 1 or len(selection.get("validation", ())) != 1:
            raise ValueError("mechanism smoke requires exactly one E task and one V task")
        if selection.get("heldout", ()):
            raise ValueError("Phase 3 smoke keeps H=0; heldout tasks remain sealed")
        code = capture_code_provenance()
        return cls(
            experiment_id=str(experiment["id"]),
            upstream_repository=str(upstream["repository"]),
            upstream_commit=str(upstream["commit"]),
            upstream_package_version=str(upstream["package_version"]),
            evotau_git_commit=code.git_commit,
            evotau_working_tree_clean=code.working_tree_clean,
            evotau_source_sha256=code.source_sha256,
            evolution_task_id=str(selection["evolution"][0]),
            validation_task_id=str(selection["validation"][0]),
            seed=int(experiment["seed"]), max_steps=int(experiment["max_steps"]),
            max_episodes=int(experiment["max_episodes"]),
            request_budget_cap=int(experiment["request_budget_cap"]),
            provider_retries=int(experiment["provider_retries"]),
            max_concurrency=int(experiment["max_concurrency"]),
            real_provider_enabled=bool(experiment["real_provider_enabled"]),
            role_models=_freeze_role_models(models),
            role_model_args=freeze_role_model_args(experiment.get("model_args")),
            source_blob_sha1=tuple(sorted((str(path), str(digest).lower())
                                          for path, digest in experiment["source_blob_sha1"].items())),
            output_path=_relative_path(str(experiment["output_path"]), "output_path"),
            checkpoint_path=_relative_path(str(experiment["checkpoint_path"]), "checkpoint_path"),
            customer_candidates=int(experiment.get("customer_candidates", 2)),
            generations=int(experiment.get("generations", 2)),
            domain=str(experiment["domain"]),
            communication_mode=str(experiment["communication_mode"]),
            evaluation_type=str(experiment["evaluation_type"]),
            split_name=str(selection["source_split"]),
            excluded_task_ids=tuple(str(task_id) for task_id in selection.get("excluded", ())),
            customer_strategy_sha256=sha256_json(
                {"disclosure": "minimal_on_request", "request_order": "scenario_order",
                 "challenge_style": "none", "challenge_budget": 0}
                if experiment.get("customer_strategy") is None else experiment["customer_strategy"]
            ),
            service_strategy_sha256=sha256_json(
                {"rules": []} if experiment.get("service_strategy") is None else experiment["service_strategy"]
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "phase": "3-two-generation-smoke",
            "upstream": {"repository": self.upstream_repository, "commit": self.upstream_commit,
                         "package_version": self.upstream_package_version},
            "evotau": {"git_commit": self.evotau_git_commit,
                       "working_tree_clean": self.evotau_working_tree_clean,
                       "source_sha256": self.evotau_source_sha256},
            "domain": self.domain, "communication_mode": self.communication_mode,
            "evaluation_type": self.evaluation_type, "split_name": self.split_name,
            "evolution_task_id": self.evolution_task_id, "validation_task_id": self.validation_task_id,
            "excluded_task_ids": list(self.excluded_task_ids),
            "seed": self.seed, "max_steps": self.max_steps, "max_episodes": self.max_episodes,
            "request_budget_cap": self.request_budget_cap, "provider_retries": self.provider_retries,
            "max_concurrency": self.max_concurrency, "customer_candidates": self.customer_candidates,
            "generations": self.generations, "real_provider_enabled": self.real_provider_enabled,
            "role_models": dict(self.role_models),
            "role_model_args": _role_model_args_payload(self.role_model_args),
            "source_blob_sha1": dict(self.source_blob_sha1),
            "runtime_arguments": _role_runtime_arguments(self.role_model_args),
            "failure_taxonomy": [
                {
                    "workflow_stage": stage,
                    "policy_rule_id": policy_rule_id,
                    "mistake_type": mistake,
                }
                for stage, policy_rule_id, mistake in MVP_FAILURE_TAXONOMY
            ],
            "failure_taxonomy_sha256": sha256_json(MVP_FAILURE_TAXONOMY),
            "strategy_sha256": {"customer": self.customer_strategy_sha256,
                                "service": self.service_strategy_sha256},
            "output_path": self.output_path, "checkpoint_path": self.checkpoint_path,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_payload())

    def to_document(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


def write_manifest_once(path: str | Path, manifest: ExperimentManifest | MechanismManifest) -> Path:
    """Write an immutable JSON manifest; never replace an existing artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_document(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target
