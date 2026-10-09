"""Shared τ-bench pins and provenance helpers for the active EvoTau method."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

TAU_BENCH_REPOSITORY = "sierra-research/tau2-bench"
TAU_BENCH_COMMIT = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
TAU2_PACKAGE_VERSION = "1.0.1"
ROLE_NAMES = ("agent", "customer", "reviewer", "evaluator")
EVOLUTION_ROLE_NAMES = ("agent", "customer", "evaluator", "evolver")
MODEL_ARGUMENT_NAMES = frozenset({
    "temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty",
    "api_base", "thinking_mode", "enable_thinking", "reasoning_effort", "api_protocol", "api_key_env", "stream",
})
DEFAULT_ROLE_MODEL_ARGS = {
    role: {"temperature": 0.0} for role in (*ROLE_NAMES, "evolver")
}
REQUIRED_SOURCE_PATHS = frozenset({
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
})
ALTERNATING_REQUIRED_SOURCE_PATHS = REQUIRED_SOURCE_PATHS - frozenset({
    "src/tau2/evaluator/review_llm_judge.py",
    "src/tau2/evaluator/reviewer.py",
})
def alternating_source_paths(domain="retail"):
    if domain not in ("retail", "airline"):
        raise ValueError("unsupported native domain")
    paths = frozenset(path.replace("domains/retail/", f"domains/{domain}/")
                      for path in ALTERNATING_REQUIRED_SOURCE_PATHS)
    if domain == "airline":
        paths |= frozenset(f"src/tau2/domains/airline/{name}.py"
                           for name in ("environment", "data_model", "utils"))
    return paths


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

@dataclass(frozen=True, slots=True)
class CodeProvenance:
    git_commit: str | None
    working_tree_clean: bool | None
    source_sha256: str
    console_source_sha256: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        payload = {
            "git_commit": self.git_commit,
            "working_tree_clean": self.working_tree_clean,
            "source_sha256": self.source_sha256,
        }
        if self.console_source_sha256 is not None:
            payload.update(source_scope="runtime-v2", runtime_source_sha256=self.source_sha256,
                           console_source_sha256=self.console_source_sha256)
        return payload

def capture_code_provenance(*, runtime_only: bool = False) -> CodeProvenance:
    """Fingerprint EvoTau sources and, when available, the containing Git state."""

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    in_project = (project_root / "pyproject.toml").is_file()
    if in_project:
        source_paths = sorted(
            [*package_dir.rglob("*.py"), project_root / "pyproject.toml"]
        )
        relative = lambda path: path.relative_to(project_root).as_posix()
    else:
        source_paths = sorted(package_dir.rglob("*.py"))
        relative = lambda path: f"evotau/{path.relative_to(package_dir).as_posix()}"
    digest = hashlib.sha256()
    console_digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            continue
        name = relative(path).encode("utf-8")
        content = path.read_bytes()
        is_console = path.is_relative_to(package_dir / "web") or path == package_dir / "observability.py"
        target = console_digest if runtime_only and is_console else digest
        target.update(len(name).to_bytes(8, "big"))
        target.update(name)
        target.update(len(content).to_bytes(8, "big"))
        target.update(content)
    if runtime_only:
        for path in sorted((package_dir / "web").rglob("*")):
            if not path.is_file() or path.suffix not in {".html", ".js", ".css"}:
                continue
            name, content = relative(path).encode("utf-8"), path.read_bytes()
            console_digest.update(len(name).to_bytes(8, "big") + name)
            console_digest.update(len(content).to_bytes(8, "big") + content)

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
    return CodeProvenance(commit, clean, digest.hexdigest(), console_digest.hexdigest() if runtime_only else None)

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
    *,
    roles: tuple[str, ...] = ROLE_NAMES,
) -> tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...]:
    """Validate and freeze the generation arguments applied to each model role."""

    if raw is None:
        raw = {role: DEFAULT_ROLE_MODEL_ARGS[role] for role in roles}
    if not isinstance(raw, Mapping) or set(raw) != set(roles):
        raise ValueError(f"model_args must contain exactly {sorted(roles)}")
    frozen = []
    for role in roles:
        params = raw[role]
        if not isinstance(params, Mapping) or not params:
            raise ValueError(f"model_args.{role} must be a non-empty mapping")
        if set(params) - MODEL_ARGUMENT_NAMES:
            raise ValueError(
                f"model_args.{role} contains unsupported arguments: "
                f"{sorted(set(params) - MODEL_ARGUMENT_NAMES)}"
            )
        normalized: list[tuple[str, float | int | str]] = []
        for name, value in sorted(params.items()):
            if name == "enable_thinking":
                if type(value) is not bool or "thinking_mode" in params:
                    raise ValueError("enable_thinking must be boolean and cannot coexist with thinking_mode")
                normalized.append((name, value))
                continue
            if name == "stream":
                if role != "evolver" or type(value) is not bool:
                    raise ValueError("stream must be boolean and is supported for evolver only")
                if params.get("api_protocol") == "responses":
                    raise ValueError("Responses streaming is not implemented; no implicit transport change")
                normalized.append((name, value))
                continue
            if name == "api_protocol":
                if value != "responses" or role != "evolver":
                    raise ValueError(
                        f"model_args.{role}.api_protocol supports 'responses' for evolver only"
                    )
                normalized.append((name, value))
                continue
            if name == "api_key_env":
                if role != "evolver" or not isinstance(value, str) or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*", value,
                ):
                    raise ValueError(
                        f"model_args.{role}.api_key_env must be an environment variable name "
                        "for the evolver"
                    )
                normalized.append((name, value))
                continue
            if name == "max_tokens":
                if type(value) is not int or value <= 0:
                    raise ValueError(f"model_args.{role}.max_tokens must be a positive integer")
                normalized.append((name, value))
                continue
            if name == "thinking_mode":
                if not isinstance(value, str) or value not in {"disabled", "enabled"}:
                    raise ValueError(
                        f"model_args.{role}.thinking_mode must be 'disabled' or 'enabled'"
                    )
                normalized.append((name, value))
                continue
            if name == "reasoning_effort":
                allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
                if not isinstance(value, str) or value not in allowed_efforts:
                    raise ValueError(
                        f"model_args.{role}.reasoning_effort must be one of "
                        f"{sorted(allowed_efforts)}"
                    )
                normalized.append((name, value))
                continue
            if name == "api_base":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"model_args.{role}.api_base must be an HTTP(S) URL")
                parsed = urlsplit(value)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.netloc
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError(
                        f"model_args.{role}.api_base must be an HTTP(S) URL without credentials"
                    )
                normalized.append((name, value.rstrip("/")))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"model_args.{role}.{name} must be numeric")
            low, high = (0.0, 2.0) if name == "temperature" else (
                (0.0, 1.0) if name == "top_p" else (-2.0, 2.0)
            )
            if not low <= value <= high:
                raise ValueError(f"model_args.{role}.{name} must be in {low}..{high}")
            normalized.append((name, value))
        normalized_names = {name for name, _ in normalized}
        if "api_protocol" in normalized_names:
            if not {"api_base", "reasoning_effort"} <= normalized_names:
                raise ValueError(
                    "model_args.evolver.api_protocol='responses' requires api_base "
                    "and reasoning_effort"
                )
        elif "api_key_env" in normalized_names:
            raise ValueError("model_args.evolver.api_key_env requires api_protocol='responses'")
        frozen.append((role, tuple(normalized)))
    return tuple(frozen)

def _freeze_role_models(
    raw: Mapping[str, Any],
    *,
    roles: tuple[str, ...] = ROLE_NAMES,
) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(raw, Mapping) or set(raw) != set(roles):
        raise ValueError(f"models must contain exactly {sorted(roles)}")
    frozen = []
    for role in roles:
        value = raw[role]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"models.{role} must be a non-empty model ID or null")
        frozen.append((role, value))
    return tuple(frozen)

def _role_model_args_payload(
    values: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...],
) -> dict[str, dict[str, float | int | str]]:
    return {role: dict(params) for role, params in values}

def role_model_args_for_runtime(
    values: tuple[tuple[str, tuple[tuple[str, float | int | str], ...]], ...],
) -> dict[str, dict[str, Any]]:
    """Translate frozen role settings into LiteLLM completion keyword arguments."""

    params = _role_model_args_payload(values)
    for role in params:
        thinking_mode = params[role].pop("thinking_mode", None)
        if thinking_mode is not None:
            params[role]["extra_body"] = {"thinking": {"type": thinking_mode}}
        if "enable_thinking" in params[role]:
            params[role]["extra_body"] = {"enable_thinking": params[role].pop("enable_thinking")}
    return params
def write_manifest_once(path: str | Path, manifest: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest.to_document(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target
