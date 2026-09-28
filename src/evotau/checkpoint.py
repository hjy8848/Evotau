"""Atomic, manifest-bound evolution checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import sha256_json


@dataclass(frozen=True, slots=True)
class EvolutionCheckpoint:
    manifest_hash: str
    generation: int
    state: dict[str, Any]
    completed_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if len(self.completed_units) != len(set(self.completed_units)):
            raise ValueError("completed_units must be unique for idempotent resume")


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    return sha256_json(manifest)


def save_checkpoint(path: str | Path, checkpoint: EvolutionCheckpoint) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "schema_version": 1,
        "manifest_hash": checkpoint.manifest_hash,
        "generation": checkpoint.generation,
        "state": checkpoint.state,
        "completed_units": list(checkpoint.completed_units),
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(path: str | Path, *, expected_manifest_hash: str) -> EvolutionCheckpoint:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported evolution checkpoint schema")
    if value.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("checkpoint manifest does not match the frozen experiment manifest")
    return EvolutionCheckpoint(
        manifest_hash=value["manifest_hash"], generation=int(value["generation"]),
        state=dict(value["state"]), completed_units=tuple(value.get("completed_units", ())),
    )
