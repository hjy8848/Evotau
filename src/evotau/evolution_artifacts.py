"""Immutable stage journal with manifest/input/output digests and exact reuse."""

import json
from pathlib import Path

from .tau_provenance import sha256_json


class EvolutionJournal:
    def __init__(self, root, manifest_sha):
        self.root = Path(root) / "evolution-v2"
        self.manifest_sha = manifest_sha
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("journal cannot be a symlink")
        for path in self.root.glob("*.json"):
            self.read(path)

    def read(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe stage artifact")
        doc = json.loads(path.read_text())
        if doc.get("manifest_sha256") != self.manifest_sha or doc.get(
            "payload_sha256"
        ) != sha256_json(doc["payload"]):
            raise ValueError("stage manifest/payload digest mismatch")
        if doc.get("envelope_sha256") != sha256_json(
            {k: v for k, v in doc.items() if k != "envelope_sha256"}
        ):
            raise ValueError("stage envelope digest mismatch")
        return doc

    def freeze(self, name, inputs, callback):
        from .alternating import _write_json_once

        path = self.root / f"{name}.json"
        input_sha = sha256_json(inputs)
        if path.exists():
            doc = self.read(path)
            if doc["input_sha256"] != input_sha:
                raise ValueError(f"frozen stage input changed: {name}")
            return doc["payload"]
        payload = callback()
        doc = {
            "schema_version": 3,
            "manifest_sha256": self.manifest_sha,
            "stage": name,
            "input_sha256": input_sha,
            "payload": payload,
            "payload_sha256": sha256_json(payload),
        }
        doc["envelope_sha256"] = sha256_json(doc)
        _write_json_once(path, doc)
        return payload
