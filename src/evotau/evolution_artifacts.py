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


def summarize_activation_artifacts(root, manifest_sha, allowed_tasks):
    """Freeze observational E/V activation statistics; association is not causal credit."""
    stats = {}
    total_turns = activated_turns = tokens = 0
    for path in sorted(Path(root).glob("episodes/*/skill-activation-trace.json")):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("unsafe activation summary source")
        record_path = path.parent / "episode-record.json"
        if not record_path.exists():
            continue  # Incomplete attempts are evidence, never scored observations.
        record = json.loads(record_path.read_text())
        if record["task_id"] not in allowed_tasks:
            continue
        trace = json.loads(path.read_text())
        if (
            trace.get("manifest_sha256") != manifest_sha
            or trace.get("episode_id") != record["episode_id"]
            or trace.get("artifact_sha256")
            != sha256_json({k: v for k, v in trace.items() if k != "artifact_sha256"})
        ):
            raise ValueError("activation summary source digest mismatch")
        for decision in trace["decisions"]:
            if decision.get("artifact_sha256") != sha256_json(
                {k: v for k, v in decision.items() if k != "artifact_sha256"}
            ):
                raise ValueError("activation summary decision digest mismatch")
            ids = decision["decision"]["active_skill_ids"]
            total_turns += 1
            activated_turns += bool(ids)
            tokens += decision["activated_guidance_tokens"]
            for ident in ids:
                row = stats.setdefault(
                    ident, {"count": 0, "tokens": 0, "scored_episodes": {}}
                )
                row["count"] += 1
                row["tokens"] += decision["activated_guidance_tokens"]
                row["scored_episodes"][record["episode_id"]] = record.get(
                    "task_success"
                )
    for row in stats.values():
        outcomes = [v for v in row["scored_episodes"].values() if type(v) is bool]
        row["success_rate"] = sum(outcomes) / len(outcomes) if outcomes else None
    return {
        "available": total_turns > 0,
        "scope": "completed E/V research rollouts including rejected candidates",
        "interpretation": "success rate is episode-level association; per-skill tokens are joint-turn overhead",
        "total_service_turns": total_turns,
        "activated_turns": activated_turns,
        "activation_rate": activated_turns / total_turns if total_turns else None,
        "rendered_skill_tokens": tokens,
        "skills": stats,
    }
