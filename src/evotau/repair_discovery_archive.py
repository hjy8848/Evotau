"""Immutable indexes of verified E discoveries, separate from incumbent selection."""

import json
from pathlib import Path

from .alternating import _write_json_once
from .customer_skills import CustomerSkill
from .records import EpisodeRecord, customer_strategy_id
from .repair_feedback import discovery_feedback, validate_pair
from .tau_provenance import sha256_json

PROTOCOL = "repair_discovery_handoff_v1"


class RepairDiscoveryArchive:
    def __init__(self, root, manifest_sha, task_ids):
        self.root = Path(root) / "repair-discoveries"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_sha, self.task_ids = manifest_sha, sorted(map(str, task_ids))
        if self.root.is_symlink():
            raise ValueError("unsafe discovery archive")
        for path in self.root.glob("*.json"):
            self.read(path)

    def read(self, path):
        if path.is_symlink():
            raise ValueError("unsafe discovery event")
        doc = json.loads(path.read_text())
        if (
            doc.get("protocol_version") != PROTOCOL
            or doc.get("manifest_sha256") != self.manifest_sha
            or doc.get("task_ids") != self.task_ids
            or doc.get("sha256")
            != sha256_json({k: v for k, v in doc.items() if k != "sha256"})
        ):
            raise ValueError("discovery archive identity/digest mismatch")
        if doc.get("kind") == "discovery":
            trial_path = self.root.parent / doc["payload"]["trial_ref"]
            if (
                trial_path.is_symlink()
                or trial_path.resolve().parent.parent != self.root.parent.resolve()
                or trial_path.parent.name
                not in (
                    "rc-bandit-trials",
                    "open-repair-trials",
                )
            ):
                raise ValueError("invalid source trial reference")
            envelope = json.loads(trial_path.read_text())
            if (
                envelope.get("manifest_sha256") != self.manifest_sha
                or envelope.get("sha256")
                != sha256_json({k: v for k, v in envelope.items() if k != "sha256"})
                or sha256_json(envelope["payload"]) != doc["payload"]["trial_sha256"]
            ):
                raise ValueError("discovery source trial changed")
        return doc

    def publish(
        self,
        trial,
        skill,
        pair,
        review_context,
        *,
        selected,
        trial_ref,
        customer_policy,
        evidence_rows,
        min_replications,
    ):
        """Recheck the actual review, not the integer reward. Index the immutable Trial."""
        if not trial["reward"]:
            return []
        source_path = Path(trial_ref)
        if source_path.is_symlink() or source_path.parent not in (
            self.root.parent / "rc-bandit-trials",
            self.root.parent / "open-repair-trials",
        ):
            raise ValueError("unsafe discovery source")
        source = json.loads(source_path.read_text())
        if (
            source.get("manifest_sha256") != self.manifest_sha
            or source.get("task_ids") != self.task_ids
            or source.get("payload") != trial
            or source.get("kind") != "final"
            or source.get("sha256")
            != sha256_json({k: v for k, v in source.items() if k != "sha256"})
        ):
            raise ValueError("discovery must index the actual immutable final Trial")
        validate_pair(pair)
        if review_context["pair"] != pair or trial["observations"]["service_pair_id"] != pair["pair_id"]:
            raise ValueError("discovery repair evidence identity mismatch")
        f = trial["feedback"]
        if f["observations_sha256"] != sha256_json(trial["observations"]):
            raise ValueError("discovery original observation digest mismatch")
        verified = discovery_feedback(
            trial["observations"],
            review_context,
            f["review"],
            reviewer_provenance=f["reviewer_provenance"],
            prior_keys=[],
            min_replications=min_replications,
        )
        if (
            not set(f["discovery_keys"]) <= set(verified["discovery_keys"])
            or trial["candidate_validity"] != "valid"
            or pair["pair_id"] != trial["service_pair_id"]
            or pair["kind"] != "promoted"
        ):
            raise ValueError("unverified discovery")
        customer = CustomerSkill.from_mapping(
            skill, evidence_rows=evidence_rows, policy=customer_policy
        )
        if (
            customer.procedure_id != trial["procedure_id"]
            or customer.candidate_id != trial["candidate_id"]
        ):
            raise ValueError("discovery Customer identity mismatch")
        results, indexed_keys = [], set()
        for item in f["review"]["discoveries"]:
            key = sha256_json(
                {
                    "task": item["task_id"],
                    "type": item["discovery_type"],
                    "mechanism": " ".join(item["mechanism"].lower().split()),
                }
            )
            if key not in f["discovery_keys"] or key in indexed_keys:
                continue
            indexed_keys.add(key)
            if item["task_id"] not in self.task_ids:
                raise ValueError("discovery outside E")
            payload = {
                "discovery_id": key,
                "repair_pair_id": pair["pair_id"],
                "before_service_id": pair["before_id"],
                "after_service_id": pair["after_id"],
                "trial_id": trial["trial_id"],
                "generation": trial["generation"],
                "trial_ref": str(Path(trial_ref).relative_to(self.root.parent)),
                "trial_sha256": sha256_json(trial),
                "candidate_id": customer.candidate_id,
                "procedure_id": customer.procedure_id,
                "customer_strategy_id": customer_strategy_id(customer.compile()),
                "customer_skill": customer.to_dict(),
                "compiled_customer": customer.compile().to_dict(),
                "discovery": item,
                "cost": trial["cost"],
                "reviewer_provenance": f["reviewer_provenance"],
                "selected_as_incumbent": selected,
                "novelty_scope": "reviewer_supported_observed_E_history",
                "source_cells": [
                    {
                        "seed": c["seed"],
                        "label": c["classification"],
                        "before_ref": c["before"]["trajectory_ref"],
                        "after_ref": c["after"]["trajectory_ref"],
                        "before_sha256": sha256_json(c["before"]),
                        "after_sha256": sha256_json(c["after"]),
                        "customer_validity": c["customer_validity"],
                    }
                    for c in trial["observations"]["cells"]
                    if c["task_id"] == item["task_id"]
                ],
            }
            doc = {
                "protocol_version": PROTOCOL,
                "manifest_sha256": self.manifest_sha,
                "task_ids": self.task_ids,
                "kind": "discovery",
                "payload": payload,
            }
            doc["sha256"] = sha256_json(doc)
            _write_json_once(self.root / f"discovery-{key}.json", doc)
            results.append(payload)
        return results

    def event(self, discovery_id, generation, candidate_id, result):
        if any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for c in candidate_id + discovery_id
        ):
            raise ValueError("unsafe discovery use identity")
        payload = {
            "discovery_id": discovery_id,
            "generation": generation,
            "service_candidate_id": candidate_id,
            "repair_result": result,
        }
        doc = {
            "protocol_version": PROTOCOL,
            "manifest_sha256": self.manifest_sha,
            "task_ids": self.task_ids,
            "kind": "repair_evaluation",
            "payload": payload,
        }
        doc["sha256"] = sha256_json(doc)
        _write_json_once(
            self.root / f"use-{generation}-{candidate_id}-{discovery_id}.json", doc
        )
        return doc


def service_handoff(discoveries, trials, runner, tasks, policy):
    """Service-safe projection of S+ observations, never Customer scenario/reviewer prose."""
    from .alternating import _service_context_episodes
    from .evolution_context import build_service_mutation_evidence

    if not discoveries:
        return []
    result = []
    for d in discoveries:
        trial = next(t for t in trials if sha256_json(t) == d["trial_sha256"])
        cells = [
            c
            for c in trial["observations"]["cells"]
            if c["task_id"] == d["discovery"]["task_id"]
        ]
        records = [EpisodeRecord.from_dict(c["after"]) for c in cells]
        rows = build_service_mutation_evidence(
            _service_context_episodes(records, runner, tasks),
            representative_cases=len(records),
            case_chars=policy["case_chars"],
        )
        # Free reviewer explanations belong to the Customer side, never to Service prompts.
        result.append(
            {
                "discovery_id": d["discovery_id"],
                "training_panel": "E",
                "type": d["discovery"]["discovery_type"],
                "task_id": d["discovery"]["task_id"],
                "seeds": d["discovery"]["seeds"],
                "observable_mechanism": d["discovery"]["mechanism"],
                "task_interactions": rows,
            }
        )
    return result
