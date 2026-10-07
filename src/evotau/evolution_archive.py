"""Evolution-only candidate retention, ancestry, semantic dedup and replay."""

import json
import re
from difflib import SequenceMatcher

from .tau_provenance import sha256_json


def normalized_text(value):
    return " ".join(re.findall(r"\w+", value.lower()))


def mutation_text(mutation):
    parts = [mutation.get("skill") or {}, *(mutation.get("children") or [])]
    return normalized_text(
        " ".join(
            p.get("trigger", "")
            + " "
            + p.get("guidance", "")
            + " "
            + json.dumps(p.get("activation_signature", {}), sort_keys=True)
            for p in parts
        )
    )


def semantic_duplicate(mutation, entries):
    for entry in entries:
        prior = entry["mutation"]
        if (entry.get("effect") or {}).get("accepted") is True:
            continue
        same_structure = (prior["operation"], prior.get("target_skill_id")) == (
            mutation["operation"],
            mutation.get("target_skill_id"),
        )
        same_family = normalized_text(prior["semantic_family"]) == normalized_text(
            mutation["semantic_family"]
        )
        similarity = SequenceMatcher(
            None, mutation_text(prior), mutation_text(mutation)
        ).ratio()
        delta = mutation.get("substantive_delta_from_prior", "").strip()
        if (
            same_structure
            and (same_family or similarity >= 0.88)
            and (similarity >= 0.72 or same_family)
            and not delta
        ):
            return entry["mutation_id"]
        if similarity == 1 and same_structure:
            return entry["mutation_id"]
    return None


def crossover_eligible(left, right):
    a, b = left["effect"], right["effect"]
    fa, fb = set(a["fail_to_pass"]), set(b["fail_to_pass"])
    ba, bb = set(a["pass_to_fail"]), set(b["pass_to_fail"])
    return bool(
        fa - fb
        and fb - fa
        and not (fa & bb or fb & ba or ba & bb)
        and left["mutation"]["semantic_family"] != right["mutation"]["semantic_family"]
    )


class EvolutionArchive:
    def __init__(self, entries=(), *, max_candidates=5):
        self.entries = list(entries)
        self.max_candidates = max_candidates

    def add(self, entry):
        if any(e["mutation_id"] == entry["mutation_id"] for e in self.entries):
            return
        self.entries.append(entry)

        def score(e):
            effect = e["effect"]
            return (
                float(effect.get("candidate_accuracy") or 0),
                float(effect.get("helpfulness") or 0),
                -float(effect.get("harmfulness") or 0),
                -float(effect.get("new_stuck_rate") or 0),
                -float(effect.get("token_delta") or 0),
            )

        def dominated(e):
            x = score(e)
            return any(
                all(a >= b for a, b in zip(score(other), x, strict=True))
                and score(other) != x
                for other in self.entries
            )

        ranked = sorted(
            self.entries,
            key=lambda e: (
                dominated(e),
                not e["effect"]["accepted"],
                -len(e["effect"]["fail_to_pass"]),
                tuple(-x for x in score(e)),
                e["mutation_id"],
            ),
        )
        self.entries = ranked[: self.max_candidates]

    def to_dict(self):
        return {
            "entries": self.entries,
            "max_candidates": self.max_candidates,
            "runtime_deployed": False,
        }


class CustomerChallengeArchive:
    def __init__(self, entries=(), *, max_candidates=20):
        self.entries = list(entries)
        self.max_candidates = max_candidates

    def add(self, proposal, accuracy, generation, accepted, semantic_preserved):
        text = proposal["strategy"]
        ident = sha256_json({"text": text})[:16]
        overlap = max(
            (
                SequenceMatcher(
                    None, normalized_text(text), normalized_text(e["strategy"])
                ).ratio()
                for e in self.entries
            ),
            default=0,
        )
        prior = next((e for e in self.entries if e["strategy_id"] == ident), None)
        if prior is not None:
            prior["failure_frequency"] += 1 - accuracy
            prior["last_generation"] = generation
            return
        self.entries.append(
            {
                "strategy_id": ident,
                **proposal,
                "accuracy": accuracy,
                "generation": generation,
                "last_generation": generation,
                "accepted": accepted,
                "novelty": 1 - overlap,
                "overlap": overlap,
                "semantic_preservation_status": semantic_preserved,
                "failure_frequency": 1 - accuracy,
            }
        )
        self.entries = sorted(
            self.entries, key=lambda e: (e["accuracy"], -e["novelty"], e["strategy_id"])
        )[: self.max_candidates]

    def replay(self, count, generation, current_strategy):
        eligible = [
            e
            for e in self.entries
            if e["strategy"] != current_strategy
            and e["semantic_preservation_status"] is True
        ]
        return sorted(
            eligible,
            key=lambda e: (
                -(e["failure_frequency"] + 0.05 * (generation - e["last_generation"])),
                e["strategy_id"],
            ),
        )[:count]
