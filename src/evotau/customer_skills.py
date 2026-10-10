"""Versioned task-faithful procedures compiled into the existing Customer overlay."""

import json
import re
from dataclasses import dataclass

from .strategies import PromptStrategy
from .tau_provenance import sha256_json

PROTOCOL = "task_faithful_customer_skill_v1"
DEFAULT_CUSTOMER_POLICY = {
    "protocol_version": PROTOCOL,
    "max_field_chars": 1200,
    "max_skill_tokens": 1600,
    "max_context_tokens": 40000,
    "representative_cases": 3,
    "case_chars": 24000,
    "max_history": 20,
}
FIELDS = {
    "schema_version",
    "mechanism",
    "trigger",
    "procedure",
    "intensity",
    "stop_conditions",
    "hypothesis",
    "evidence_refs",
}
INVARIANTS = """Task-faithful Customer invariants override this procedure:
Preserve the original business goals, identity, facts, budget, payment constraints and
conditional authorization. Use only facts supplied by the original scenario or observed
environment. Never invent a request or impersonate, deceive, inject instructions, or jailbreak.
Provide directly requested necessary information truthfully and promptly. Preserve dependencies
between actions and obtain adequate cost explanations before authorizing constrained actions.
Do not extend dialogue solely to exhaust steps, withhold necessary facts, or stop before the
original goals are addressed. If this procedure conflicts with the scenario or native guidelines,
stop the procedure and follow the original scenario and guidelines. Intensity affects tone only.
"""


def validate_customer_policy(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_CUSTOMER_POLICY):
        raise ValueError("unknown Customer protocol fields")
    policy = {**DEFAULT_CUSTOMER_POLICY, **value}
    if policy["protocol_version"] != PROTOCOL:
        raise ValueError("unsupported Customer protocol")
    for key in set(policy) - {"protocol_version"}:
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"Customer {key} must be a positive integer")
    if policy["representative_cases"] < 2:
        raise ValueError("Customer evidence needs contrast cases")
    return policy


def proxy_tokens(text):
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def evidence_registry(rows):
    """Only supplied E messages are legal; hashes stay program-owned and exact."""
    registry = set()
    for row in rows:
        for item in row["trajectory"]["messages"]:
            registry.add(
                sha256_json(
                    {
                        "task_id": row["task"]["task_id"],
                        "seed": row["seed"],
                        "trajectory_ref": row["trajectory_ref"],
                        **item["evidence_ref"],
                    }
                )
            )
    return registry


@dataclass(frozen=True)
class CustomerSkill:
    payload_json: str

    @classmethod
    def from_mapping(cls, value, *, evidence_rows=(), policy=None):
        settings = validate_customer_policy(policy or {})
        if not isinstance(value, dict) or set(value) != FIELDS:
            raise ValueError("Customer Skill requires exactly its versioned fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported Customer Skill schema")
        if value["intensity"] not in ("low", "medium", "high"):
            raise ValueError("invalid Customer intensity")
        if (
            not isinstance(value["procedure"], list)
            or not 2 <= len(value["procedure"]) <= 5
        ):
            raise ValueError("Customer procedure requires 2–5 steps")
        if (
            not isinstance(value["stop_conditions"], list)
            or not 1 <= len(value["stop_conditions"]) <= 5
        ):
            raise ValueError("Customer stop_conditions requires 1–5 conditions")
        texts = [value[k] for k in ("mechanism", "trigger", "hypothesis")]
        texts += value["procedure"] + value["stop_conditions"]
        if any(
            not isinstance(t, str)
            or not t.strip()
            or len(t) > settings["max_field_chars"]
            for t in texts
        ):
            raise ValueError("empty or oversized Customer content")
        for group in (value["procedure"], value["stop_conditions"]):
            normalized = [" ".join(v.casefold().split()) for v in group]
            if len(normalized) != len(set(normalized)):
                raise ValueError("duplicate Customer procedure/stop content")
        # Small explicit catches, never a claim of general natural-language proof.
        # Negated safety instructions are permitted; paraphrases require semantic review.
        for text in value["procedure"]:
            for match in re.finditer(
                r"\b(?:change|increase|replace|override) (?:the |your |original )?(?:budget|identity|goal|reservation id|order id|payment method)\b|"
                r"\b(?:give|grant|use) (?:a |an )?(?:unconditional|blanket) (?:authorization|consent)|"
                r"\b(?:refuse to provide|never provide|withhold) (?:necessary|required) (?:facts|information)|"
                r"\b(?:invent|fabricate) (?:a |new )?(?:request|identity|fact)|"
                r"\b(?:ignore|bypass) (?:the )?(?:policy|system instructions|evaluator)|"
                r"\b(?:keep asking indefinitely|stop early to fail)\b|"
                r"(?:修改预算|更换身份|无条件授权|拒绝提供必要信息|编造事实)",
                text,
                re.IGNORECASE,
            ):
                if not re.search(
                    r"(?:do not|never|don't|avoid)\s+(?:\w+\s+){0,3}$",
                    text[: match.start()],
                    re.IGNORECASE,
                ):
                    raise ValueError(
                        "explicit task-changing or noncooperative instruction"
                    )
        refs = value["evidence_refs"]
        if not isinstance(refs, list):
            raise TypeError("Customer evidence_refs must be array")
        registry = evidence_registry(evidence_rows)
        seen = set()
        for ref in refs:
            if (
                not isinstance(ref, dict)
                or set(ref)
                != {
                    "task_id",
                    "seed",
                    "trajectory_ref",
                    "projected_message_index",
                    "message_sha256",
                }
                or type(ref["seed"]) is not int
                or type(ref["projected_message_index"]) is not int
            ):
                raise ValueError("invalid Customer evidence reference")
            digest = sha256_json(ref)
            if digest not in registry or digest in seen:
                raise ValueError(
                    "Customer evidence absent from supplied E or duplicated"
                )
            seen.add(digest)
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if proxy_tokens(payload) > settings["max_skill_tokens"]:
            raise ValueError("Customer Skill exceeds explicit token allowance")
        skill = cls(payload)
        if proxy_tokens(skill.compile().text) > settings["max_skill_tokens"]:
            raise ValueError(
                "compiled Customer overlay exceeds explicit token allowance"
            )
        return skill

    def to_dict(self):
        return json.loads(self.payload_json)

    @property
    def candidate_id(self):
        return sha256_json(self.to_dict())[:20]

    @property
    def procedure_id(self):
        """Catch identical procedures despite cosmetic mechanism/evidence changes.

        Semantic paraphrases remain a model-assessed search concern, not a proof.
        """
        value = self.to_dict()
        normalized = {
            k: [" ".join(t.casefold().split()) for t in value[k]]
            for k in ("procedure", "stop_conditions")
        }
        normalized.update(
            trigger=" ".join(value["trigger"].casefold().split()),
            intensity=value["intensity"],
        )
        return sha256_json(normalized)

    def compile(self):
        value = self.to_dict()
        # Evidence identifiers and hypothesis never enter the native simulator overlay.
        body = {
            k: value[k]
            for k in (
                "mechanism",
                "trigger",
                "procedure",
                "intensity",
                "stop_conditions",
            )
        }
        return PromptStrategy(
            INVARIANTS
            + "\nCustomer procedure ("
            + PROTOCOL
            + "):\n"
            + json.dumps(body, ensure_ascii=False, sort_keys=True)
        )
