"""Bounded, open-ended Customer interaction-skill proposals (schema v3).

V3 changes only the EvoTau-owned Customer overlay. It does not accept task
content, benchmark answers, or raw trajectories as proposal input.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .customer_evolver import EvolutionFailureSignal
from .manifest import canonical_json, sha256_json
from .records import CandidateEvaluation, FailureRecord, customer_strategy_id
from .strategies import CustomerStrategy

MAX_ACTIVE_SKILLS = 1
MAX_TESTED_SKILL_MEMORY = 32
MAX_PROCEDURE_STEPS = 5
MAX_FIELD_CHARS = 420
MAX_RENDERED_TOKENS = 500
MAX_SKILL_SIMILARITY = 0.92

INVALIDITY_CATEGORIES = frozenset({
    "unsupported_factual_detail",
    "identity_inconsistency",
    "goal_alteration",
    "deception_or_impersonation",
    "policy_boundary_violation",
    "failure_to_answer_necessary_fact",
    "unknown",
})

_SKILL_FIELDS = frozenset({
    "name", "trigger", "procedure", "stop_conditions", "hypothesis", "evidence_refs",
})
_TASK_REFERENCE = re.compile(
    r"\b(?:task|scenario)\s*(?:id\s*)?(?:#|:)?\s*\d+[a-z-]*\b", re.IGNORECASE,
)
_LONG_IDENTIFIER = re.compile(r"(?<!\w)(?:#?[A-Z]{1,4}[-_]?)?\d{6,}(?!\w)", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_BENCHMARK_REFERENCE = re.compile(
    r"\b(?:tau[- ]?bench|tau2|tau3|split_tasks|gold actions?|reference answers?|"
    r"held[- ]?out tasks?|validation tasks?)\b|(?:tau[- ]?bench|tau2|tau3)",
    re.IGNORECASE,
)
_GUARANTEED_FAILURE = re.compile(
    r"\b(?:guarantee(?:d)?|certain(?:ly)?|will\s+fail|must\s+fail|ensure\s+(?:a\s+)?failure)\b|"
    r"(?:保证失败|必然失败|确保失败|导致失败)", re.IGNORECASE,
)
_PROHIBITED_BEHAVIOR = re.compile(
    r"\b(?:lie|deceiv\w*|impersonat\w*|fabricat\w*|pretend\s+(?:that|to)|"
    r"make\s+up\s+(?:a|an|the)?\s*(?:fact|detail|identifier|address|price|date)|"
    r"claim\s+(?:that\s+)?(?:the\s+)?(?:action|change|order|refund)\s+(?:was|has been|is)\s+(?:done|completed))\b|"
    r"(?:撒谎|欺骗|冒充|捏造|编造事实|谎称)", re.IGNORECASE,
)
_WITHHOLD_NECESSARY = re.compile(
    r"\b(?:withhold|refuse\s+to\s+(?:answer|provide|state)|never\s+(?:answer|provide|state)|"
    r"do\s+not\s+(?:answer|provide|state))\b.{0,100}\b(?:necessary|requested|truthful|fact|information)\b|"
    r"\b(?:never\s+answer|withhold)\b", re.IGNORECASE,
)
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_FORBIDDEN_CONTEXT_KEYS = frozenset({
    "task_id", "task_ids", "scenario", "user_scenario", "transcript", "trajectory",
    "customer_name", "address", "order_id", "product_id", "payment_information",
    "gold_actions", "reference_actions", "reference_answer", "expected_database_state",
    "heldout_tasks", "validation_tasks",
})


FIXED_CUSTOMER_INVARIANTS = """Fixed EvoTau Customer invariants (these always override the skill):
- Preserve every fixed task fact, the user's identity, account/order/product facts, and actual business goal.
- Answer directly requested necessary factual questions truthfully and follow tau-bench user-simulation guidelines and Retail policy boundaries.
- Never fabricate identifiers, addresses, prices, dates, product attributes, payment methods, or other facts.
- Never impersonate, knowingly deceive, or request an explicitly prohibited action to provoke a failure.
- Never claim that an action happened when it did not; never use task answers or gold actions.
- Treat this procedure as reusable: it must not encode task-specific identifiers or literal benchmark entities.
"""


SKILL_V3_SYSTEM_PROMPT = """You are the Customer Evolver in EvoTau.

Propose reusable interaction procedures for the EvoTau-owned Customer overlay. You may invent new procedures in natural language, but you may not rewrite the tau-bench prompt, task scenario, business policy, tools, or evaluator. The fixed invariants in the renderer are authoritative and are not part of the evolvable object.

Return exactly K candidates with the schema shown in the user context. Each skill has a short name, a reusable trigger, 2–5 ordered procedure steps, concise stop conditions, a falsifiable hypothesis, and evidence_refs. Supported structural operations are create, refine, and replace. Use create when the incumbent has no V3 skill; use refine or replace when it does. Names are labels, never strategy identity.

The supplied context is a sanitized, aggregate E-only reflection. It may include applicability/adherence counts, Customer-validity categories, task-completion counts, provisional/verified failure signals, and a disputed unsupported-factual-detail review signal. Treat disputed feedback as a caution only: it is not a verified Service failure and cannot affect fitness. Do not infer or claim a known Service weakness from exploration feedback. Without verified failure evidence, evidence_refs must be empty and hypothesis must start with "Exploration:".

Do not request or invent hidden facts. Do not mention or encode task IDs, names, addresses, order/product identifiers, payment information, benchmark entities, transcripts, task scenarios, gold actions, reference answers, expected database state, validation/H tasks, deception, or guaranteed failures. Cite only failure IDs explicitly present in verified_failure_summaries. Return JSON only; do not add dialogue or extra fields.
"""


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _display_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


@dataclass(frozen=True, slots=True)
class CustomerSkill:
    """One bounded, reusable V3 procedure; no fixed behavior axes are added."""

    name: str
    trigger: str
    procedure: tuple[str, ...]
    stop_conditions: str
    hypothesis: str
    evidence_refs: tuple[str, ...] = ()

    @property
    def is_skill_v3(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "skill": {
                "name": _display_text(self.name),
                "trigger": _display_text(self.trigger),
                "procedure": [_display_text(step) for step in self.procedure],
                "stop_conditions": _display_text(self.stop_conditions),
                "hypothesis": _display_text(self.hypothesis),
                "evidence_refs": list(self.evidence_refs),
            },
        }

    @property
    def canonical_content(self) -> dict[str, Any]:
        payload = self.to_dict()["skill"]
        return {
            "name": normalize_text(payload["name"]),
            "trigger": normalize_text(payload["trigger"]),
            "procedure": [normalize_text(step) for step in payload["procedure"]],
            "stop_conditions": normalize_text(payload["stop_conditions"]),
            "hypothesis": normalize_text(payload["hypothesis"]),
        }

    @property
    def strategy_id(self) -> str:
        return sha256_json({"schema_version": 3, "skill": self.canonical_content})


@dataclass(frozen=True, slots=True)
class CustomerSkillCandidate:
    strategy: CustomerStrategy | CustomerSkill
    parent_id: str
    operator: str
    rationale: str
    changed_fields: tuple[str, ...]
    expected_behavioral_effect: str
    supporting_failure_ids: tuple[str, ...]
    proposal_context_sha256: str
    proposal_response_sha256: str
    rendered_skill_sha256: str
    generation: int

    @property
    def strategy_id(self) -> str:
        return customer_strategy_id(self.strategy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_schema": "skill_v3",
            "strategy_id": self.strategy_id,
            "strategy": self.strategy.to_dict(),
            "parent_id": self.parent_id,
            "operator": self.operator,
            "rationale": self.rationale,
            "changed_fields": list(self.changed_fields),
            "expected_behavioral_effect": self.expected_behavioral_effect,
            "supporting_failure_ids": list(self.supporting_failure_ids),
            "proposal_context_sha256": self.proposal_context_sha256,
            "proposal_response_sha256": self.proposal_response_sha256,
            "rendered_skill_sha256": self.rendered_skill_sha256,
            "generation": self.generation,
        }


@dataclass(frozen=True, slots=True)
class CustomerSkillProposalInput:
    generation: int
    incumbent: CustomerStrategy | CustomerSkill
    candidate_count: int
    proposal_seed: int
    tested_skill_fingerprints: tuple[str, ...]
    tested_skills: tuple[CustomerSkill, ...]
    tested_skill_memory: tuple[Mapping[str, Any], ...]
    verified_failures: tuple[EvolutionFailureSignal, ...]
    reflection_feedback: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("V3 proposal generation must be non-negative")
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise ValueError("V3 proposal K must be positive")
        if type(self.proposal_seed) is not int or self.proposal_seed < 0:
            raise ValueError("V3 proposal seed must be non-negative")
        if len(self.tested_skills) > MAX_TESTED_SKILL_MEMORY or len(self.tested_skill_memory) > MAX_TESTED_SKILL_MEMORY:
            raise ValueError("V3 tested-skill memory exceeds its fixed bound")
        if self.tested_skill_fingerprints != tuple(item.strategy_id for item in self.tested_skills):
            raise ValueError("V3 tested-skill fingerprints must match canonical skill content")
        expected_memory_fields = {
            "skill_id", "skill", "parent_id", "generation", "applicability_count",
            "adherence_count", "customer_valid_count", "provisional_failure_count",
            "verified_failure_refs", "invalidity_categories",
        }
        memory_ids: set[str] = set()
        for item in self.tested_skill_memory:
            if not isinstance(item, Mapping) or set(item) != expected_memory_fields:
                raise ValueError("V3 tested-skill memory has unsupported fields")
            if not re.fullmatch(r"[0-9a-f]{64}", str(item["skill_id"])):
                raise ValueError("V3 tested-skill memory IDs must be canonical SHA-256 values")
            if item["skill_id"] in memory_ids:
                raise ValueError("V3 tested-skill memory cannot repeat a skill ID")
            memory_ids.add(item["skill_id"])
            if not re.fullmatch(r"(?:[0-9a-f]{16}|[0-9a-f]{64})", str(item["parent_id"])):
                raise ValueError("V3 tested-skill memory parent ID is malformed")
            if type(item["generation"]) is not int or item["generation"] < 0:
                raise ValueError("V3 tested-skill memory generation is invalid")
            skill_data = item["skill"]
            if not isinstance(skill_data, Mapping) or set(skill_data) != _SKILL_FIELDS:
                raise ValueError("V3 tested-skill memory contains an invalid reusable skill representation")
            historical_skill = validate_skill(
                skill_data,
                incumbent=CustomerStrategy.v2_baseline(),
                operation="create",
                verified_failure_ids=tuple(item["verified_failure_refs"]),
            )
            if historical_skill.strategy_id != item["skill_id"]:
                raise ValueError("V3 tested-skill memory ID differs from canonical skill content")
            for key in (
                "applicability_count", "adherence_count", "customer_valid_count",
                "provisional_failure_count",
            ):
                if type(item[key]) is not int or item[key] < 0:
                    raise ValueError("V3 tested-skill memory outcome counts must be non-negative integers")
            if (not isinstance(item["verified_failure_refs"], list)
                    or any(not isinstance(ref, str) or not ref for ref in item["verified_failure_refs"])):
                raise ValueError("V3 tested-skill memory failure refs must be strings")
            if not isinstance(item["invalidity_categories"], Mapping) or any(
                key not in INVALIDITY_CATEGORIES or type(value) is not int or value < 0
                for key, value in item["invalidity_categories"].items()
            ):
                raise ValueError("V3 tested-skill memory invalidity counts are malformed")
        _assert_sanitized_context({
            "tested_skills": [item.to_dict()["skill"] for item in self.tested_skills],
            "tested_skill_memory": [dict(item) for item in self.tested_skill_memory],
            "verified_failure_summaries": [item.to_dict() for item in self.verified_failures],
            "reflection_feedback": dict(self.reflection_feedback),
        })

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 3,
            "generation": self.generation,
            "K": self.candidate_count,
            "proposal_seed": self.proposal_seed,
            "incumbent": self.incumbent.to_dict(),
            "tested_skill_fingerprints": list(self.tested_skill_fingerprints),
            "tested_skills": [item.to_dict()["skill"] for item in self.tested_skills],
            "tested_skill_memory": [dict(item) for item in self.tested_skill_memory],
            "verified_failure_summaries": [item.to_dict() for item in self.verified_failures],
            "reflection_feedback": dict(self.reflection_feedback),
            "response_schema": {
                "candidates": [{
                    "operation": "create|refine|replace",
                    "skill": {
                        "name": "short reusable label",
                        "trigger": "general interaction condition",
                        "procedure": ["ordered step 1", "ordered step 2"],
                        "stop_conditions": "when the reusable procedure ends",
                        "hypothesis": "Exploration: falsifiable behavior-level prediction",
                        "evidence_refs": [],
                    },
                }],
            },
        }
        _assert_sanitized_context(payload)
        return payload


@dataclass(frozen=True, slots=True)
class CustomerSkillProviderResponse:
    payload: Mapping[str, Any]
    response_sha256: str


def render_customer_skill_v3(skill: CustomerSkill | None) -> str:
    """Render fixed invariants plus at most one reusable interaction procedure."""

    if skill is None:
        return ""
    rows = [
        "<evotau_customer_skill>",
        FIXED_CUSTOMER_INVARIANTS.rstrip(),
        f"Reusable skill trigger: {_display_text(skill.trigger)}",
        "Procedure:",
        *(f"{index}. {_display_text(step)}" for index, step in enumerate(skill.procedure, 1)),
        f"Stop conditions: {_display_text(skill.stop_conditions)}",
        "</evotau_customer_skill>",
    ]
    rendered = "\n".join(rows)
    _validate_rendered_bound(rendered)
    return rendered


def validate_skill(
    raw: Any,
    *,
    incumbent: CustomerStrategy | CustomerSkill,
    operation: Any,
    verified_failure_ids: Sequence[str],
    forbidden_literals: Sequence[str] = (),
) -> CustomerSkill:
    """Fail closed on malformed, duplicate, unsafe, or task-specific skills."""

    if not isinstance(raw, Mapping) or set(raw) != _SKILL_FIELDS:
        raise ValueError("V3 skill must contain exactly the six skill fields")
    if operation not in {"create", "refine", "replace"}:
        raise ValueError("V3 operation must be create, refine, or replace")
    if isinstance(incumbent, CustomerSkill):
        if operation == "create":
            raise ValueError("create is valid only when the incumbent has no V3 skill")
    elif operation != "create":
        raise ValueError("refine or replace requires a V3 skill incumbent")

    text_fields = ("name", "trigger", "stop_conditions", "hypothesis")
    normalized: dict[str, Any] = {}
    for field in text_fields:
        value = raw[field]
        if not isinstance(value, str):
            raise TypeError(f"V3 skill {field} must be text")
        normalized_value = _display_text(value)
        if normalized_value != value:
            raise ValueError(f"V3 skill {field} must use canonical single-line formatting")
        value = normalized_value
        if not value or len(value) > MAX_FIELD_CHARS:
            raise ValueError(f"V3 skill {field} must be non-empty and at most {MAX_FIELD_CHARS} characters")
        if "<" in value or ">" in value:
            raise ValueError("V3 skill text cannot inject renderer tags")
        normalized[field] = value

    procedure = raw["procedure"]
    if not isinstance(procedure, list) or not 2 <= len(procedure) <= MAX_PROCEDURE_STEPS:
        raise ValueError(f"V3 procedure must contain 2..{MAX_PROCEDURE_STEPS} ordered steps")
    steps: list[str] = []
    for index, step in enumerate(procedure):
        if not isinstance(step, str):
            raise TypeError(f"V3 procedure step {index + 1} must be text")
        normalized_step = _display_text(step)
        if normalized_step != step:
            raise ValueError(f"V3 procedure step {index + 1} must use canonical single-line formatting")
        step = normalized_step
        if not step or len(step) > MAX_FIELD_CHARS or "<" in step or ">" in step:
            raise ValueError(f"V3 procedure step {index + 1} is empty, too long, or contains markup")
        steps.append(step)

    refs = raw["evidence_refs"]
    if (not isinstance(refs, list) or any(not isinstance(item, str) or not item for item in refs)
            or len(refs) != len(set(refs))):
        raise ValueError("V3 evidence_refs must be a duplicate-free list of non-empty strings")
    if not set(refs) <= set(verified_failure_ids):
        raise ValueError("V3 evidence_refs may cite only supplied verified failure IDs")
    if not normalized["hypothesis"].strip():
        raise ValueError("V3 skill hypothesis must be a falsifiable non-empty statement")
    if not refs and not normalized["hypothesis"].casefold().startswith("exploration:"):
        raise ValueError("unsupported V3 hypotheses must begin with Exploration:")

    skill = CustomerSkill(
        name=normalized["name"], trigger=normalized["trigger"], procedure=tuple(steps),
        stop_conditions=normalized["stop_conditions"], hypothesis=normalized["hypothesis"],
        evidence_refs=tuple(refs),
    )
    rendered = render_customer_skill_v3(skill)
    candidate_text = "\n".join((
        skill.name, skill.trigger, *skill.procedure, skill.stop_conditions, skill.hypothesis,
    ))
    _validate_safe_candidate_text(candidate_text, forbidden_literals=forbidden_literals)
    _validate_rendered_bound(rendered)
    if isinstance(incumbent, CustomerSkill) and _near_duplicate(skill, incumbent):
        raise ValueError("V3 skill duplicates or nearly matches the incumbent")
    return skill


def _validate_safe_candidate_text(value: str, *, forbidden_literals: Sequence[str]) -> None:
    if _TASK_REFERENCE.search(value):
        raise ValueError("V3 skill contains a task or scenario identifier")
    if _LONG_IDENTIFIER.search(value) or _EMAIL.search(value) or _BENCHMARK_REFERENCE.search(value):
        raise ValueError("V3 skill contains a likely benchmark identifier or email")
    if _GUARANTEED_FAILURE.search(value):
        raise ValueError("V3 skill cannot guarantee a failure")
    if _PROHIBITED_BEHAVIOR.search(value):
        raise ValueError("V3 skill conflicts with fixed truthfulness and policy invariants")
    if _WITHHOLD_NECESSARY.search(value):
        raise ValueError("V3 skill cannot withhold a directly requested necessary fact")
    for literal in forbidden_literals:
        if not isinstance(literal, str) or not literal.strip():
            continue
        literal = _display_text(literal)
        if re.search(rf"(?<!\w){re.escape(literal)}(?!\w)", value, flags=re.IGNORECASE):
            raise ValueError("V3 skill contains a literal from a frozen benchmark scenario")


def _validate_rendered_bound(rendered: str) -> None:
    if len(_TOKEN.findall(rendered)) > MAX_RENDERED_TOKENS:
        raise ValueError(f"rendered V3 skill exceeds {MAX_RENDERED_TOKENS} tokens")


def _near_duplicate(left: CustomerSkill, right: CustomerSkill) -> bool:
    left_text = normalize_text(" ".join((left.trigger, *left.procedure, left.stop_conditions)))
    right_text = normalize_text(" ".join((right.trigger, *right.procedure, right.stop_conditions)))
    if left_text == right_text:
        return True
    return SequenceMatcher(a=left_text, b=right_text, autojunk=False).ratio() >= MAX_SKILL_SIMILARITY


def _assert_sanitized_context(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_CONTEXT_KEYS:
                raise ValueError(f"forbidden key in sanitized Evolver context: {key}")
            _assert_sanitized_context(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_sanitized_context(item)
    elif isinstance(value, str) and (
        _TASK_REFERENCE.search(value) or _EMAIL.search(value) or _LONG_IDENTIFIER.search(value)
    ):
        raise ValueError("sanitized Evolver context contains task-specific or literal identifiers")


def sanitized_reflection_context(
    evaluation: CandidateEvaluation,
    *,
    prior_signals: Mapping[str, Any] | None = None,
    verified_failures: Sequence[FailureRecord] = (),
) -> dict[str, Any]:
    """Aggregate E-only records to behavior-level counts without task/episode data."""

    episodes = tuple(evaluation.episodes)
    invalidity_counts: dict[str, int] = {}
    for item in episodes:
        category = getattr(item, "customer_invalidity_category", None)
        if category is not None:
            if category not in INVALIDITY_CATEGORIES:
                raise ValueError("episode contains an unsupported Customer invalidity category")
            invalidity_counts[category] = invalidity_counts.get(category, 0) + 1
    categories = {
        "unsupported_factual_detail": 0,
        "identity_inconsistency": 0,
        "goal_alteration": 0,
        "deception_or_impersonation": 0,
        "policy_boundary_violation": 0,
        "failure_to_answer_necessary_fact": 0,
        "unknown": 0,
    }
    categories.update(invalidity_counts)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "aggregate": {
            "episode_count": len(episodes),
            "customer_valid_count": sum(item.customer_valid is True for item in episodes),
            "applicable_count": sum(item.strategy_applicable is True for item in episodes),
            "adherent_count": sum(item.customer_strategy_adherent is True for item in episodes),
            "service_completed_count": sum(item.task_success is True for item in episodes),
            "provisional_service_failure_count": evaluation.provisional_failure_count,
            "invalidity_category_counts": categories,
        },
        "prior_review_disagreements": [],
        "verified_failure_summaries": [],
    }
    if prior_signals is not None:
        payload["prior_review_disagreements"] = _validate_prior_signals(prior_signals)
    signals = tuple(sorted(verified_failures, key=lambda item: item.failure_id))
    payload["verified_failure_summaries"] = [
        {
            "failure_id": item.failure_id,
            "workflow_stage": item.signature.workflow_stage,
            "policy_rule_id": item.signature.policy_rule_id,
            "mistake_type": item.signature.mistake_type,
        }
        for item in signals
    ]
    _assert_sanitized_context(payload)
    return payload


def _validate_prior_signals(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(value) != {"schema_version", "signals"} or value.get("schema_version") != 1:
        raise ValueError("prior reflection signal artifact has an unsupported schema")
    signals = value.get("signals")
    if not isinstance(signals, list):
        raise TypeError("prior reflection signals must be a list")
    output = []
    for item in signals:
        expected = {"category", "adjudication", "fitness_eligible", "source_fingerprint"}
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError("prior reflection signal has unsupported fields")
        if item["category"] not in INVALIDITY_CATEGORIES:
            raise ValueError("prior reflection signal has an unknown invalidity category")
        if item["adjudication"] not in {"confirmed", "disputed"}:
            raise ValueError("prior reflection signal adjudication must be confirmed or disputed")
        if type(item["fitness_eligible"]) is not bool:
            raise ValueError("prior reflection signal fitness_eligible must be boolean")
        if item["fitness_eligible"]:
            raise ValueError("Customer invalidity reflection signals can never be fitness")
        fingerprint = item["source_fingerprint"]
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("prior reflection signal source fingerprint must be a SHA-256")
        output.append({
            "category": item["category"],
            "adjudication": item["adjudication"],
            "fitness_eligible": False,
            "count": 1,
        })
    return output


def validate_reflection_seed_signals(value: Any) -> dict[str, Any]:
    """Validate and normalize a shared, task-free reflection seed artifact."""

    if not isinstance(value, Mapping) or set(value) != {"schema_version", "signals"}:
        raise ValueError("reflection seed must be a JSON object")
    normalized = {"schema_version": value["schema_version"], "signals": value["signals"]}
    _validate_prior_signals(normalized)
    return normalized


def extract_forbidden_literals(tasks: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract scenario identifiers locally for validation; never put them in Evolver input."""

    literals: set[str] = {str(task_id) for task_id in tasks}
    patterns = (
        _EMAIL,
        re.compile(r"(?<!\w)(?:#?[A-Z]{1,4}[-_]?)?\d{4,}(?!\w)", re.IGNORECASE),
        re.compile(r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
                   r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b", re.IGNORECASE),
    )
    name_patterns = (
        re.compile(r"\b(?:my name is|your name is|name is|i am|i'm)\s+"
                   r"([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+){0,2})", re.IGNORECASE),
    )
    for task in tasks.values():
        scenario = getattr(task, "user_scenario", None)
        if hasattr(scenario, "model_dump"):
            scenario = scenario.model_dump(mode="json")
        elif hasattr(scenario, "dict") and callable(scenario.dict):
            scenario = scenario.dict()
        text = json.dumps(scenario, ensure_ascii=False, sort_keys=True, default=str)
        for pattern in patterns:
            literals.update(match.group(0) for match in pattern.finditer(text))
        for pattern in name_patterns:
            literals.update(match.group(1) for match in pattern.finditer(text))
    return tuple(sorted((item for item in literals if item.strip()), key=str.casefold))


def propose_customer_skills(
    incumbent: CustomerStrategy | CustomerSkill,
    count: int,
    *,
    generation: int,
    seed: int,
    recent_failures: Sequence[FailureRecord],
    already_seen: Sequence[str],
    already_tested_skills: Sequence[CustomerSkill],
    tested_skill_memory: Sequence[Mapping[str, Any]] = (),
    reflection_feedback: Mapping[str, Any],
    forbidden_literals: Sequence[str],
    proposal_provider: Any,
) -> tuple[CustomerSkillCandidate, ...]:
    if not callable(proposal_provider):
        raise TypeError("V3 Customer skill provider must be callable")
    if not (isinstance(incumbent, CustomerSkill) or incumbent.is_v2):
        raise ValueError("skill_v3 requires a V3 skill or the matched seven-axis baseline")
    if (type(count) is not int or count < 1 or type(generation) is not int or generation < 0
            or type(seed) is not int or seed < 0):
        raise ValueError("V3 generation, K, and seed must be valid")
    failure_ids = {item.failure_id for item in recent_failures}
    signals = tuple(
        EvolutionFailureSignal(
            failure_id=item.failure_id,
            workflow_stage=item.signature.workflow_stage,
            policy_rule_id=item.signature.policy_rule_id,
            mistake_type=item.signature.mistake_type,
        )
        for item in sorted(recent_failures, key=lambda item: item.failure_id)
    )
    tested = tuple(sorted(already_tested_skills, key=lambda item: item.strategy_id))[-MAX_TESTED_SKILL_MEMORY:]
    memory = tuple(dict(item) for item in tested_skill_memory)[-MAX_TESTED_SKILL_MEMORY:]
    context = CustomerSkillProposalInput(
        generation=generation,
        incumbent=incumbent,
        candidate_count=count,
        proposal_seed=seed,
        tested_skill_fingerprints=tuple(item.strategy_id for item in tested),
        tested_skills=tested,
        tested_skill_memory=memory,
        verified_failures=signals,
        reflection_feedback=dict(reflection_feedback),
    )
    response = proposal_provider(context)
    if isinstance(response, CustomerSkillProviderResponse):
        result = response.payload
        response_hash = response.response_sha256
    else:
        result = response
        response_hash = sha256_json(result)
    if not re.fullmatch(r"[0-9a-f]{64}", response_hash):
        raise ValueError("V3 provider response fingerprint must be a SHA-256")
    if not isinstance(result, Mapping) or set(result) != {"candidates"}:
        raise ValueError("V3 Evolver response must contain only candidates")
    rows = result["candidates"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("V3 Evolver must return exactly K candidates")
    parent_id = customer_strategy_id(incumbent)
    context_hash = sha256_json(context.to_dict())
    seen = set(already_seen) | {parent_id}
    seen.update(item.strategy_id for item in tested)
    candidates: list[CustomerSkillCandidate] = []
    candidate_skills: list[CustomerSkill] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"operation", "skill"}:
            raise ValueError(f"V3 candidate {index} must contain exactly operation and skill")
        operation = row["operation"]
        skill = validate_skill(
            row["skill"], incumbent=incumbent, operation=operation,
            verified_failure_ids=tuple(failure_ids), forbidden_literals=forbidden_literals,
        )
        if skill.strategy_id in seen:
            raise ValueError("V3 Evolver returned an incumbent, tested, or duplicate skill")
        if any(_near_duplicate(skill, prior) for prior in tested):
            raise ValueError("V3 Evolver returned a skill too similar to a previously tested skill")
        if any(_near_duplicate(skill, item) for item in candidate_skills):
            raise ValueError("V3 Evolver returned duplicate or nearly identical skills")
        seen.add(skill.strategy_id)
        candidate_skills.append(skill)
        rendered_hash = hashlib.sha256(render_customer_skill_v3(skill).encode("utf-8")).hexdigest()
        candidates.append(CustomerSkillCandidate(
            strategy=skill,
            parent_id=parent_id,
            operator=operation,
            rationale="failure_conditioned" if skill.evidence_refs else "exploration",
            changed_fields=("skill",),
            expected_behavioral_effect=skill.hypothesis,
            supporting_failure_ids=skill.evidence_refs,
            proposal_context_sha256=context_hash,
            proposal_response_sha256=response_hash,
            rendered_skill_sha256=rendered_hash,
            generation=generation,
        ))
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class LLMCustomerSkillEvolver:
    model: str
    model_args: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("V3 Customer Evolver requires a frozen model ID")
        if "temperature" not in self.model_args:
            raise ValueError("V3 Customer Evolver model arguments must freeze temperature")

    def __call__(self, context: CustomerSkillProposalInput) -> CustomerSkillProviderResponse:
        from tau2.data_model.message import SystemMessage, UserMessage
        from tau2.utils.llm_utils import generate

        serialized = canonical_json(context.to_dict())
        message = generate(
            model=self.model,
            messages=[
                SystemMessage(role="system", content=SKILL_V3_SYSTEM_PROMPT),
                UserMessage(role="user", content=serialized),
            ],
            call_name="evotau_customer_skill_evolver_v3",
            num_retries=0,
            **dict(self.model_args),
        )
        content = message.content or ""
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"V3 Evolver returned invalid JSON (response_sha256={hashlib.sha256(content.encode()).hexdigest()})"
            ) from exc
        if not isinstance(payload, dict):
            raise TypeError("V3 Evolver response must be a JSON object")
        return CustomerSkillProviderResponse(
            payload=payload,
            response_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
