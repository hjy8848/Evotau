"""Immutable, local-mutation Service SkillMemory carrier for EvoTau V1."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_SKILL_ID = re.compile(r"^skill-(\d{4,})$")


def _non_empty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _skill_id_number(value: str) -> int:
    match = _SKILL_ID.fullmatch(value)
    if match is None:
        raise ValueError("skill_id must use the stable skill-0001 format")
    return int(match.group(1))


def _skill_order(skill: ServiceSkill) -> tuple[int, str]:
    return (_skill_id_number(skill.skill_id), skill.skill_id)


@dataclass(frozen=True, slots=True)
class ServiceSkill:
    """Runtime-visible procedural guidance; provenance is stored separately."""

    skill_id: str
    trigger: str
    guidance: str

    def __post_init__(self) -> None:
        _skill_id_number(self.skill_id)
        _non_empty_text(self.trigger, "trigger")
        _non_empty_text(self.guidance, "guidance")

    def to_dict(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "trigger": self.trigger,
            "guidance": self.guidance,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServiceSkill:
        if set(value) != {"skill_id", "trigger", "guidance"}:
            raise ValueError(
                "a Service skill requires exactly skill_id, trigger, and guidance"
            )
        return cls(
            skill_id=value["skill_id"],
            trigger=value["trigger"],
            guidance=value["guidance"],
        )


@dataclass(frozen=True, slots=True)
class ServiceSkillDraft:
    """Evolver-proposed skill content. EvoTau, not the model, assigns identity."""

    trigger: str
    guidance: str

    def __post_init__(self) -> None:
        _non_empty_text(self.trigger, "trigger")
        _non_empty_text(self.guidance, "guidance")

    def to_dict(self) -> dict[str, str]:
        return {"trigger": self.trigger, "guidance": self.guidance}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServiceSkillDraft:
        if set(value) != {"trigger", "guidance"}:
            raise ValueError("a proposed skill requires exactly trigger and guidance")
        return cls(trigger=value["trigger"], guidance=value["guidance"])


@dataclass(frozen=True, slots=True)
class ServiceSkillMemory:
    """Canonical immutable collection of all active Service skills."""

    skills: tuple[ServiceSkill, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.skills, tuple):
            raise TypeError("ServiceSkillMemory.skills must be a tuple")
        if any(not isinstance(item, ServiceSkill) for item in self.skills):
            raise TypeError(
                "ServiceSkillMemory.skills must contain ServiceSkill values"
            )
        identifiers = [item.skill_id for item in self.skills]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ServiceSkillMemory skill IDs must be unique")
        object.__setattr__(self, "skills", tuple(sorted(self.skills, key=_skill_order)))

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        return {"skills": [skill.to_dict() for skill in self.skills]}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServiceSkillMemory:
        if set(value) != {"skills"} or not isinstance(value.get("skills"), list):
            raise ValueError("a ServiceSkillMemory requires exactly a skills array")
        return cls(tuple(ServiceSkill.from_mapping(item) for item in value["skills"]))


class ServiceMutationOperation(StrEnum):
    ADD = "add"
    UPDATE = "update"
    NO_OP = "no_op"


@dataclass(frozen=True, slots=True)
class ServiceSkillMutation:
    """One strict ADD, UPDATE, or NO_OP proposal from the Service Evolver."""

    analysis: str
    operation: ServiceMutationOperation
    target_skill_id: str | None
    skill: ServiceSkillDraft | None

    def __post_init__(self) -> None:
        _non_empty_text(self.analysis, "mutation analysis")
        if not isinstance(self.operation, ServiceMutationOperation):
            try:
                object.__setattr__(
                    self, "operation", ServiceMutationOperation(self.operation)
                )
            except ValueError as exc:
                raise ValueError("operation must be add, update, or no_op") from exc
        if self.operation == ServiceMutationOperation.NO_OP:
            if self.target_skill_id is not None or self.skill is not None:
                raise ValueError("NO_OP requires null target_skill_id and null skill")
        elif self.operation == ServiceMutationOperation.ADD:
            if self.target_skill_id is not None:
                raise ValueError("ADD requires null target_skill_id")
            if not isinstance(self.skill, ServiceSkillDraft):
                raise ValueError("ADD requires a trigger/guidance skill payload")
        elif self.operation == ServiceMutationOperation.UPDATE:
            if not isinstance(self.target_skill_id, str):
                raise ValueError("UPDATE requires a target_skill_id")
            _skill_id_number(self.target_skill_id)
            if not isinstance(self.skill, ServiceSkillDraft):
                raise ValueError("UPDATE requires a trigger/guidance skill payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "operation": self.operation.value,
            "target_skill_id": self.target_skill_id,
            "skill": None if self.skill is None else self.skill.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServiceSkillMutation:
        expected = {"analysis", "operation", "target_skill_id", "skill"}
        if set(value) != expected:
            raise ValueError(
                "Service mutation requires exactly analysis, operation, target_skill_id, and skill"
            )
        try:
            operation = ServiceMutationOperation(value["operation"])
        except (TypeError, ValueError) as exc:
            raise ValueError("operation must be add, update, or no_op") from exc
        payload = value["skill"]
        skill = None if payload is None else ServiceSkillDraft.from_mapping(payload)
        return cls(
            analysis=value["analysis"],
            operation=operation,
            target_skill_id=value["target_skill_id"],
            skill=skill,
        )


@dataclass(frozen=True, slots=True)
class ServiceSkillProvenance:
    """Bookkeeping kept outside runtime-visible ServiceSkillMemory."""

    skill_id: str
    created_generation: int
    updated_generation: int
    source_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _skill_id_number(self.skill_id)
        if (
            type(self.created_generation) is not int
            or self.created_generation < 0
            or type(self.updated_generation) is not int
            or self.updated_generation < self.created_generation
        ):
            raise ValueError(
                "skill provenance generations must be ordered non-negative integers"
            )
        if not isinstance(self.source_task_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.source_task_ids
        ):
            raise ValueError("source_task_ids must be a tuple of non-empty task IDs")
        object.__setattr__(
            self, "source_task_ids", tuple(dict.fromkeys(self.source_task_ids))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "created_generation": self.created_generation,
            "updated_generation": self.updated_generation,
            "source_task_ids": list(self.source_task_ids),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ServiceSkillProvenance:
        return cls(
            skill_id=value["skill_id"],
            created_generation=value["created_generation"],
            updated_generation=value["updated_generation"],
            source_task_ids=tuple(value["source_task_ids"]),
        )


def next_service_skill_id(memory: ServiceSkillMemory) -> str:
    """Assign the next deterministic monotonically increasing ID; never hash text."""

    next_number = (
        max((_skill_id_number(item.skill_id) for item in memory.skills), default=0) + 1
    )
    return f"skill-{next_number:04d}"


def service_skill_id_high_watermark(memory: ServiceSkillMemory) -> int:
    """Return the highest active numeric ID; run checkpoints also retain rejected reservations."""

    return max((_skill_id_number(item.skill_id) for item in memory.skills), default=0)


def apply_skill_mutation(
    memory: ServiceSkillMemory,
    mutation: ServiceSkillMutation,
    *,
    next_skill_id_number: int | None = None,
) -> ServiceSkillMemory:
    """Apply one pure local mutation while preserving every unrelated skill."""

    if mutation.operation == ServiceMutationOperation.NO_OP:
        return memory
    if mutation.operation == ServiceMutationOperation.ADD:
        assert mutation.skill is not None
        next_number = (
            service_skill_id_high_watermark(memory) + 1
            if next_skill_id_number is None
            else next_skill_id_number
        )
        if type(
            next_number
        ) is not int or next_number <= service_skill_id_high_watermark(memory):
            raise ValueError(
                "next_skill_id_number must exceed every active SkillMemory ID"
            )
        new_skill = ServiceSkill(
            skill_id=f"skill-{next_number:04d}",
            trigger=mutation.skill.trigger,
            guidance=mutation.skill.guidance,
        )
        return ServiceSkillMemory((*memory.skills, new_skill))
    assert mutation.operation == ServiceMutationOperation.UPDATE
    assert mutation.skill is not None
    if not any(item.skill_id == mutation.target_skill_id for item in memory.skills):
        raise ValueError(f"UPDATE target does not exist: {mutation.target_skill_id}")
    updated = tuple(
        ServiceSkill(
            skill_id=item.skill_id,
            trigger=mutation.skill.trigger,
            guidance=mutation.skill.guidance,
        )
        if item.skill_id == mutation.target_skill_id
        else item
        for item in memory.skills
    )
    return ServiceSkillMemory(updated)


def render_service_skill_memory(memory: ServiceSkillMemory) -> str:
    """Deterministically render every active skill without provenance or retrieval."""

    if not memory.skills:
        return ""
    sections = [
        "# Learned Service Skills",
        (
            "These are reusable behavioral skills learned from previous interactions. "
            "Apply a skill only when its trigger matches the current situation."
        ),
        (
            "Authority order: native task policy, the current user request within policy, "
            "tool/backend evidence, then learned skill guidance. Learned skills are procedural "
            "guidance, never new business policy."
        ),
        (
            "If multiple skills apply, use only guidance that is mutually compatible and "
            "consistent with the native policy and current task evidence. The native policy wins."
        ),
        "Do not invent facts or actions unsupported by the conversation, policy, backend, or tool results.",
    ]
    for skill in memory.skills:
        sections.extend(
            (
                f"## {skill.skill_id}",
                "Trigger:",
                skill.trigger.strip(),
                "Guidance:",
                skill.guidance.strip(),
            )
        )
    return "\n\n".join(sections)


def update_skill_provenance(
    provenance: Sequence[ServiceSkillProvenance],
    *,
    before: ServiceSkillMemory,
    after: ServiceSkillMemory,
    mutation: ServiceSkillMutation,
    generation: int,
    source_task_ids: Sequence[str],
) -> tuple[ServiceSkillProvenance, ...]:
    """Update experiment bookkeeping separately from the runtime memory object."""

    if mutation.operation == ServiceMutationOperation.NO_OP or before == after:
        return tuple(provenance)
    by_id = {item.skill_id: item for item in provenance}
    task_ids = tuple(dict.fromkeys(str(item) for item in source_task_ids))
    if mutation.operation == ServiceMutationOperation.ADD:
        added = next(
            item
            for item in after.skills
            if item.skill_id not in {x.skill_id for x in before.skills}
        )
        if added.skill_id in by_id:
            raise ValueError("ADD attempted to reuse a provenance skill ID")
        by_id[added.skill_id] = ServiceSkillProvenance(
            added.skill_id,
            generation,
            generation,
            task_ids,
        )
    elif mutation.operation == ServiceMutationOperation.UPDATE:
        current = by_id.get(str(mutation.target_skill_id))
        if current is None:
            raise ValueError("UPDATE target is missing provenance")
        by_id[current.skill_id] = ServiceSkillProvenance(
            skill_id=current.skill_id,
            created_generation=current.created_generation,
            updated_generation=generation,
            source_task_ids=tuple(dict.fromkeys((*current.source_task_ids, *task_ids))),
        )
    return tuple(
        sorted(
            by_id.values(),
            key=lambda item: _skill_order(
                next(
                    skill for skill in after.skills if skill.skill_id == item.skill_id
                ),
            ),
        )
    )
