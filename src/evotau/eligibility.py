"""Ex-ante checks for the two task units in the minimal smoke."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WRITE_ACTIONS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
DECEPTION_PATTERNS = (
    re.compile(r"\b(?:wrong|incorrect|fake|false)\s+(?:order|id|number|identifier)\b", re.IGNORECASE),
    re.compile(r"\b(?:lie|deceiv\w*|mislead\w*|pretend to be)\b", re.IGNORECASE),
    re.compile(r"\bprovide\s+(?:the\s+)?(?:wrong|incorrect)\b", re.IGNORECASE),
)


class TaskEligibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SmokeTaskSelection:
    evolution_task_id: str
    validation_task_id: str
    evolution_entity_keys: tuple[str, ...]
    validation_entity_keys: tuple[str, ...]


def _scenario_text(task: Mapping[str, Any]) -> str:
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    if isinstance(instructions, str):
        return instructions
    return " ".join(
        str(instructions.get(key, ""))
        for key in ("task_instructions", "reason_for_call", "known_info", "unknown_info")
    )


def business_entity_keys(task: Mapping[str, Any]) -> tuple[str, ...]:
    """Build conservative, normalized keys for smoke E/V entity separation."""

    text = _scenario_text(task)
    keys: set[str] = set()
    for email in re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.IGNORECASE):
        keys.add(f"email:{email.casefold()}")
    for user_id in re.findall(r"\b[a-z]+(?:_[a-z]+)+_\d+\b", text, re.IGNORECASE):
        keys.add(f"user:{user_id.casefold()}")
    for order_id in re.findall(r"#?W\d{7}", text, re.IGNORECASE):
        keys.add(f"order:{order_id.upper().lstrip('#')}")
    for action in (task.get("evaluation_criteria") or {}).get("actions") or ():
        arguments = action.get("arguments") or {}
        for name, prefix in (("user_id", "user"), ("order_id", "order")):
            value = arguments.get(name)
            if value:
                normalized = str(value).casefold()
                if prefix == "order":
                    normalized = normalized.lstrip("#").upper()
                keys.add(f"{prefix}:{normalized}")
    name_match = re.search(
        r"\b(?:you are|you name is|my name is)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b",
        text,
    )
    zip_match = re.search(r"\bzip(?:\s*code)?\s*(?:is|:)?\s*(\d{5})\b", text, re.IGNORECASE)
    if name_match and zip_match:
        keys.add(
            f"person_zip:{name_match.group(1).casefold()}_{name_match.group(2).casefold()}_{zip_match.group(1)}"
        )
    return tuple(sorted(keys))


def _write_actions(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or ()
    return [action for action in actions if action.get("name") in WRITE_ACTIONS]


def _is_ex_ante_eligible(task: Mapping[str, Any], excluded_ids: set[str]) -> None:
    task_id = str(task.get("id", ""))
    if task_id in excluded_ids:
        raise TaskEligibilityError(f"task {task_id} is on the explicit exclusion list")
    if task.get("evaluation_criteria") is None:
        raise TaskEligibilityError(f"task {task_id} has no evaluation criteria")
    if not _write_actions(task):
        raise TaskEligibilityError(f"task {task_id} has no reference write action")
    text = _scenario_text(task)
    if any(pattern.search(text) for pattern in DECEPTION_PATTERNS):
        raise TaskEligibilityError(f"task {task_id} appears to require deceptive user behavior")


def validate_smoke_selection(
    tasks: Sequence[Mapping[str, Any]],
    split_data: Mapping[str, Sequence[str]],
    *,
    evolution_task_id: str,
    validation_task_id: str,
    excluded_task_ids: Sequence[str] = ("46", "47"),
) -> SmokeTaskSelection:
    """Check that E/V are eligible train tasks with different business entities."""

    if evolution_task_id == validation_task_id:
        raise TaskEligibilityError("E and V must use different task IDs")
    by_id = {str(task.get("id")): task for task in tasks}
    missing = {evolution_task_id, validation_task_id} - by_id.keys()
    if missing:
        raise TaskEligibilityError(f"selected task IDs are absent from the task set: {sorted(missing)}")
    train = {str(item) for item in split_data.get("train", ())}
    test = {str(item) for item in split_data.get("test", ())}
    selected = {evolution_task_id, validation_task_id}
    if not selected <= train:
        raise TaskEligibilityError("E and V must both belong to the official train split")
    if selected & test:
        raise TaskEligibilityError("E or V overlaps the official test split")
    excluded = {str(item) for item in excluded_task_ids}
    evolution = by_id[evolution_task_id]
    validation = by_id[validation_task_id]
    _is_ex_ante_eligible(evolution, excluded)
    _is_ex_ante_eligible(validation, excluded)

    e_actions = _write_actions(evolution)
    v_actions = _write_actions(validation)
    if not any(
        max(
            len((action.get("arguments") or {}).get("item_ids") or ()),
            len((action.get("arguments") or {}).get("new_item_ids") or ()),
        )
        >= 2
        for action in e_actions
    ):
        raise TaskEligibilityError("E must include a multi-item write target")
    if sum(
        len((action.get("arguments") or {}).get("item_ids") or ())
        + len((action.get("arguments") or {}).get("new_item_ids") or ())
        for action in v_actions
    ) != 2:
        raise TaskEligibilityError("V must be a single-item write validation task")

    e_entities = business_entity_keys(evolution)
    v_entities = business_entity_keys(validation)
    if not e_entities or not v_entities:
        raise TaskEligibilityError("could not derive stable business-entity keys for E and V")
    if set(e_entities) & set(v_entities):
        raise TaskEligibilityError("E and V share a customer or order entity")
    return SmokeTaskSelection(
        evolution_task_id=evolution_task_id,
        validation_task_id=validation_task_id,
        evolution_entity_keys=e_entities,
        validation_entity_keys=v_entities,
    )
