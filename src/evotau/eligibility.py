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


@dataclass(frozen=True, slots=True)
class GeneralizationTaskSelection:
    """A leakage-checked E/V/H task partition for multi-task pilot studies."""

    evolution_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    evolution_entity_keys: tuple[str, ...]
    validation_entity_keys: tuple[str, ...]
    heldout_entity_keys: tuple[str, ...]


def validate_activation_selection(
    tasks: Sequence[Mapping[str, Any]],
    split_data: Mapping[str, Sequence[str]],
    *,
    evolution_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
    excluded_task_ids: Sequence[str] = ("46", "47"),
) -> GeneralizationTaskSelection:
    """Validate a reviewed E/V activation panel while leaving H unopened."""

    evolution = _task_id_tuple(evolution_task_ids, "evolution")
    validation = _task_id_tuple(validation_task_ids, "validation")
    if len(evolution) != 10:
        raise TaskEligibilityError("activation smoke requires exactly ten E tasks")
    if not validation:
        raise TaskEligibilityError("activation smoke requires at least one reviewed V task")
    if set(evolution) & set(validation):
        raise TaskEligibilityError("activation E and V task IDs must be disjoint")

    by_id = {str(task.get("id")): task for task in tasks}
    if len(by_id) != len(tasks):
        raise TaskEligibilityError("task data contains duplicate task IDs")
    selected = evolution + validation
    missing = set(selected) - by_id.keys()
    if missing:
        raise TaskEligibilityError(
            f"selected activation task IDs are absent from the pinned task set: {sorted(missing)}"
        )
    train = {str(item) for item in split_data.get("train", ())}
    test = {str(item) for item in split_data.get("test", ())}
    if not set(selected) <= train or set(selected) & test:
        raise TaskEligibilityError("activation E/V tasks must be official-train and disjoint from test")

    excluded = {str(item) for item in excluded_task_ids}
    entities: dict[str, set[str]] = {"evolution": set(), "validation": set()}
    for panel, task_ids in (("evolution", evolution), ("validation", validation)):
        for task_id in task_ids:
            task = by_id[task_id]
            _is_ex_ante_eligible(task, excluded)
            keys = set(business_entity_keys(task))
            if not keys:
                raise TaskEligibilityError(
                    f"could not derive stable business-entity keys for {panel} task {task_id}"
                )
            entities[panel].update(keys)
    shared = entities["evolution"] & entities["validation"]
    if shared:
        raise TaskEligibilityError(
            f"evolution and validation tasks share business entities: {sorted(shared)}"
        )
    return GeneralizationTaskSelection(
        evolution_task_ids=evolution,
        validation_task_ids=validation,
        heldout_task_ids=(),
        evolution_entity_keys=tuple(sorted(entities["evolution"])),
        validation_entity_keys=tuple(sorted(entities["validation"])),
        heldout_entity_keys=(),
    )


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


def validate_generalization_selection(
    tasks: Sequence[Mapping[str, Any]],
    split_data: Mapping[str, Sequence[str]],
    *,
    evolution_task_ids: Sequence[str],
    validation_task_ids: Sequence[str],
    heldout_task_ids: Sequence[str],
    excluded_task_ids: Sequence[str] = ("46", "47"),
) -> GeneralizationTaskSelection:
    """Validate train E/V and official-test H without shared business entities.

    This is a conservative ID-based leakage check, not semantic task-family,
    policy/tool-conflict, or satisfiability adjudication. All selected tasks
    are checked ex ante; the function never drops an ineligible task and
    silently changes the requested sample. Those semantic judgments remain a
    required pre-run review.
    """

    groups = {
        "evolution": _task_id_tuple(evolution_task_ids, "evolution"),
        "validation": _task_id_tuple(validation_task_ids, "validation"),
        "heldout": _task_id_tuple(heldout_task_ids, "heldout"),
    }
    if any(not ids for ids in groups.values()):
        raise TaskEligibilityError("pilot E, V, and H selections must each be non-empty")
    all_ids = tuple(task_id for ids in groups.values() for task_id in ids)
    if len(set(all_ids)) != len(all_ids):
        raise TaskEligibilityError("E, V, and H task IDs must be unique and disjoint")

    by_id = {str(task.get("id")): task for task in tasks}
    if len(by_id) != len(tasks):
        raise TaskEligibilityError("task data contains duplicate task IDs")
    missing = set(all_ids) - by_id.keys()
    if missing:
        raise TaskEligibilityError(f"selected task IDs are absent from the task set: {sorted(missing)}")
    train = {str(item) for item in split_data.get("train", ())}
    test = {str(item) for item in split_data.get("test", ())}
    if not set(groups["evolution"] + groups["validation"]) <= train:
        raise TaskEligibilityError("E and V tasks must belong to the official train split")
    if not set(groups["heldout"]) <= test:
        raise TaskEligibilityError("H tasks must belong to the official test split")
    if set(all_ids) & (train & test):
        raise TaskEligibilityError("selected tasks cannot be in both official train and test splits")

    excluded = {str(item) for item in excluded_task_ids}
    entities: dict[str, set[str]] = {}
    for group_name, ids in groups.items():
        keys: set[str] = set()
        for task_id in ids:
            task = by_id[task_id]
            _is_ex_ante_eligible(task, excluded)
            task_keys = set(business_entity_keys(task))
            if not task_keys:
                raise TaskEligibilityError(
                    f"could not derive stable business-entity keys for {group_name} task {task_id}"
                )
            keys.update(task_keys)
        entities[group_name] = keys

    names = ("evolution", "validation", "heldout")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = entities[left] & entities[right]
            if shared:
                raise TaskEligibilityError(
                    f"{left} and {right} share business entities: {sorted(shared)}"
                )

    return GeneralizationTaskSelection(
        evolution_task_ids=groups["evolution"],
        validation_task_ids=groups["validation"],
        heldout_task_ids=groups["heldout"],
        evolution_entity_keys=tuple(sorted(entities["evolution"])),
        validation_entity_keys=tuple(sorted(entities["validation"])),
        heldout_entity_keys=tuple(sorted(entities["heldout"])),
    )


def _task_id_tuple(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} task IDs must be a sequence of IDs, not one string")
    result = tuple(str(value) for value in values)
    if any(not value.strip() for value in result):
        raise TaskEligibilityError(f"{label} task IDs must not be empty")
    if len(set(result)) != len(result):
        raise TaskEligibilityError(f"{label} task IDs must be unique")
    return result
