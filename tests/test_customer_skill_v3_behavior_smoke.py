from __future__ import annotations

import pytest

from evotau.customer_skill_v3_behavior_smoke import (
    validate_behavior_smoke_task_id,
    validate_inter_arm_cooldown_seconds,
)


def test_behavior_smoke_accepts_only_a_reviewed_e_task() -> None:
    assert validate_behavior_smoke_task_id("73", ("16", "22", "73")) == "73"


@pytest.mark.parametrize("task_id", ["93", "46", "999"])
def test_behavior_smoke_rejects_validation_excluded_or_unknown_task(task_id: str) -> None:
    with pytest.raises(ValueError, match="reviewed E panel"):
        validate_behavior_smoke_task_id(task_id, ("16", "22", "73"))


@pytest.mark.parametrize("seconds", [0, 65, 300])
def test_behavior_smoke_accepts_bounded_inter_arm_cooldown(seconds: int) -> None:
    assert validate_inter_arm_cooldown_seconds(seconds) == seconds


@pytest.mark.parametrize("seconds", [-1, 301, True, 1.5])
def test_behavior_smoke_rejects_invalid_inter_arm_cooldown(seconds: int) -> None:
    with pytest.raises(ValueError, match="integer from 0 to 300"):
        validate_inter_arm_cooldown_seconds(seconds)
