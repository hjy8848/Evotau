from __future__ import annotations

import pytest

from evotau.customer_skill_v3_behavior_smoke import validate_behavior_smoke_task_id


def test_behavior_smoke_accepts_only_a_reviewed_e_task() -> None:
    assert validate_behavior_smoke_task_id("73", ("16", "22", "73")) == "73"


@pytest.mark.parametrize("task_id", ["93", "46", "999"])
def test_behavior_smoke_rejects_validation_excluded_or_unknown_task(task_id: str) -> None:
    with pytest.raises(ValueError, match="reviewed E panel"):
        validate_behavior_smoke_task_id(task_id, ("16", "22", "73"))
