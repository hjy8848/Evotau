"""Small records for native τ-bench episodes and prompt strategies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any

from .tau_provenance import sha256_json


class EpisodeStatus(StrEnum):
    COMPLETE = "complete"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One native τ-bench simulation and its visible evaluation signals."""

    episode_id: str
    task_id: str
    seed: int
    customer_strategy_id: str
    service_strategy_id: str
    status: EpisodeStatus
    task_success: bool | None
    native_reward: float | None = None
    termination_reason: str | None = None
    trajectory_ref: str | None = None
    tool_calls: int = 0
    enforce_communication_protocol: bool | None = None
    mixed_text_tool_call_messages: int = 0
    raw_review: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""
    total_steps: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_errors: int = 0
    hard_policy_protocol_violations: int = 0
    activated_skill_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task_id:
            raise ValueError("episode_id and task_id must be non-empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("episode seed must be a non-negative integer")
        if self.task_success is not None and type(self.task_success) is not bool:
            raise TypeError("task_success must be bool or None")
        if self.status == EpisodeStatus.COMPLETE and self.task_success is None:
            raise ValueError("a complete episode must record task_success")
        if self.native_reward is not None and (
            isinstance(self.native_reward, bool)
            or not isinstance(self.native_reward, (int, float))
            or not isfinite(self.native_reward)
        ):
            raise ValueError("native_reward must be a finite number or None")
        if type(self.tool_calls) is not int or self.tool_calls < 0:
            raise ValueError("tool_calls must be a non-negative integer")
        if (self.enforce_communication_protocol is not None
                and type(self.enforce_communication_protocol) is not bool):
            raise TypeError("enforce_communication_protocol must be bool or None")
        if (type(self.mixed_text_tool_call_messages) is not int
                or self.mixed_text_tool_call_messages < 0):
            raise ValueError("mixed_text_tool_call_messages must be a non-negative integer")
        for name in ('total_steps', 'prompt_tokens', 'completion_tokens', 'tool_errors', 'hard_policy_protocol_violations'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer or unavailable')
        object.__setattr__(self, 'activated_skill_ids', tuple(self.activated_skill_ids))
        if not isinstance(self.raw_review, Mapping):
            raise TypeError("raw_review must be a mapping")

    @property
    def episode_key(self) -> str:
        return sha256_json({
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "customer": self.customer_strategy_id,
            "service": self.service_strategy_id,
        })

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["activated_skill_ids"] = list(self.activated_skill_ids)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeRecord:
        fields = dict(value)
        fields["status"] = EpisodeStatus(fields["status"])
        return cls(**fields)


def customer_strategy_id(strategy: Any | None) -> str:
    if strategy is None:
        return sha256_json({"mode": "native_no_overlay"})[:16]
    return sha256_json(strategy.to_dict())[:16]


def service_strategy_id(strategy: Any) -> str:
    return sha256_json(strategy.to_dict())[:16]
