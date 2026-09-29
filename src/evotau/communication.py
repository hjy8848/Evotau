"""Telemetry for τ-bench half-duplex communication-format observations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def observe_communication_protocol(
    messages: Sequence[Mapping[str, Any]],
    *,
    enforcement_enabled: bool,
) -> dict[str, Any]:
    """Count mixed text/tool-call messages without changing the trajectory."""

    if type(enforcement_enabled) is not bool:
        raise TypeError("enforcement_enabled must be boolean")
    mixed_turn_indices: list[int] = []
    mixed_by_role: Counter[str] = Counter()
    for index, message in enumerate(messages):
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if not isinstance(content, str) or not content.strip() or not tool_calls:
            continue
        role = message.get("role")
        mixed_by_role[role if isinstance(role, str) and role else "unknown"] += 1
        turn_index = message.get("turn_idx")
        mixed_turn_indices.append(
            turn_index if type(turn_index) is int and turn_index >= 0 else index
        )
    return {
        "enforcement_enabled": enforcement_enabled,
        "mixed_text_tool_call_message_count": len(mixed_turn_indices),
        "mixed_text_tool_call_messages_by_role": dict(sorted(mixed_by_role.items())),
        "mixed_text_tool_call_turn_indices": mixed_turn_indices,
    }
