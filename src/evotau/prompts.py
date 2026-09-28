"""Prompt composition helpers that preserve the upstream prompt verbatim."""

from __future__ import annotations


def append_strategy_block(base_prompt: str, strategy_block: str) -> str:
    """Append a separate strategy section without modifying the native prompt."""

    if not strategy_block:
        return base_prompt
    if not isinstance(base_prompt, str):
        raise TypeError("base_prompt must be a string")
    return f"{base_prompt}\n\n{strategy_block}"
