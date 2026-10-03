"""InferAI Qwen3.7 Plus callbacks for the frozen τ-bench behavior smoke."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..manifest import MechanismManifest
from .deepseek_v4_flash import _build_callbacks_for_model


def build_callbacks(
    *, config: Mapping[str, Any], manifest: MechanismManifest
) -> dict[str, Any]:
    """Build the shared independent-audit callbacks with Qwen3.7 Plus."""

    return _build_callbacks_for_model(
        config=config,
        manifest=manifest,
        expected_model="openai/qwen3.7-plus",
        verifier_model_label="qwen3.7-plus",
    )
