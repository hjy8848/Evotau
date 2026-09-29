from __future__ import annotations

from pathlib import Path

import pytest

from evotau.manifest import ExperimentManifest
from evotau.phase0 import load_config
from evotau.recover_phase0_review import _verify_compatible_parent

ROOT = Path(__file__).resolve().parents[1]


def test_review_recovery_requires_identical_frozen_runtime_and_episode_settings() -> (
    None
):
    source = ExperimentManifest.from_mapping(
        load_config(
            ROOT
            / "configs/phase0-inferai-deepseek-v4-flash-evolution-smoke-attempt2.yaml"
        )
    ).to_document()
    recovery = ExperimentManifest.from_mapping(
        load_config(
            ROOT
            / "configs/phase0-inferai-deepseek-v4-flash-evolution-smoke-review-recovery.yaml"
        )
    ).to_document()
    _verify_compatible_parent(source, recovery)

    recovery["role_model_args"]["agent"]["thinking_mode"] = "enabled"
    with pytest.raises(ValueError, match="role_model_args"):
        _verify_compatible_parent(source, recovery)
