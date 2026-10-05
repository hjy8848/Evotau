from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evotau.alternating_manifest import AlternatingManifest
from evotau.budget import RequestBudget
from evotau.strategies import PromptStrategy
from evotau.tau_episode_runner import NativeEpisodeRunError, TauBenchEpisodeRunner


def test_value_error_message_and_cause_are_saved_in_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs/real-alternating-deepseek-v4-flash-5e3v5h.yaml")
        .read_text(encoding="utf-8"),
    )
    experiment = config["experiment"]
    experiment["id"] = "episode-failure-message-test"
    experiment["output_path"] = "experiments/episode-failure-message-test"
    experiment["checkpoint_path"] = "experiments/episode-failure-message-test/checkpoint.json"
    manifest = AlternatingManifest.from_mapping(config)
    data_root = tmp_path / "tau-data"
    policy_path = data_root / "tau2/domains/retail/policy.md"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("Fixed test policy.", encoding="utf-8")
    output = tmp_path / "run"
    task = SimpleNamespace(
        id="29", description="Test task", user_scenario="Do the fixed test task.",
    )

    def fail_before_native_run(**_kwargs):
        raise ValueError("task 29 invariant failed: expected order state was absent")

    monkeypatch.setattr(
        "evotau.tau_episode_runner.build_phase0_orchestrator", fail_before_native_run,
    )
    runner = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_root,
        request_budget=RequestBudget(manifest.request_budget_cap),
        output_directory=output,
        task_objects={"29": task},
    )

    with pytest.raises(NativeEpisodeRunError, match="expected order state was absent") as raised:
        runner(
            task_id="29", seed=1, customer=None,
            service=PromptStrategy("S0"), panel_name="generation-0-customer-incumbent",
        )

    artifact_path = next(output.glob("episodes/*/incomplete-run.json"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["failure_type"] == "ValueError"
    assert artifact["failure_message"] == "task 29 invariant failed: expected order state was absent"
    assert "expected order state was absent" in artifact["failure_repr"]
    assert artifact["cause_type"] is None
    assert artifact["cause_message"] is None
    assert artifact["exception_chain"][0]["message"] == artifact["failure_message"]
    assert "ValueError" in artifact["traceback"]
    assert "task 29 invariant failed" in str(raised.value)
    assert isinstance(raised.value.__cause__, ValueError)

    resumed = TauBenchEpisodeRunner(
        manifest=manifest,
        config=config,
        data_dir=data_root,
        request_budget=RequestBudget(manifest.request_budget_cap),
        output_directory=output,
        task_objects={"29": task},
    )
    with pytest.raises(NativeEpisodeRunError, match="expected order state was absent"):
        resumed(
            task_id="29", seed=1, customer=None,
            service=PromptStrategy("S0"), panel_name="generation-0-customer-incumbent",
        )
    assert len(tuple(output.glob("episodes/*/incomplete-run.json"))) == 2
