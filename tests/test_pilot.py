from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evotau.archive import FailureArchive
from evotau.budget import RequestBudget
from evotau.lifecycle import TwoGenerationSmoke
from evotau.manifest import PilotManifest, git_blob_sha1
from evotau.native_runner import run_native_pilot
from evotau.phase0 import load_config
from evotau.pilot_preflight import main, validate_pilot_config
from evotau.records import (
    EpisodeRecord,
    EpisodeStatus,
    EvidenceRef,
    customer_strategy_id,
    service_strategy_id,
)
from evotau.strategies import CustomerStrategy, ServiceStrategy

ROOT = Path(__file__).resolve().parents[1]


def _eligible_task(task_id: str, index: int) -> dict:
    order_id = f"W{index:07d}"
    return {
        "id": task_id,
        "user_scenario": {"instructions": {
            "task_instructions": "You are a customer.",
            "reason_for_call": "You want to modify a pending order.",
            "known_info": f"You name is Person{index} Smith and your email is user{index}@example.com. Your order is {order_id}.",
            "unknown_info": None,
        }},
        "evaluation_criteria": {"actions": [{
            "name": "modify_pending_order_address",
            "arguments": {"order_id": order_id},
        }]},
    }


def _pilot_fixture(tmp_path: Path):
    config = load_config(ROOT / "configs/phase3-mechanism.yaml")
    experiment = config["experiment"]
    experiment.update({
        "id": "evotau-pilot-fixture",
        "phase": "4-pilot",
        "condition": "adaptive_coevolution",
        "evolution_seeds": [11, 17, 23],
        "generations": 3,
        "max_episodes": 100,
        "request_budget_cap": 5000,
        "output_path": "experiments/pilot-fixture",
        "checkpoint_path": "experiments/pilot-fixture/checkpoint.json",
    })
    experiment["task_selection"] = {
        "source_split": "train",
        "evolution": ["E1", "E2", "E3", "E4"],
        "validation": ["V1", "V2"],
        "heldout_split": "test",
        "heldout": ["H1", "H2"],
        "excluded": ["X1"],
    }
    tasks = [
        *(_eligible_task(f"E{index}", index) for index in range(1, 5)),
        *(_eligible_task(f"V{index}", index + 10) for index in range(1, 3)),
        *(_eligible_task(f"H{index}", index + 20) for index in range(1, 3)),
    ]
    split = {
        "train": ["E1", "E2", "E3", "E4", "V1", "V2"],
        "test": ["H1", "H2"],
    }
    task_bytes = json.dumps(tasks).encode()
    split_bytes = json.dumps(split).encode()
    data_root = tmp_path / "data"
    task_path = data_root / "tau2/domains/retail/tasks.json"
    split_path = data_root / "tau2/domains/retail/split_tasks.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_bytes(task_bytes)
    split_path.write_bytes(split_bytes)
    experiment["source_blob_sha1"]["data/tau2/domains/retail/tasks.json"] = git_blob_sha1(task_bytes)
    experiment["source_blob_sha1"]["data/tau2/domains/retail/split_tasks.json"] = git_blob_sha1(split_bytes)
    return config, data_root


def test_pilot_manifest_freezes_three_generation_multiseed_evh_design(tmp_path):
    config, _data_root = _pilot_fixture(tmp_path)
    manifest = PilotManifest.from_mapping(config)

    assert manifest.generations == 3
    assert manifest.evolution_seeds == (11, 17, 23)
    assert len(manifest.evolution_task_ids) == 4
    assert len(manifest.validation_task_ids) == 2
    assert len(manifest.heldout_task_ids) == 2
    assert manifest.to_payload()["failure_taxonomy_sha256"]
    assert manifest.to_document()["manifest_sha256"] == manifest.sha256


def test_typed_pilot_manifest_runs_one_independent_seed_checkpoint(tmp_path):
    config, _data_root = _pilot_fixture(tmp_path)
    manifest = PilotManifest.from_mapping(config)
    customer, service = CustomerStrategy(), ServiceStrategy()
    calls = []

    def runner(*, task_id, seed, customer, service, panel_name):
        calls.append((task_id, seed, panel_name))
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{customer_strategy_id(customer)}",
            task_id=task_id, seed=seed, customer_strategy_id=customer_strategy_id(customer),
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=True, customer_valid=True, customer_strategy_adherent=True,
            invalid_repeated_write_calls=0,
        )

    controller = TwoGenerationSmoke(
        manifest=manifest,
        checkpoint_path=str(tmp_path / "pilot-seed-11.json"),
        runner=runner,
        task_ids=(manifest.evolution_task_ids[0], manifest.validation_task_ids[0]),
        evolution_task_ids=manifest.evolution_task_ids,
        seed=manifest.evolution_seeds[0],
        request_budget=RequestBudget(manifest.request_budget_cap),
    )
    commits = controller.run(customer, service)
    assert [item.generation for item in commits] == [0, 1, 2]
    assert len(calls) == 36


def test_random_pilot_freezes_service_after_audited_failures(tmp_path):
    config, _data_root = _pilot_fixture(tmp_path)
    config["experiment"]["condition"] = "random_mutation"
    manifest = PilotManifest.from_mapping(config)
    archive = FailureArchive(tmp_path / "random-control.sqlite")
    service = ServiceStrategy()

    def runner(*, task_id, seed, customer, service, panel_name):
        strategy_id = customer_strategy_id(customer)
        return EpisodeRecord(
            episode_id=f"{panel_name}:{task_id}:{seed}:{strategy_id}",
            task_id=task_id, seed=seed, customer_strategy_id=strategy_id,
            service_strategy_id=service_strategy_id(service), status=EpisodeStatus.COMPLETE,
            task_success=False, customer_valid=True, strategy_applicable=True,
            customer_strategy_adherent=True, policy_violation=True,
            invalid_repeated_write_calls=0,
            policy_rule_id="retail.policy:explicit_confirmation",
            mistake_type="missing_explicit_confirmation", workflow_stage="pre_write",
            evidence=(EvidenceRef(2, "tool", "write occurred without confirmation"),),
        )

    controller = TwoGenerationSmoke(
        manifest=manifest, checkpoint_path=str(tmp_path / "random-control-checkpoint.json"),
        runner=runner,
        task_ids=(manifest.evolution_task_ids[0], manifest.validation_task_ids[0]),
        evolution_task_ids=manifest.evolution_task_ids, seed=manifest.evolution_seeds[0],
        request_budget=RequestBudget(manifest.request_budget_cap), failure_archive=archive,
    )
    commits = controller.run(
        CustomerStrategy(), service, failure_verifier=lambda item: f"audit:{item.episode_id}",
        customer_proposal_mode="random_mutation", allow_frozen_service=True,
    )

    assert [commit.service_id for commit in commits] == [service_strategy_id(service)] * 3
    assert all(
        commit.decision_record["customer"]["proposal_mode"] == "random_mutation"
        and not commit.decision_record["service"]["transition_ran"]
        and commit.decision_record["service"]["incumbent_before"]
        == commit.decision_record["service"]["incumbent_after"]
        for commit in commits
    )
    assert archive.recent()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evolution_seeds", [11, 17], "at least three"),
        ("generations", 2, "exactly three generations"),
        ("max_episodes", 0, "positive per-seed cap"),
    ],
)
def test_pilot_manifest_rejects_incomplete_frozen_design(tmp_path, field, value, message):
    config, _data_root = _pilot_fixture(tmp_path)
    config["experiment"][field] = value
    with pytest.raises(ValueError, match=message):
        PilotManifest.from_mapping(config)


def test_pilot_preflight_checks_pinned_files_entity_leakage_and_write_once(tmp_path):
    config, data_root = _pilot_fixture(tmp_path)
    result = validate_pilot_config(config, data_dir=data_root)
    assert result["task_selection"]["status"] == "eligible_partition_validated"
    assert len(result["task_selection"]["evolution_entity_keys"]) == 8
    assert result["task_selection"]["heldout_task_ids"] == ["H1", "H2"]

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = tmp_path / "preflight.json"
    assert main([
        "--config", str(config_path), "--tau2-data-dir", str(data_root),
        "--output", str(output_path),
    ]) == 0
    original = output_path.read_bytes()
    assert main([
        "--config", str(config_path), "--tau2-data-dir", str(data_root),
        "--output", str(output_path),
    ]) == 2
    assert output_path.read_bytes() == original

    config["experiment"]["task_selection"]["heldout"] = ["H1", "E1"]
    with pytest.raises(ValueError, match="unique and disjoint"):
        validate_pilot_config(config, data_dir=data_root)


def test_native_pilot_never_loads_provider_callbacks_while_provider_is_disabled(tmp_path):
    config, data_root = _pilot_fixture(tmp_path)
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []

    def callback_factory(*_args):
        calls.append(True)
        return {}

    with pytest.raises(RuntimeError, match="disabled in the frozen manifest"):
        run_native_pilot(
            config_path=config_path, data_dir=data_root, callback_factory=callback_factory,
        )
    assert not calls


@pytest.mark.parametrize("condition", ["adaptive_coevolution", "random_mutation"])
def test_native_pilot_runs_three_seed_blocks_and_indexes_final_heldout_panels(
    tmp_path, monkeypatch, condition,
):
    config, data_root = _pilot_fixture(tmp_path)
    experiment = config["experiment"]
    experiment["condition"] = condition
    experiment["real_provider_enabled"] = True
    experiment["models"] = {
        "agent": "mock-agent", "customer": "mock-customer",
        "reviewer": "mock-reviewer", "evaluator": "mock-evaluator",
        "evolver": "mock-evolver",
    }
    config_path = tmp_path / "pilot-live-fixture.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []
    callback_seed_bases = []

    class FakePilotEpisodeRunner:
        def __init__(self, *, manifest, output_directory, **_kwargs):
            self.manifest = manifest
            self.output_directory = Path(output_directory)
            self.output_directory.mkdir(parents=True, exist_ok=True)
            manifest_path = self.output_directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_document()), encoding="utf-8")
            self.service_policy_text = "fixed pilot policy"
            self._counter = 0

        def __call__(self, *, task_id, seed, customer, service, panel_name):
            self._counter += 1
            calls.append((task_id, seed, panel_name))
            record = EpisodeRecord(
                episode_id=f"pilot-{self._counter}-{task_id}-{seed}",
                task_id=task_id, seed=seed,
                customer_strategy_id=customer_strategy_id(customer),
                service_strategy_id=service_strategy_id(service),
                status=EpisodeStatus.COMPLETE, task_success=True, native_reward=1.0,
                customer_valid=True, strategy_applicable=customer is not None,
                customer_strategy_adherent=True if customer is not None else None,
                policy_violation=False, invalid_repeated_write_calls=0,
            )
            episode_dir = self.output_directory / "episodes" / f"episode-{self._counter:03d}"
            episode_dir.mkdir(parents=True)
            (episode_dir / "episode-record.json").write_text(
                json.dumps(record.to_dict()), encoding="utf-8",
            )
            return record

    from evotau import native_runner

    monkeypatch.setattr(
        native_runner, "_load_pinned_tasks",
        lambda _manifest, *, task_ids, **_kwargs: {task_id: object() for task_id in task_ids},
    )
    monkeypatch.setattr(native_runner, "TauBenchEpisodeRunner", FakePilotEpisodeRunner)

    def selector(context):
        return context.allowed_operators[:context.candidate_count]

    def callback_factory(_config, _manifest, seed_base):
        callback_seed_bases.append(seed_base)
        if condition == "random_mutation":
            return {"audit_provider": lambda *_args: None}
        return {
            "audit_provider": lambda *_args: None,
            "customer_proposal_provider": selector,
            "service_transition": lambda *_args: None,
        }

    result = run_native_pilot(
        config_path=config_path, data_dir=data_root, callback_factory=callback_factory,
    )
    assert result["status"] == "complete"
    assert result["condition"] == condition
    assert result["evolution_seeds"] == [11, 17, 23]
    assert callback_seed_bases == [100_000, 200_000, 300_000]
    observed_episode_seed_blocks = []
    for index, item in enumerate(result["seed_blocks"]):
        seed_result = json.loads(Path(item["result_path"]).read_text(encoding="utf-8"))
        expected_seed_base = (index + 1) * 100_000
        assert item["episode_seed_base"] == seed_result["episode_seed_base"] == expected_seed_base
        actual_seeds = {episode["seed"] for episode in seed_result["episodes"]}
        assert actual_seeds and all(expected_seed_base <= value < expected_seed_base + 100_000
                                   for value in actual_seeds)
        observed_episode_seed_blocks.append(actual_seeds)
        assert [len(commit["decision_record"]["customer"]["proposals"])
                for commit in seed_result["generation_commits"]] == [2, 1, 0]
        expected_mode = "random_mutation" if condition == "random_mutation" else "failure_conditioned"
        assert all(
            commit["decision_record"]["customer"]["proposal_mode"] == expected_mode
            for commit in seed_result["generation_commits"]
        )
        if condition == "random_mutation":
            assert len({commit["service_id"] for commit in seed_result["generation_commits"]}) == 1
            assert all(
                proposal["rationale"] == "random_mutation"
                and not proposal["supporting_failure_ids"]
                for commit in seed_result["generation_commits"]
                for proposal in commit["decision_record"]["customer"]["proposals"]
            )
        expected_count = (
            sum(
                (1 + len(commit["decision_record"]["customer"]["proposals"]))
                * len(config["experiment"]["task_selection"]["evolution"])
                for commit in seed_result["generation_commits"]
            )
            + len(config["experiment"]["task_selection"]["validation"])
            + len(config["experiment"]["task_selection"]["heldout"])
        )
        assert item["episode_count"] == expected_count == len(seed_result["episodes"])
    assert all(
        not (left & right)
        for index, left in enumerate(observed_episode_seed_blocks)
        for right in observed_episode_seed_blocks[index + 1:]
    )
    assert all(len(item["provider_budget"]["model_usage"]) == 0 for item in result["seed_blocks"])
    assert {task_id for task_id, _seed, _panel in calls if task_id.startswith("H")} == {"H1", "H2"}
    assert not any("H" in panel for _task, _seed, panel in calls if "heldout" not in panel)
    previous_call_count = len(calls)
    resumed = run_native_pilot(
        config_path=config_path, data_dir=data_root, callback_factory=callback_factory,
    )
    assert resumed == result
    assert len(calls) == previous_call_count
