"""Explicit native baseline import for authorized context/budget-only continuation; zero API calls."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from evotau.alternating import _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.budget import RequestBudget
from evotau.evolution_artifacts import EvolutionJournal
from evotau.records import customer_strategy_id, service_strategy_id
from evotau.service_skills import ServiceSkillMemoryV2
from evotau.strategies import PromptStrategy
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import sha256_json, write_manifest_once

ROOT = Path(__file__).resolve().parents[2]
REVIEWED_RUNTIME = "f161c3d385d9e452b56c7a2facfb6b3f1d43b5ab"
EVOLUTION_CHANGES = {
    "src/evotau/" + name + ".py"
    for name in (
        "alternating_run",
        "evolution_candidates",
        "evolution_context",
        "skill_evolution",
        "skill_evolution_config",
    )
}


def native_condition(doc):
    keys = (
        "upstream",
        "source_blob_sha1",
        "domain",
        "task_panels",
        "seed",
        "evolution_fitness_seed",
        "max_steps",
        "communication_enforcement",
        "initial_customer_strategy_sha256",
        "initial_service_strategy_sha256",
    )
    result = {k: doc[k] for k in keys}
    for k in ("role_models", "role_model_args"):
        result[k] = {role: value for role, value in doc[k].items() if role != "evolver"}
    p = doc["skill_evolution_v2"]
    result["skill_runtime"] = {
        k: p[k]
        for k in (
            "service_skill_runtime",
            "max_active_service_skills",
            "activator",
            "skill_budgets",
        )
    }
    return result


def audit_code(parent, reviewed_runtime=REVIEWED_RUNTIME):
    digest, changes = hashlib.sha256(), []
    for path in sorted([*(ROOT / "src/evotau").rglob("*.py"), ROOT / "pyproject.toml"]):
        if (
            path.is_relative_to(ROOT / "src/evotau/web")
            or path.name == "observability.py"
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        before = subprocess.run(
            ["git", "show", parent["evotau"]["git_commit"] + ":" + relative],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        reviewed = subprocess.run(
            ["git", "show", reviewed_runtime + ":" + relative],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        after = path.read_bytes()
        if after != reviewed:
            raise ValueError("Unreviewed runtime change: " + relative)
        name = relative.encode()
        digest.update(
            len(name).to_bytes(8, "big")
            + name
            + len(before).to_bytes(8, "big")
            + before
        )
        if before != after:
            if relative not in EVOLUTION_CHANGES:
                raise ValueError("Native source changed: " + relative)
            changes.append(
                {
                    "file": relative,
                    "before_sha256": hashlib.sha256(before).hexdigest(),
                    "after_sha256": hashlib.sha256(after).hexdigest(),
                    "scope": "Reviewed external evolution refactor; native condition audited separately",
                }
            )
    if digest.hexdigest() != parent["evotau"]["source_sha256"]:
        raise ValueError("Parent source does not match recorded Git bytes")
    return changes


def rebind(value, old, new):
    if isinstance(value, list):
        return [rebind(v, old, new) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: rebind(v, old, new) for k, v in value.items()}
    if result.get("manifest_sha256") == old:
        result["manifest_sha256"] = new
    for field in ("artifact_sha256", "completion_sha256"):
        if field in result:
            result[field] = sha256_json({k: v for k, v in result.items() if k != field})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reviewed-runtime", required=True)
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = str(args.tau2_data_dir.resolve())
    source = args.source.resolve()
    if source.is_symlink() or not source.is_relative_to(ROOT / "experiments/runs"):
        raise ValueError("Unsafe source run")
    parent = json.loads((source / "manifest.json").read_text())
    if parent["manifest_sha256"] != sha256_json(
        {k: v for k, v in parent.items() if k != "manifest_sha256"}
    ):
        raise ValueError("Parent manifest digest mismatch")
    if (
        json.loads((source / "run-execution-state.json").read_text())["status"]
        != "failed"
    ):
        raise ValueError("Parent must be terminal failed")
    raw = yaml.safe_load(args.config.read_text())
    child = AlternatingManifest.from_mapping(raw)
    if json.loads(child.skill_evolution_v2_json)["algorithm_version"] not in (
        "direct_skill_v_validation_v2", "analyst_skill_v_validation_v3", "analyst_skill_recovery_v4"
    ):
        raise ValueError("Import requires a supported native-baseline-only V2 continuation")
    if native_condition(parent) != native_condition(child.to_document()):
        raise ValueError(
            "Native task/model/args/seed/strategy/evaluator condition changed"
        )
    old_policy = json.loads(json.dumps(parent["skill_evolution_v2"]))
    new_policy = json.loads(child.skill_evolution_v2_json)
    if old_policy["algorithm_version"] != new_policy["algorithm_version"]:
        raise ValueError("Algorithm version changed")
    for policy in (old_policy, new_policy):
        policy.pop("mutation_context", None)
        policy["evaluation"].pop("allow_unbounded_requests", None)
    if old_policy != new_policy:
        raise ValueError("Scientific gate/search policy changed")
    old_evolver = dict(parent["role_model_args"]["evolver"])
    new_evolver = child.role_model_args_dict["evolver"]
    old_evolver.pop("max_tokens", None)
    if old_evolver != new_evolver or parent["role_models"]["evolver"] != dict(child.role_models)["evolver"]:
        raise ValueError("Evolver model or non-output-limit args changed")
    changes = audit_code(parent, args.reviewed_runtime)
    journal = EvolutionJournal(source, parent["manifest_sha256"])
    baseline_file = journal.root / "g0000-customer_incumbent.json"
    if any(p.name != baseline_file.name for p in journal.root.glob("g*.json")):
        raise ValueError("Source contains additional search stages; native-only import refused")
    rows = journal.read(baseline_file)["payload"]["episodes"]
    expected = {(t, child.evolution_fitness_seed) for t in child.evolution_task_ids}
    if len(rows) != len(expected) or {(r["task_id"], r["seed"]) for r in rows} != expected:
        raise ValueError("Source is not exact complete E fitness baseline")
    cs = customer_strategy_id(PromptStrategy(raw["experiment"]["customer_strategy"]))
    ss = service_strategy_id(ServiceSkillMemoryV2())
    for row in rows:
        path = (source / row["trajectory_ref"]).resolve()
        if not path.is_relative_to(source / "episodes"):
            raise ValueError("Unsafe native source reference")
        record = json.loads(path.with_name("episode-record.json").read_text())
        sim = json.loads(path.read_text())
        if (
            record != row
            or row["status"] != "complete"
            or type(row["task_success"]) is not bool
        ):
            raise ValueError("Source scored record mismatch")
        if (
            sim["id"],
            str(sim["task_id"]),
            sim["seed"],
            row["customer_strategy_id"],
            row["service_strategy_id"],
        ) != (row["episode_id"], row["task_id"], row["seed"], cs, ss):
            raise ValueError("Source native simulation condition mismatch")
    usage = json.loads((source / "api-usage-live.json").read_text())
    if usage["provider_usage"].get("in_flight", 0) or usage["provider_usage"].get(
        "reserved", 0
    ):
        raise ValueError("Parent has unresolved requests")
    report = {
        "parent_run": str(source.relative_to(ROOT)),
        "parent_manifest_sha256": parent["manifest_sha256"],
        "manifest_sha256": child.sha256,
        "reviewed_runtime_commit": args.reviewed_runtime,
        "native_condition_sha256": sha256_json(native_condition(parent)),
        "source_code_changes": changes,
        "imported_native_episodes": len(rows),
        "native_successes": sum(r["task_success"] for r in rows),
        "imported_search_stages": [],
        "imported_evolver_calls": 0,
        "provider_calls_during_import": 0,
        "parent_usage_carried_forward": usage["provider_usage"],
        "interpretation": "Context/budget-only continuation with exact native baseline reuse; thresholds and native conditions unchanged. No Evolver requests were completed in the source. Historical usage counters carried forward.",
    }
    if args.dry_run:
        print(json.dumps(report))
        return
    destination = ROOT / child.output_path
    if destination.exists():
        raise FileExistsError(
            "New destination required; do not overwrite frozen imports"
        )
    destination.mkdir(parents=True)
    write_manifest_once(destination / "manifest.json", child)
    copies = []
    for row in rows:
        folder = (source / row["trajectory_ref"]).parent
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError("Unsafe source artifact")
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                value = json.loads(path.read_text())
                rebound = rebind(value, parent["manifest_sha256"], child.sha256)
                if value != rebound:
                    _write_json_once(target, rebound)
                else:
                    shutil.copyfile(path, target)
            else:
                shutil.copyfile(path, target)
            copies.append(
                {
                    "path": str(path.relative_to(source)),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "import_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "change": "none"
                    if path.read_bytes() == target.read_bytes()
                    else "manifest binding/digests only",
                }
            )
    derived_usage = json.loads(json.dumps(usage))
    derived_usage["provider_usage"]["cap"] = child.request_budget_cap
    _write_json_once(destination / "api-usage-live.json", derived_usage)
    report["budget_cap_migration"] = {"source": usage["provider_usage"]["cap"], "target": child.request_budget_cap, "all_request_and_token_counters_unchanged": True}
    report["copies"] = copies
    _write_json_once(destination / "native-baseline-import.json", report)
    budget = RequestBudget(child.request_budget_cap)
    budget.enable_live_usage(destination / "api-usage-live.json")
    tasks = load_alternating_tasks(
        child, args.tau2_data_dir, include_validation=True, include_heldout=False
    )
    runner = TauBenchEpisodeRunner(
        manifest=child,
        config=raw,
        data_dir=args.tau2_data_dir,
        request_budget=budget,
        output_directory=destination,
        task_objects=tasks,
    )
    if len(runner._completed_episode_cache) != len(expected):
        raise ValueError("Imported cache does not validate all native conditions")
    print(
        json.dumps(
            {
                "cache_entries": len(expected),
                "parent_usage_charged": budget.snapshot().attempts,
                "provider_calls_during_import": 0,
                "destination": str(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
