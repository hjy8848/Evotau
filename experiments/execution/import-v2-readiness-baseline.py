"""Audited same-runtime baseline reuse across independent readiness panels; no API."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evotau.alternating import _write_json_once
from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_alternating_tasks
from evotau.budget import RequestBudget
from evotau.evolution_artifacts import EvolutionJournal
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import write_manifest_once


def contract(document):
    doc = deepcopy(document)
    for k in ("manifest_sha256", "experiment_id", "config_sha256", "output_path",
              "checkpoint_path", "provider_provenance"):
        doc.pop(k, None)
    doc["evotau"] = {"source_sha256": doc["evotau"]["source_sha256"]}
    doc["task_panels"].pop("E")
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = str(args.tau2_data_dir.resolve())
    parent_raw = yaml.safe_load(args.parent_config.read_text())
    raw = yaml.safe_load(args.config.read_text())
    parent = AlternatingManifest.from_mapping(parent_raw)
    source = ROOT / parent.output_path
    parent = parent.bind_saved_provenance(json.loads((source / "manifest.json").read_text()))
    child = AlternatingManifest.from_mapping(raw)
    if contract(parent.to_document()) != contract(child.to_document()):
        raise ValueError("readiness import changed source/runtime/strategy/seed/gate beyond E panel")
    if json.loads((source / "run-execution-state.json").read_text())["status"] != "complete":
        raise ValueError("readiness baseline parent is not terminal complete")
    journal = EvolutionJournal(source, parent.sha256)
    baseline = journal.read(journal.root / "g0000-customer_incumbent.json")["payload"]["episodes"]
    rows = [r for r in baseline if r["task_id"] in child.evolution_task_ids]
    if not rows:
        raise ValueError("no baseline overlap to reuse")
    tasks = load_alternating_tasks(parent, args.tau2_data_dir,
                                   include_validation=True, include_heldout=False)
    budget = RequestBudget(parent.request_budget_cap)
    budget.enable_live_usage(source / "api-usage-live.json")
    runner = TauBenchEpisodeRunner(manifest=parent, config=parent_raw,
                                  data_dir=args.tau2_data_dir, request_budget=budget,
                                  output_directory=source, task_objects=tasks)
    for record in rows:
        if json.loads((source / record["trajectory_ref"]).with_name("episode-record.json").read_text()) != record:
            raise ValueError("source baseline record differs from frozen stage")
    if len(runner._completed_episode_cache) != len(baseline):
        raise ValueError("source cache validation failed")
    destination = ROOT / child.output_path
    if destination.exists():
        raise FileExistsError("new readiness destination required")
    module_path = Path(__file__).with_name("import-v2-evolver-handoff.py")
    spec = importlib.util.spec_from_file_location("handoff_rebind", module_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination.mkdir(parents=True)
    write_manifest_once(destination / "manifest.json", child)
    copies = []
    for record in rows:
        for path in (source / record["trajectory_ref"]).parent.rglob("*"):
            if path.is_symlink():
                raise ValueError("unsafe source artifact")
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            changed = path.suffix == ".json" and (path.name == "skill-activation-trace.json"
                                                   or path.parent.name == "skill-activation")
            if changed:
                _write_json_once(target, helper.rebind(json.loads(path.read_text()), parent.sha256, child.sha256))
            else:
                shutil.copyfile(path, target)
            copies.append({"path": str(path.relative_to(source)),
                           "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "import_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                           "change": "manifest binding only" if changed else "none"})
    # Charge all earlier calls, including the unused E16 and earlier Evolver calls;
    # do not reset the allowance or pretend these were newly billed cache hits.
    for name in ("api-usage-live.json", "actual-provider-http.jsonl"):
        shutil.copyfile(source / name, destination / name)
    _write_json_once(destination / "readiness-baseline-import.json", {
        "parent_run": str(source.relative_to(ROOT)), "parent_manifest_sha256": parent.sha256,
        "manifest_sha256": child.sha256, "records": rows, "copies": copies,
        "provider_calls_during_import": 0, "imported_stages": [],
        "parent_usage_carried_forward": json.loads((source / "api-usage-live.json").read_text())["provider_usage"],
        "interpretation": "same native condition/source, explicit E-only panel expansion; no search stage imported",
    })
    child_budget = RequestBudget(child.request_budget_cap)
    child_budget.enable_live_usage(destination / "api-usage-live.json")
    child_runner = TauBenchEpisodeRunner(
        manifest=child, config=raw, data_dir=args.tau2_data_dir,
        request_budget=child_budget, output_directory=destination,
        task_objects=load_alternating_tasks(child, args.tau2_data_dir,
                                            include_validation=True, include_heldout=False))
    if len(child_runner._completed_episode_cache) != len(rows):
        raise ValueError("destination cache validation failed")
    print(json.dumps({"cache_entries": len(rows), "provider_calls_during_import": 0,
                      "parent_usage_charged": child_budget.snapshot().attempts}))


if __name__ == "__main__":
    main()
