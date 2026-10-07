"""Explicit, audited import of native evidence after an Evolver-only handoff."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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
from evotau.evolution_artifacts import EvolutionJournal, summarize_activation_artifacts
from evotau.tau_episode_runner import TauBenchEpisodeRunner
from evotau.tau_provenance import sha256_json, write_manifest_once

PREFIX_STAGES = (
    "g0000-customer_incumbent",
    "g0000-customer_candidates",
    "g0000-customer-validator-0",
    "g0000-customer_selection",
)


def native_contract(document):
    doc = deepcopy(document)
    for key in (
        "manifest_sha256",
        "experiment_id",
        "config_sha256",
        "output_path",
        "checkpoint_path",
        "provider_provenance",
        "evotau",
    ):
        doc.pop(key, None)
    doc["role_models"].pop("evolver")
    doc["role_model_args"].pop("evolver")
    return doc


def validate_contract(parent, child):
    if parent["manifest_sha256"] != sha256_json(
        {k: v for k, v in parent.items() if k != "manifest_sha256"}
    ):
        raise ValueError("parent manifest digest mismatch")
    if native_contract(parent) != native_contract(child):
        raise ValueError("handoff must preserve every native/search/gate input")
    if child["role_models"]["evolver"] != "openai/gpt-6.1-sol" or child[
        "role_model_args"
    ]["evolver"] != {
        "api_protocol": "responses",
        "api_base": "https://inferaiapi.com/v1",
        "api_key_env": "INFERAI_API_KEY",
        "reasoning_effort": "high",
    }:
        raise ValueError(
            "handoff requires the explicitly authorized GPT Evolver settings"
        )


def audit_sources(parent, child):
    digest = hashlib.sha256()
    changes = []
    sources = sorted([*(ROOT / "src/evotau").rglob("*.py"), ROOT / "pyproject.toml"])
    for path in sources:
        if (
            path.is_relative_to(ROOT / "src/evotau/web")
            or path == ROOT / "src/evotau/observability.py"
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        before = subprocess.run(
            ["git", "show", parent["evotau"]["git_commit"] + ":" + relative],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
        name = relative.encode()
        digest.update(
            len(name).to_bytes(8, "big")
            + name
            + len(before).to_bytes(8, "big")
            + before
        )
        after = path.read_bytes()
        if before != after:
            if relative != "src/evotau/inferai_responses.py":
                raise ValueError(
                    "native runtime source changed outside the GPT-only route: "
                    + relative
                )
            # Permit only the specifically tested provider-prefix normalization.
            expected = before.replace(
                b'"model": model,\n',
                b'# Match LiteLLM\'s provider/model convention on the direct Responses route.\n        # The configured model identity stays unchanged in budget/provenance records.\n        "model": model.removeprefix("openai/"),\n',
                1,
            ).replace(
                b'"model": model, "api_base": api_base, "api_protocol": "responses",',
                b'"model": payload["model"], "api_base": api_base, "api_protocol": "responses",',
                1,
            )
            if expected != after:
                raise ValueError("GPT route changed beyond the reviewed normalization")
            changes.append(
                {
                    "file": relative,
                    "before_sha256": hashlib.sha256(before).hexdigest(),
                    "after_sha256": hashlib.sha256(after).hexdigest(),
                    "scope": "Evolver Responses wire ID only; never used by native Qwen roles",
                }
            )
    if digest.hexdigest() != parent["evotau"]["source_sha256"]:
        raise ValueError("parent source was not its recorded Git commit")
    return changes


def rebind(document, old_sha, new_sha):
    if isinstance(document, list):
        return [rebind(v, old_sha, new_sha) for v in document]
    if not isinstance(document, dict):
        return document
    result = {k: rebind(v, old_sha, new_sha) for k, v in document.items()}
    if result.get("manifest_sha256") == old_sha:
        result["manifest_sha256"] = new_sha
    if "payload_sha256" in result:
        result["payload_sha256"] = sha256_json(result["payload"])
    for field in ("artifact_sha256", "envelope_sha256"):
        if field in result:
            result[field] = sha256_json({k: v for k, v in result.items() if k != field})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau2-data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ["TAU2_DATA_DIR"] = str(args.tau2_data_dir.resolve())
    parent_config = yaml.safe_load(args.parent_config.read_text())
    raw = yaml.safe_load(args.config.read_text())
    source = ROOT / parent_config["experiment"]["output_path"]
    parent = json.loads((source / "manifest.json").read_text())
    # Current sources may differ solely by the separately audited GPT adapter fix.
    parent_manifest = AlternatingManifest.from_mapping(parent_config)
    if parent_manifest.config_sha256 != parent.get("config_sha256"):
        raise ValueError("parent configuration differs from its frozen manifest")
    child = AlternatingManifest.from_mapping(raw)
    child_doc = child.to_document()
    validate_contract(parent, child_doc)
    source_changes = audit_sources(parent, child_doc)
    if (
        json.loads((source / "run-execution-state.json").read_text())["status"]
        != "failed"
    ):
        raise ValueError("handoff parent must be terminal failed")
    journal = EvolutionJournal(source, parent["manifest_sha256"])
    stages = {
        name: journal.read(journal.root / (name + ".json")) for name in PREFIX_STAGES
    }
    rows = stages["g0000-customer_incumbent"]["payload"]["episodes"]
    if len(rows) != 20 or {(r["task_id"], r["seed"]) for r in rows} != {
        (t, 1) for t in child.evolution_task_ids
    }:
        raise ValueError("source is not the complete fixed E20 seed1 baseline")
    for record in rows:
        d = source / record["trajectory_ref"]
        folder = d.parent
        if json.loads((folder / "episode-record.json").read_text()) != record:
            raise ValueError("source score differs from immutable native stage")
        sim = json.loads(d.read_text())
        if (sim["id"], sim["task_id"], sim["seed"]) != (
            record["episode_id"],
            record["task_id"],
            record["seed"],
        ):
            raise ValueError("source native simulation condition mismatch")
    summarize_activation_artifacts(
        source, parent["manifest_sha256"], set(child.evolution_task_ids)
    )
    selection = stages["g0000-customer_selection"]["payload"]
    if selection["selected_source"] != "incumbent" or selection["customer"] != {
        "text": raw["experiment"]["customer_strategy"]
    }:
        raise ValueError(
            "this handoff requires the exact previously selected incumbent"
        )
    print(
        json.dumps(
            {
                "imported_episodes": len(rows),
                "native_successes": sum(r["task_success"] for r in rows),
                "imported_customer_stages_model": parent["role_models"]["evolver"],
                "future_evolver": dict(child.role_models)["evolver"],
                "source_changes": source_changes,
                "dry_run": args.dry_run,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    destination = ROOT / child.output_path
    if destination.exists():
        raise FileExistsError(
            "handoff destination already exists; resume its launcher instead"
        )
    destination.mkdir(parents=True)
    write_manifest_once(destination / "manifest.json", child)
    index = {}
    changes = []

    def copy_file(src, dst, transform=False):
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = src.read_bytes()
        index[str(src.relative_to(source))] = hashlib.sha256(before).hexdigest()
        if transform:
            value = rebind(json.loads(before), parent["manifest_sha256"], child.sha256)
            _write_json_once(dst, value)
            changes.append(
                {
                    "path": str(dst.relative_to(destination)),
                    "source_sha256": hashlib.sha256(before).hexdigest(),
                    "import_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                    "change": "manifest binding/hashes only",
                }
            )
        else:
            shutil.copyfile(src, dst)

    for record in rows:
        folder = (source / record["trajectory_ref"]).parent
        for file in folder.rglob("*"):
            if file.is_symlink():
                raise ValueError("unsafe source artifact")
            if file.is_file():
                transform = file.suffix == ".json" and (
                    file.name == "skill-activation-trace.json"
                    or file.parent.name == "skill-activation"
                )
                copy_file(file, destination / file.relative_to(source), transform)
    for name in PREFIX_STAGES:
        copy_file(
            journal.root / (name + ".json"),
            destination / "evolution-v2" / (name + ".json"),
            True,
        )
    # Preserve actual Pro authorship and both failed calls byte-for-byte.
    for file in (source / "evolver-calls").rglob("*"):
        if file.is_file():
            copy_file(file, destination / file.relative_to(source))
    for name in ("api-usage-live.json", "actual-provider-http.jsonl"):
        copy_file(source / name, destination / name)
    _write_json_once(
        destination / "evolver-handoff.json",
        {
            "schema_version": 1,
            "parent_run": str(source.relative_to(ROOT)),
            "parent_manifest_sha256": parent["manifest_sha256"],
            "manifest_sha256": child.sha256,
            "reason": "Pro diagnosis twice HTTP504 after120s; explicit authorized GPT fallback",
            "native_contract_sha256": sha256_json(native_contract(parent)),
            "source_code_changes": source_changes,
            "source_artifact_sha256": index,
            "binding_transformations": changes,
            "imported_stages": list(PREFIX_STAGES),
            "imported_stages_author_model": parent["role_models"]["evolver"],
            "imported_native_episodes": len(rows),
            "baseline_accuracy": sum(r["task_success"] for r in rows) / len(rows),
            "interpretation": "Mixed Pro→GPT continuation, not a pure GPT replicate; no native prompt/reward/trajectory content or selection rule changed",
        },
    )
    tasks = load_alternating_tasks(
        child, args.tau2_data_dir, include_validation=False, include_heldout=False
    )
    budget = RequestBudget(None)
    budget.enable_live_usage(destination / "api-usage-live.json")
    runner = TauBenchEpisodeRunner(
        manifest=child,
        config=raw,
        data_dir=args.tau2_data_dir,
        request_budget=budget,
        output_directory=destination,
        task_objects=tasks,
    )
    if len(runner._completed_episode_cache) != 20:
        raise ValueError("import failed native cache validation")
    summarize_activation_artifacts(
        destination, child.sha256, set(child.evolution_task_ids)
    )
    print(
        json.dumps(
            {
                "handoff_ready": str(destination),
                "cache_entries": 20,
                "provider_calls_during_import": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
