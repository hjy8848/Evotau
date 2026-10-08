"""Operator-authorized recovery; makes no provider calls and changes no model output."""

import argparse
import json
from pathlib import Path

from evotau.alternating_manifest import AlternatingManifest
from evotau.alternating_run import load_config
from evotau.release_recovery import (
    authorize_schema_retry,
    frozen_run_lock,
    reconcile_interrupted_requests,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--authorize-schema-retry", metavar="CALL_DIRECTORY_ID")
    actions.add_argument("--charge-interrupted-requests", action="store_true")
    args = parser.parse_args()
    file = args.config.resolve()
    manifest = AlternatingManifest.from_mapping(load_config(file))
    project = file.parent.parent if file.parent.name == "configs" else file.parent
    root = project / manifest.output_path
    manifest.bind_saved_provenance(json.loads((root / "manifest.json").read_text()))
    with frozen_run_lock(project / manifest.checkpoint_path):
        result = (
            authorize_schema_retry(root, args.authorize_schema_retry, args.reason)
            if args.authorize_schema_retry
            else reconcile_interrupted_requests(root, args.reason)
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
