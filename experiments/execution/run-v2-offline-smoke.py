"""Export one deterministic native V2 integration run. Never calls a live provider."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--tau2-data-dir", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
environment = dict(os.environ)
environment["EVOTAU_V2_OFFLINE_EXPORT"] = str(args.export.absolute())
if args.tau2_data_dir:
    environment["EVOTAU_TAU2_DATA_DIR"] = str(args.tau2_data_dir)
    environment["TAU2_DATA_DIR"] = str(args.tau2_data_dir)
if args.export.exists():
    parser.error("export directory already exists; choose a new immutable export")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_skill_evolution_v2_native.py"],
    cwd=root,
    env=environment,
    check=False,
)
if result.returncode == 0 and not (args.export / "offline-evidence.json").is_file():
    sys.exit("Pinned native integration was skipped; no evidence was exported.")
sys.exit(result.returncode)
