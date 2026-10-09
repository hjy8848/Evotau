"""E-only baseline/A-A entry; existing InferAI pacing and sanitized HTTP observer."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evotau.aa_diagnostic import main
from evotau.phase0 import load_config

if __name__ == "__main__":
    os.chdir(ROOT)
    config = load_config(sys.argv[sys.argv.index("--config") + 1])
    for name, expected in (
        config.get("launch_readiness", {}).get("execution_source_sha256", {}).items()
    ):
        path = ROOT / name
        if (
            not path.resolve().is_relative_to(ROOT)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"frozen execution source differs: {name}")
    if "--execute" in sys.argv:
        # The CLI still enforces frozen finite cap; no credential is written or printed.
        key = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "inferaiapi.com/v1",
                "-a",
                "openai-api-key",
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not key:
            raise RuntimeError("InferAI Keychain credential missing")
        os.environ["OPENAI_API_KEY"] = key
        del key
        index = sys.argv.index("--output")
        output = Path(sys.argv[index + 1])
        output.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location(
            "native_wire",
            Path(__file__).with_name("run-v2-activation-morphology-smoke.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install_transport(output)
    main()
