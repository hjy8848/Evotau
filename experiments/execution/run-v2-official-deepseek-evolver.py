"""Explicit official-DeepSeek Evolver auth; native InferAI credentials stay isolated."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from evotau.provider_diagnostics import response_metadata, safe_error


def install_official_transport(key, original):
    lock = threading.Lock()

    def record(value):
        process = json.loads(Path("/tmp/evotau-v2-live-process.json").read_text())
        if process["pid"] != os.getpid():
            raise RuntimeError("Official observer process binding mismatch")
        value["at"] = datetime.now(UTC).isoformat()
        with (
            lock,
            (Path(process["output"]) / "actual-provider-http.jsonl").open(
                "a"
            ) as stream,
        ):
            stream.write(json.dumps(value) + "\n")

    def send(client, request, *args, **kwargs):
        if request.url.host != "api.deepseek.com" or request.method != "POST":
            return original(client, request, *args, **kwargs)
        body = json.loads(request.content)
        if body.get("model") != "deepseek-flash" or body.get("tools"):
            raise ValueError(
                "Official credential is scoped to the configured no-tools Evolver"
            )
        request.headers["Authorization"] = "Bearer " + key
        record(
            {
                "event": "request",
                "provider": "official-deepseek",
                "args": {
                    k: body[k]
                    for k in (
                        "model",
                        "thinking",
                        "reasoning_effort",
                        "temperature",
                        "max_tokens",
                        "stream",
                    )
                    if k in body
                },
                "tools_count": len(body.get("tools") or []),
            }
        )
        start = perf_counter()
        try:
            response = original(client, request, *args, **kwargs)
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            record(
                {
                    "event": "response",
                    "provider": "official-deepseek",
                    "model": body["model"],
                    "http_status": response.status_code,
                    "metadata": response_metadata(payload),
                    "elapsed_seconds": perf_counter() - start,
                }
            )
            return response
        except BaseException as error:
            record(
                {
                    "event": "failure",
                    "provider": "official-deepseek",
                    "error": safe_error(error),
                    "elapsed_seconds": perf_counter() - start,
                }
            )
            raise

    return send


def main():
    if "--dry-run" not in sys.argv:
        key = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "api.deepseek.com/v1",
                "-a",
                "evotau-evolver-api-key",
                "-w",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not key:
            raise RuntimeError("Official DeepSeek Evolver credential missing")
        httpx.Client.send = install_official_transport(key, httpx.Client.send)
    path = Path(__file__).with_name("run-v2-live-evolution.py")
    spec = importlib.util.spec_from_file_location("official_live_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
