"""Minimal direct InferAI Responses API client for alternating Evolver calls."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from .budget import RequestBudget

_KEYCHAIN_SERVICE = "inferaiapi.com/v1"
_KEYCHAIN_ACCOUNT = "openai-gpt-api-key"
_REQUEST_TIMEOUT_SECONDS = 180.0


def generate_text(
    *,
    model: str,
    api_base: str,
    api_key_env: str,
    reasoning_effort: str,
    system_prompt: str,
    user_prompt: str,
    call_name: str,
    request_budget: RequestBudget | None = None,
) -> str:
    """Call InferAI directly using Responses API, with no SDK or retries."""

    api_key = _resolve_api_key(api_key_env)
    payload = {
        # Match LiteLLM's provider/model convention on the direct Responses route.
        # The configured model identity stays unchanged in budget/provenance records.
        "model": model.removeprefix("openai/"),
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": reasoning_effort},
    }

    def dispatch() -> dict[str, Any]:
        return _post_responses(
            f"{api_base.rstrip('/')}/responses",
            payload,
            api_key,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    response = (
        request_budget.dispatch_external_call(
            model=model,
            call_name=call_name,
            dispatch=dispatch,
            request_args={
                "model": payload["model"], "api_base": api_base, "api_protocol": "responses",
                "reasoning": payload["reasoning"], "timeout": _REQUEST_TIMEOUT_SECONDS,
                "num_retries": 0,
            },
        )
        if request_budget is not None
        else dispatch()
    )
    return _response_text(response)


def _resolve_api_key(environment_name: str) -> str:
    key = os.environ.get(environment_name, "").strip()
    if key:
        return key
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                [
                    "security", "find-generic-password", "-s", _KEYCHAIN_SERVICE,
                    "-a", _KEYCHAIN_ACCOUNT, "-w",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            key = result.stdout.strip()
            if key:
                return key
    raise RuntimeError(
        f"InferAI API key is unavailable; set {environment_name} or add the GPT key "
        f"to the macOS Keychain service {_KEYCHAIN_SERVICE!r}, account {_KEYCHAIN_ACCOUNT!r}"
    )


def _build_opener() -> urllib.request.OpenerDirector:
    has_environment_proxy = any(
        os.environ.get(name)
        for name in ("https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
    )
    if has_environment_proxy or sys.platform != "darwin":
        return urllib.request.build_opener()
    try:
        settings = subprocess.check_output(["scutil", "--proxy"], text=True, timeout=3)
        enabled = re.search(r"HTTPSProxyEnabled\s*:\s*1", settings)
        host = re.search(r"HTTPSProxy\s*:\s*(\S+)", settings)
        port = re.search(r"HTTPSPort\s*:\s*(\d+)", settings)
        if enabled and host and port:
            return urllib.request.build_opener(urllib.request.ProxyHandler({
                "https": f"http://{host.group(1)}:{port.group(1)}",
            }))
    except (OSError, subprocess.SubprocessError):
        pass
    return urllib.request.build_opener()


def _post_responses(
    url: str,
    payload: Mapping[str, Any],
    api_key: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    opener = _build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        detail = raw.decode("utf-8", "replace")[:500]
        raise RuntimeError(
            f"InferAI Responses API returned HTTP {error.code}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"InferAI Responses API request failed: {error.reason}") from error
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("InferAI Responses API returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise TypeError("InferAI Responses API returned a non-object response")
    return parsed


def _response_text(response: Mapping[str, Any]) -> str:
    if response.get("status") not in (None, "completed"):
        raise RuntimeError("InferAI Responses API did not reach completed status")
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    pieces: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
    if pieces:
        return "".join(pieces)
    raise RuntimeError("InferAI Responses API returned no text output")
