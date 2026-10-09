"""Response metadata for diagnosis; never persist credentials or reasoning text."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def safe_error(error: BaseException, *, limit: int = 8_000) -> str:
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", str(error))
    return re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)[:limit]


def safe_request_args(args: Mapping[str, Any]) -> dict[str, Any]:
    # Explicit allowlist: headers, credentials, prompts, clients and arbitrary
    # extra_body fields must never get serialized to telemetry.
    names = (
        "model", "temperature", "seed", "reasoning_effort", "max_tokens",
        "max_completion_tokens", "timeout", "num_retries", "tool_choice",
        "parallel_tool_calls", "api_protocol", "stream",
    )
    result = {name: args[name] for name in names if name in args
              and isinstance(args[name], (str, int, float, bool, type(None)))}
    base = args.get("api_base")
    if isinstance(base, str):
        from urllib.parse import urlsplit, urlunsplit
        url = urlsplit(base)
        result["api_base"] = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    for container, allowed in (("extra_body", ("thinking", "enable_thinking")), ("reasoning", ("effort",))):
        value = args.get(container)
        if isinstance(value, Mapping):
            filtered = {}
            for name in allowed:
                item = value.get(name)
                if name == "thinking" and isinstance(item, Mapping):
                    item = {"type": item.get("type")}
                if isinstance(item, (str, dict, bool)):
                    filtered[name] = item
            if filtered:
                result[container] = filtered
    tools = args.get("tools")
    result["tool_count"] = len(tools) if isinstance(tools, (list, tuple)) else 0
    return result


def response_metadata(response: Any) -> dict[str, Any]:
    def field(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)

    choices = field(response, "choices") or ()
    message = field(choices[0], "message") if choices else None
    content = field(message, "content")
    reasoning = field(message, "reasoning_content")
    tool_calls = field(message, "tool_calls") or ()
    if message is None:
        # Responses API output: retain metadata without dumping reasoning/output.
        content = field(response, "output_text")
        if not isinstance(content, str):
            content = "".join(
                str(field(block, "text") or "")
                for item in (field(response, "output") or ())
                if field(item, "type") == "message"
                for block in (field(item, "content") or ())
                if field(block, "type") == "output_text"
            )
    usage = field(response, "usage")
    details = field(usage, "completion_tokens_details") or field(usage, "output_tokens_details")
    return {
        "response_id": field(response, "id"),
        "response_model": field(response, "model"),
        "finish_reason": field(choices[0], "finish_reason") if choices else None,
        "response_status": field(response, "status"),
        "incomplete_reason": field(field(response, "incomplete_details"), "reason"),
        "visible_content_chars": len(content) if isinstance(content, str) else None,
        "reasoning_content_chars": len(reasoning) if isinstance(reasoning, str) else None,
        "reasoning_tokens": field(details, "reasoning_tokens"),
        "tool_call_count": len(tool_calls),
        "output_state": "usable" if content or tool_calls else "empty",
        "prompt_tokens": field(usage, "prompt_tokens") if choices else field(usage, "input_tokens"),
        "completion_tokens": field(usage, "completion_tokens") if choices else field(usage, "output_tokens"),
    }
