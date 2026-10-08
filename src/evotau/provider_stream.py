"""Collect one research completion, keeping partial streams out of fitness."""

from collections.abc import Mapping


def _field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


class IncompleteProviderStream(RuntimeError):
    def __init__(self, reason, text):
        super().__init__(reason)
        self.raw_response = text[:16000]
        self.raw_response_truncated = len(text) > 16000


def collect_completion(stream, builder):
    chunks, texts, endings = [], [], []
    usage = None
    try:
        for chunk in stream:
            chunks.append(chunk)
            if _field(chunk, "error"):
                raise RuntimeError("provider emitted a stream error")
            if _field(chunk, "usage") is not None:
                usage = _field(chunk, "usage")
            for choice in _field(chunk, "choices") or ():
                if _field(choice, "index") not in (None, 0):
                    raise RuntimeError("research stream returned multiple choices")
                delta = _field(choice, "delta")
                if _field(delta, "tool_calls"):
                    raise RuntimeError("research stream unexpectedly called tools")
                text = _field(delta, "content")
                if text:
                    if endings:
                        raise RuntimeError("visible output arrived after stream termination")
                    texts.append(text)
                end = _field(choice, "finish_reason")
                if end:
                    endings.append(end)
        if endings != ["stop"]:
            raise RuntimeError("provider stream has no unique finish_reason=stop")
        if not "".join(texts).strip():
            raise RuntimeError("provider stream returned no visible completion")
        response = builder(chunks)
        # Never substitute tokenizer estimates for missing provider usage.
        if isinstance(response, dict):
            response["usage"] = usage
        else:
            response.usage = usage
        return response
    except Exception as exc:
        raise IncompleteProviderStream(str(exc), "".join(texts)) from exc
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
