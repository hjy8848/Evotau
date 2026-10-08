"""Explicit recovery operations; immutable failed evidence is never rewritten."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .tau_provenance import sha256_json


def _retry_binding(root, directory):
    def read(name):
        path = directory / name
        if path.is_symlink() or directory.is_symlink():
            raise ValueError("unsafe recovery artifact")
        return json.loads(path.read_text())

    manifest = json.loads((Path(root) / "manifest.json").read_text())
    error = read("schema-error.json")
    inputs, output = read("input.json"), read("output.json")
    if (
        error["input_sha256"] != sha256_json(inputs)
        or error["output_sha256"] != sha256_json(output)
        or output["response_sha256"] != sha256_json(output["response"])
    ):
        raise ValueError("schema recovery evidence digest mismatch")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "input_sha256": sha256_json(inputs),
        "output_sha256": sha256_json(output),
        "schema_error_sha256": sha256_json(error),
    }


def schema_retry_authorized(root, directory):
    path = directory / "retry-authorization.json"
    if not path.exists():
        return False
    if path.is_symlink():
        raise ValueError("unsafe retry authorization")
    document = json.loads(path.read_text())
    if document["binding"] != _retry_binding(root, directory) or document[
        "authorization_sha256"
    ] != sha256_json(
        {k: v for k, v in document.items() if k != "authorization_sha256"}
    ):
        raise ValueError("retry authorization binding/digest mismatch")
    return True


def authorize_schema_retry(root, call_id, reason):
    from .alternating import _write_json_once

    if not reason.strip() or Path(call_id).name != call_id or call_id in (".", ".."):
        raise ValueError(
            "a call directory ID and nonempty operator reason are required"
        )
    directory = Path(root) / "evolver-calls" / call_id
    document = {
        "binding": _retry_binding(root, directory),
        "at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "operation": "authorize one replacement of this invalid parsed response",
        "output_modified": False,
    }
    document["authorization_sha256"] = sha256_json(document)
    _write_json_once(directory / "retry-authorization.json", document)
    return document


@contextmanager
def frozen_run_lock(checkpoint_path):
    """One writer per checkpoint, including recovery; OS releases lock after death."""
    import fcntl

    path = Path(checkpoint_path).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("unsafe run lock")
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "this frozen experiment already has an active writer"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def reconcile_interrupted_requests(root, reason):
    """Explicit conservative charge, never a fabricated benchmark outcome or token count."""
    from copy import deepcopy

    from .alternating import _write_json_atomic, _write_json_once
    from .budget import BudgetSnapshot

    root = Path(root)
    path = root / "api-usage-live.json"
    if not reason.strip() or path.is_symlink():
        raise ValueError("reconciliation requires a reason and regular accounting file")
    before = json.loads(path.read_text())
    usage = before["provider_usage"]
    pending = usage["in_flight"]
    if not pending or usage["reserved"]:
        raise ValueError(
            "only unresolved dispatched requests with no unconsumed reservations can be reconciled"
        )
    audit_path = root / "recovery" / (sha256_json(before) + "-accounting.json")
    after = deepcopy(before)
    totals = after["provider_usage"]
    for counters in (totals, *totals["model_usage"]):
        count = counters["in_flight"]
        counters["attempts"] += count
        counters["failures"] += count
        counters["usage_unavailable"] += count
        counters["in_flight"] = 0
    calls = after["api_usage_by_call_name"]
    row = calls.setdefault(
        "evotau_interrupted_unknown",
        dict.fromkeys(
            (
                "calls",
                "successes",
                "failures",
                "prompt_tokens",
                "completion_tokens",
                "usage_responses",
                "usage_unavailable",
                "total_elapsed_seconds",
            ),
            0,
        ),
    )
    for key in ("calls", "failures", "usage_unavailable"):
        row[key] += pending
    # Also cover a kill after aggregate completion but before call-name persistence.
    gap = totals["attempts"] - sum(c["calls"] for c in calls.values())
    if gap < 0:
        raise ValueError("call-name counters exceed global request accounting")
    for key in ("calls", "failures", "usage_unavailable"):
        row[key] += gap
    BudgetSnapshot(**totals)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = {
        "manifest_sha256": manifest["manifest_sha256"],
        "at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "disposition": "unknown possibly billed calls charged; transport failures only, no benchmark score",
        "before": before,
        "after": after,
        "before_sha256": sha256_json(before),
        "after_sha256": sha256_json(after),
    }
    if audit_path.exists():
        saved = json.loads(audit_path.read_text())
        if (
            saved["before"] != before
            or saved["after"] != after
            or saved["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise ValueError(
                "interrupted reconciliation audit does not match accounting"
            )
    else:
        _write_json_once(audit_path, audit)
    _write_json_atomic(path, after)
    return {"charged_unknown_requests": pending + gap, "audit": str(audit_path)}
