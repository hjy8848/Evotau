"""Versioned format recovery: immutable attempts, exact resume, conservative exhaustion.

Only JSON and returned schema errors are eligible. A submitted request without a durable
response, transport errors, budget failures and integrity errors are never auto-retried.
"""

import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .evolution_artifacts import EvolutionJournal
from .provider_diagnostics import safe_error, safe_request_args
from .tau_provenance import sha256_json

DEFAULT_RECOVERY = {"protocol_version": "format_recovery_v1", "max_format_recoveries": 2}


class RecoveryExhausted(RuntimeError):
    def __init__(self, outcome):
        super().__init__("RECOVERY_EXHAUSTED: " + outcome["reason"])
        self.outcome = outcome
        self.diagnostics_ref = outcome["diagnostics_ref"]


class UnknownRequestState(RuntimeError):
    """Possibly submitted/charged request; requires explicit operator reconciliation."""


class RecoveryIntegrityError(RuntimeError):
    """Never convert input/cache/journal contamination into a candidate rejection."""


def validate_recovery_policy(policy):
    if (not isinstance(policy, dict) or set(policy) != set(DEFAULT_RECOVERY)
            or policy["protocol_version"] != DEFAULT_RECOVERY["protocol_version"]
            or type(policy["max_format_recoveries"]) is not int
            or not 0 <= policy["max_format_recoveries"] <= 2):
        raise ValueError("invalid frozen format recovery policy (0..2 extra attempts)")


def replay_failed_or_unknown(directory, name):
    """Do not blindly repeat a known failed or potentially submitted call on resume."""
    from .alternating import EvolverJSONError

    failure = directory / "failure.json"
    if not failure.exists():
        raise UnknownRequestState(f"unresolved provider dispatch: {directory.name}")
    if failure.is_symlink():
        raise RecoveryIntegrityError("unsafe failed request evidence")
    data = json.loads(failure.read_text())
    if data["failure_type"] == "EvolverJSONError":
        raw = data.get("raw_response") or ""
        visible = directory / "visible-completion.json"
        if visible.exists():
            document = json.loads(visible.read_text())
            raw = document["visible_text"]
            if document["visible_text_sha256"] != sha256_json(raw):
                raise RecoveryIntegrityError("failed visible output digest mismatch")
        error = EvolverJSONError(call_name=name, raw_response=raw,
                                 parse_error=data["failure_message"])
        error.diagnostics_ref = str(failure)
        raise error
    raise UnknownRequestState("previous provider failure requires explicit reconciliation: "
                              + data["failure_type"])


def guard_service_context(context):
    """Reject explicit private/evaluation payloads before any recovery request."""
    forbidden = {"user_scenario", "reference_actions", "evaluation_criteria", "gold_targets",
                 "heldout_episodes", "validation_episodes", "heldout_tasks", "validation_tasks"}

    def walk(value):
        if isinstance(value, dict):
            if forbidden & value.keys():
                raise RecoveryIntegrityError("hidden scenario or V/H evaluation payload in Service context")
            if value.get("panel") in ("V", "H"):
                raise RecoveryIntegrityError("V/H evidence in Service context")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
    walk(context)



class EvolverRecovery:
    methods = frozenset({"analyze_service_failures", "deduplicate_failure_hypotheses", "propose_skill_mutation"})

    def __init__(self, root, manifest_sha, policy):
        validate_recovery_policy(policy)
        self.root = Path(root)
        self.policy = dict(policy)
        self.manifest_sha = manifest_sha
        if self.root.is_symlink() or (self.root / "format-recovery").is_symlink():
            raise RecoveryIntegrityError("unsafe recovery journal root")
        # Separate envelopes, same journal consistency/atomic immutable publication.
        self.journal = EvolutionJournal(self.root / "format-recovery", manifest_sha)

    def execute(self, owner, method, args, kwargs, callback):
        from .alternating import EvolverJSONError
        from .evolution_candidates import EvolverSchemaError

        guard_service_context(args[0] if args else kwargs.get("context", {}))
        binding = {"protocol": self.policy, "method": method, "args": args, "kwargs": kwargs,
                   "model": owner.provider.model,
                   "model_args": safe_request_args(owner.provider.model_args),
                   "manifest_sha256": self.manifest_sha}
        logical_id = sha256_json(binding)
        final_name = logical_id + "-result"
        final_path = self.journal.root / (final_name + ".json")
        if final_path.exists():
            doc = self.journal.read(final_path)
            if doc["input_sha256"] != sha256_json(binding):
                raise RecoveryIntegrityError("recovery binding changed")
            return self._return(owner, doc["payload"])
        parent_request = None
        attempts = []
        for index in range(self.policy["max_format_recoveries"] + 1):
            attempt_inputs = {"binding": binding, "attempt_index": index,
                              "parent_request": parent_request}
            attempt_id = logical_id + f"-attempt-{index}"
            # Intent survives interruption. The V2 cache only permits reuse of durable responses.
            self.journal.freeze(attempt_id + "-intent", attempt_inputs,
                                lambda ident=attempt_id: {"status": "DISPATCH_INTENT", "request_id": ident})

            def perform(index=index, parent_request=parent_request):
                owner.recovery_attempt = index
                started = perf_counter()
                try:
                    value = callback()  # Includes ALL original schema/business validation.
                    status, error, reason = "VALID", None, None
                except (EvolverJSONError, EvolverSchemaError) as exc:
                    # No generic ValueError/Exception catch: cache/config/integrity failures escape.
                    value, status, error, reason = None, "REJECTED", type(exc).__name__, safe_error(exc)
                except BaseException as exc:
                    # Evidence-only catch: always re-raise; never reinterpret as a scored failure.
                    from .alternating import _write_json_once
                    _write_json_once(self.root / "format-recovery" / "infrastructure" / (uuid4().hex + ".json"),
                                     {"status": "FAILED_INFRASTRUCTURE", "logical_call_id": logical_id,
                                      "attempt_index": index, "error_type": type(exc).__name__,
                                      "reason": safe_error(exc), "automatic_retry": False})
                    raise
                finally:
                    owner.recovery_attempt = 0
                directory = owner.last_call_directory
                if directory is None:
                    raise RecoveryIntegrityError("recoverable output lacks immutable provider evidence")
                input_doc = json.loads((directory / "input.json").read_text())
                output_path = directory / "output.json"
                visible_path = directory / "visible-completion.json"
                output_doc = json.loads(output_path.read_text()) if output_path.exists() else None
                visible_doc = json.loads(visible_path.read_text()) if visible_path.exists() else None
                calls_file = directory / "provider-calls.jsonl"
                records = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
                usage = {"calls": len(records),
                         "prompt_tokens": sum((r.get("response") or {}).get("prompt_tokens") or 0 for r in records),
                         "completion_tokens": sum((r.get("response") or {}).get("completion_tokens") or 0 for r in records),
                         "elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in records),
                         "usage_available": bool(records)}
                return {"status": status, "response": value, "error_type": error, "reason": reason,
                        "request_id": directory.name, "parent_request": parent_request,
                        "attempt_index": index, "input_digest": sha256_json(input_doc),
                        "output_digest": sha256_json(output_doc if output_doc is not None else visible_doc),
                        "provider_call_ref": str(directory.relative_to(self.root)),
                        "elapsed_seconds": perf_counter() - started,
                        "semantic_equivalence_certified": False, "provider_usage": usage,
                        "provider_records_sha256": sha256_json(records),
                        "evidence_digests": {f.name: sha256_json(json.loads(f.read_text()))
                            for f in directory.glob("*.json") if f.name in
                            {"input.json", "output.json", "visible-completion.json", "failure.json",
                             "schema-error.json", "response-metadata.json"}}}

            attempt = self.journal.freeze(attempt_id, attempt_inputs, perform)
            # Validate source evidence even when the stage/call result itself was cached.
            self.verify_attempt(attempt)
            attempts.append(attempt)
            parent_request = attempt["request_id"]
            if attempt["status"] == "VALID":
                outcome = {"status": "RECOVERED" if index else "VALID",
                           "response": attempt["response"], "recovery_attempts": index,
                           "attempts": attempts, "logical_call_id": logical_id,
                           "diagnostics_ref": str(final_path.relative_to(self.root))}
                self.journal.freeze(final_name, binding, lambda result=outcome: result)
                return self._return(owner, outcome)
        outcome = {"status": "RECOVERY_EXHAUSTED", "reason": attempts[-1]["reason"],
                   "response": None, "recovery_attempts": len(attempts) - 1,
                   "attempts": attempts, "logical_call_id": logical_id,
                   "diagnostics_ref": str(final_path.relative_to(self.root))}
        self.journal.freeze(final_name, binding, lambda result=outcome: result)
        return self._return(owner, outcome)

    def verify_attempt(self, attempt):
        ref = Path(attempt["provider_call_ref"])
        if ref.is_absolute() or ".." in ref.parts or ref.parts[0] != "evolver-calls":
            raise RecoveryIntegrityError("unsafe recovery provider reference")
        directory = self.root / ref
        for name in ("input.json", "output.json", "visible-completion.json", "provider-calls.jsonl"):
            if (directory / name).is_symlink() or directory.is_symlink():
                raise RecoveryIntegrityError("unsafe recovery source")
        calls_file = directory / "provider-calls.jsonl"
        records = [json.loads(line) for line in calls_file.read_text().splitlines()] if calls_file.exists() else []
        if sha256_json(records) != attempt["provider_records_sha256"]:
            raise RecoveryIntegrityError("recovery provider ledger digest mismatch")
        for name, digest in attempt["evidence_digests"].items():
            file = directory / name
            if file.is_symlink() or sha256_json(json.loads(file.read_text())) != digest:
                raise RecoveryIntegrityError("recovery source evidence digest mismatch")
        inp = json.loads((directory / "input.json").read_text())
        if sha256_json(inp) != attempt["input_digest"] or inp["input_sha256"] != sha256_json(inp["context"]):
            raise RecoveryIntegrityError("recovery input digest mismatch")
        target = directory / "output.json"
        output = json.loads(target.read_text()) if target.exists() else json.loads((directory / "visible-completion.json").read_text())
        if sha256_json(output) != attempt["output_digest"]:
            raise RecoveryIntegrityError("recovery output digest mismatch")

    def _return(self, owner, outcome):
        for attempt in outcome["attempts"]:
            self.verify_attempt(attempt)
            if (self.root / attempt["provider_call_ref"] / "retry-authorization.json").exists():
                raise RecoveryIntegrityError("manual retry cannot override a frozen automatic recovery budget; use explicit new identity")
        owner.last_recovery = {k: v for k, v in outcome.items() if k != "response"}
        if outcome["status"] == "RECOVERY_EXHAUSTED":
            raise RecoveryExhausted(owner.last_recovery)
        return outcome["response"]


def summarize_recovery(root, manifest_sha):
    journal = EvolutionJournal(Path(root) / "format-recovery", manifest_sha)
    outcomes = [journal.read(p)["payload"] for p in sorted(journal.root.glob('*-result.json'))]
    attempts = [a for o in outcomes for a in o["attempts"]]
    extras = [a for a in attempts if a["attempt_index"]]
    service_journal = EvolutionJournal(root, manifest_sha)
    stages = [service_journal.read(p)["payload"] for p in service_journal.root.glob("*.json")]
    return {"protocol_version": "format_recovery_v1", "logical_calls": len(outcomes),
            "original_valid_rate": sum(o["status"] == "VALID" for o in outcomes) / len(outcomes) if outcomes else None,
            "recovery_success_rate": sum(o["status"] == "RECOVERED" for o in outcomes) /
                sum(o["status"] != "VALID" for o in outcomes) if any(o["status"] != "VALID" for o in outcomes) else None,
            "degraded_stage_count": sum(s.get("status") == "DEGRADED" for s in stages if isinstance(s, dict)),
            "extra_provider_usage": {k: sum(a["provider_usage"][k] for a in extras)
                                     for k in ("calls", "prompt_tokens", "completion_tokens", "elapsed_seconds")},
            "original_valid_calls": sum(o["status"] == "VALID" for o in outcomes),
            "recovered_calls": sum(o["status"] == "RECOVERED" for o in outcomes),
            "exhausted_calls": sum(o["status"] == "RECOVERY_EXHAUSTED" for o in outcomes),
            "format_recovery_requests": sum(o["recovery_attempts"] for o in outcomes),
            "recovery_attempt_elapsed_seconds": sum(a["elapsed_seconds"] for a in attempts if a["attempt_index"]),
            "interpretation": "format regeneration is additional sampling, not Analyst effectiveness evidence"}
