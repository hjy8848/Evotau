"""Strict LLM research stages using the existing recorded provider dispatch chain."""

from functools import wraps

from .service_skills import V2_MUTATION_TYPES


class EvolverSchemaError(ValueError):
    """A parsed response violated its stage contract; never a scored rejection."""


def _schema_checked(method):
    @wraps(method)
    def checked(self, *args, **kwargs):
        self.last_call_directory = None
        try:
            return method(self, *args, **kwargs)
        except (ValueError, TypeError, KeyError) as error:
            directory = self.last_call_directory
            if directory is None or not (directory / "output.json").exists():
                raise
            import json

            from .alternating import _write_json_once
            from .provider_diagnostics import safe_error
            from .tau_provenance import sha256_json

            failure = directory / "schema-error.json"
            _write_json_once(
                failure,
                {
                    "failure_type": "EvolverSchemaError",
                    "failure_message": safe_error(error),
                    "input_sha256": sha256_json(
                        json.loads((directory / "input.json").read_text())
                    ),
                    "output_sha256": sha256_json(
                        json.loads((directory / "output.json").read_text())
                    ),
                },
            )
            raised = EvolverSchemaError(str(error))
            raised.call_name = method.__name__
            raised.diagnostics_ref = failure.relative_to(
                self.provider.output_directory
            ).as_posix()
            raise raised from error

    return checked


MUTATOR_PROMPT = """You are the EvoTau Direct Service Skill Mutator. Analyze observed E failures and
passing controls, then propose ONE minimal reusable behavioral repair in this same inference.
No upstream diagnosis or repair-surface label decides whether you may try a skill.
Runtime termination or uncertain reward may still reveal behavioral opportunities, but never
change native task/policy/tools/evaluator/step limits, invent hidden goals, or assert an unseen cause.
Use all E result overviews, retained raw failure evidence and success contrasts, accepted/rejected
mutation effects, current memory, prior fixes/breaks, native policy and the supplied proposal_bias.
Preserve narrow applicability, minimal behavior, and structural decomposition as distinct searches.
On stagnation explore a different hypothesis or archive mechanism. Do not repeat an equivalent
rejected repair without a substantive delta. Keep guidance subordinate to native policy.
Return exactly the structured mutation schema:
{"analysis":"...","root_cause_hypothesis":"observable hypothesis or evidence insufficiency",
"semantic_family":"...","target_cluster_id":"candidate-owned short hypothesis identifier",
"operation":"add|narrow_trigger|expand_trigger|rewrite_guidance|split|delete|no_op",
"target_skill_id":null,"skill":null,"children":[],"evidence_task_ids":[],
"protected_success_task_ids":[],"evidence_refs":[],"expected_effect":"...",
"regression_risk":"...","substantive_delta_from_prior":"..."}.
Each evidence_ref has exactly task_id, seed, trajectory_ref, projected_message_index,
message_sha256 copied from a supplied retained raw message and its containing row.
Targets must be observed failed E tasks; controls must be observed successful E tasks.
Every non-NO_OP target must have at least one such original-message reference. Never cite omitted
messages or fabricate indices/hashes. Partial excerpts/JSON field projections are explicitly marked:
do not infer omitted fields. References may only support declared targets or controls.
Skill payloads have exactly trigger, guidance, activation_signature (positive_conditions,
negative_conditions, interaction_phase arrays). SPLIT has exactly two children, no skill IDs.
NO_OP is valid: use null skill/target_skill_id, no children, explain insufficiency in
root_cause_hypothesis/expected_effect/regression_risk; empty case/ref arrays are valid.
Do not force a Skill when evidence is insufficient. No hidden scenarios, gold targets or V/H data.
"""
CROSSOVER_PROMPT = """You are the EvoTau Skill Crossover Architect. Given two non-deployed mutations
with complementary validated local fixes, synthesize the smallest coherent child. Identify each
parent's fixes, regressions, useful mechanism and unsafe scope. Preserve only evidence-supported
mechanisms; never concatenate guidance. Return the normal structured mutation schema. The child
must undergo independent screen and full gate. Native policy/facts/objectives remain unchanged."""
CUSTOMER_PROMPT = """You evolve the Customer as an adversarial but task-faithful interaction policy.
Preserve the original factual world, business objective, identity, items/orders, policy/backend.
Change HOW information, corrections, persistence, timing, clarification and explanations are
communicated, never create a new objective, product, order, identity, address or factual claim.
Review the challenge archive and probe unsaturated weakness families. Do not repeat semantic
strategies without a genuinely new weakness. Return JSON {"candidates":[{"strategy":"reusable text",
"semantic_family":"...","target_weakness_family":"...","substantive_delta_from_prior":"..."}]}.
Primary selection is native Service accuracy minimization; ties preserve incumbent."""
CUSTOMER_VALIDATOR_PROMPT = """You check Customer interaction policy semantic preservation before rollout.
For EVERY supplied underlying scenario, require the strategy preserve factual truth, business
objective, original identity/items/orders and backend reality. It may only change interaction
pattern. Reject invented products/order/person/addresses/objectives or benchmark internals.
Do not infer reference targets. Return JSON {"preserves_facts":true,"preserves_objective":true,
"interaction_only":true,"no_benchmark_leakage":true,"reason":"..."}. If uncertain use false."""
SKILL_VALIDATOR_PROMPT = """Validate only whether the proposed runtime skill is reusable/domain-general,
subordinate to native policy, free of task IDs, named people, products, orders, addresses, hidden
objectives and benchmark artifacts. Do not decide effectiveness. Return JSON {"reusable":true,
"policy_subordinate":true,"no_task_entities":true,"reason":"..."}. If uncertain use false."""


class V2Providers:
    def __init__(self, provider):
        self.provider = provider

    def call(self, prompt, context, name):
        # Recover a completed recorded call if the process died before stage publication.
        import json

        from .provider_diagnostics import safe_request_args
        from .tau_provenance import sha256_json

        directory = self.provider.output_directory
        if directory is not None:
            for input_path in sorted(
                (directory / "evolver-calls").glob("*/input.json")
            ):
                output_path = input_path.parent / "output.json"
                if input_path.is_symlink() or output_path.is_symlink():
                    raise ValueError("unsafe V2 provider call cache")
                data = json.loads(input_path.read_text())
                if (
                    data.get("call_name"),
                    data.get("model"),
                    data.get("system_prompt"),
                    data.get("request_args"),
                    data.get("input_sha256"),
                ) != (
                    name,
                    self.provider.model,
                    prompt,
                    safe_request_args(self.provider.model_args),
                    sha256_json(context),
                ):
                    continue
                if data.get("input_sha256") != sha256_json(data["context"]):
                    raise ValueError("saved V2 provider input digest mismatch")
                if output_path.is_file():
                    saved = json.loads(output_path.read_text())
                    if saved.get("response_sha256") != sha256_json(
                        saved.get("response")
                    ):
                        raise ValueError("saved V2 provider output digest mismatch")
                    from .release_recovery import schema_retry_authorized

                    if schema_retry_authorized(directory, input_path.parent):
                        continue
                    self.last_call_directory = input_path.parent
                    return saved["response"]
        from .alternating import _provider_call

        signal = getattr(self.provider, "stop_before_next_episode_file", None)
        if signal is not None:
            from pathlib import Path

            from .episode_execution import StopBeforeEpisodeDispatch

            if Path(signal).is_symlink():
                raise ValueError("unsafe pause signal")
            if Path(signal).exists():
                raise StopBeforeEpisodeDispatch("paused before next Evolver request")

        result = _provider_call(
            self.provider.request_budget,
            lambda: self.provider._provider_json_call(
                self.provider.model,
                self.provider.model_args,
                prompt,
                context,
                call_name=name,
            ),
        )
        self.last_call_directory = getattr(self.provider, "last_call_directory", None)
        return result

    @_schema_checked
    def propose_skill_mutation(self, context):
        result = self.call(MUTATOR_PROMPT, context, "evotau_service_skill_mutator")
        validate_mutation(result)
        validate_direct_evidence(result, context)
        return result

    @_schema_checked
    def crossover(self, context):
        result = self.call(
            CROSSOVER_PROMPT + "\n" + MUTATOR_PROMPT, context, "evotau_skill_crossover"
        )
        validate_mutation(result)
        validate_direct_evidence(result, context)
        return result

    @_schema_checked
    def customers(self, context, count):
        result = self.call(
            CUSTOMER_PROMPT,
            {**context, "requested_candidates": count},
            "evotau_customer_evolver",
        )
        proposals = result.get("candidates")
        if (
            set(result) != {"candidates"}
            or not isinstance(proposals, list)
            or len(proposals) != count
        ):
            raise ValueError("wrong Customer candidate count")
        fields = {
            "strategy",
            "semantic_family",
            "target_weakness_family",
            "substantive_delta_from_prior",
        }
        if any(
            not isinstance(p, dict)
            or set(p) != fields
            or any(not isinstance(v, str) for v in p.values())
            or not p["strategy"].strip()
            for p in proposals
        ):
            raise ValueError("invalid Customer proposal metadata")
        return result

    @_schema_checked
    def validate_customer(self, context):
        result = self.call(
            CUSTOMER_VALIDATOR_PROMPT, context, "evotau_customer_semantic_validator"
        )
        return _validator_result(
            result,
            (
                "preserves_facts",
                "preserves_objective",
                "interaction_only",
                "no_benchmark_leakage",
            ),
        )

    @_schema_checked
    def validate_skill(self, context):
        result = self.call(
            SKILL_VALIDATOR_PROMPT, context, "evotau_skill_semantic_validator"
        )
        return _validator_result(
            result,
            (
                "reusable",
                "policy_subordinate",
                "no_task_entities",
            ),
        )


def _validator_result(result, flags):
    if (
        not isinstance(result, dict)
        or set(result) != {*flags, "reason"}
        or any(type(result[k]) is not bool for k in flags)
        or not isinstance(result["reason"], str)
    ):
        raise ValueError("semantic validator returned an invalid structured schema")
    return result


def validate_direct_evidence(result, context):
    """Bind hypotheses to supplied observed E cells; never reinterpret provider output."""
    rows = context["task_interactions"]
    failed = {
        row["task"]["task_id"]
        for row in rows
        if row["native_evaluation"]["task_success"] is False
    }
    passed = {
        row["task"]["task_id"]
        for row in rows
        if row["native_evaluation"]["task_success"] is True
    }
    if (
        not set(result["evidence_task_ids"]) <= failed
        or not set(result["protected_success_task_ids"]) <= passed
    ):
        raise ValueError(
            "mutation target/protected labels disagree with observed E outcomes"
        )
    registry = {}
    for row in rows:
        outcome = row["native_evaluation"]["task_success"]
        if type(outcome) is not bool:
            raise ValueError("mutation evidence requires known boolean outcomes")
        for item in row["trajectory"]["messages"]:
            ref = item["evidence_ref"]
            key = (
                row["task"]["task_id"],
                row["seed"],
                row["trajectory_ref"],
                ref["projected_message_index"],
                ref["message_sha256"],
            )
            registry[key] = outcome
    fields = (
        "task_id",
        "seed",
        "trajectory_ref",
        "projected_message_index",
        "message_sha256",
    )
    cited = set()
    for ref in result["evidence_refs"]:
        if (
            not isinstance(ref, dict)
            or set(ref) != set(fields)
            or type(ref["seed"]) is not int
            or type(ref["projected_message_index"]) is not int
            or any(
                not isinstance(ref[key], str)
                for key in ("task_id", "trajectory_ref", "message_sha256")
            )
        ):
            raise ValueError("invalid evidence reference schema")
        if tuple(ref[key] for key in fields) not in registry:
            raise ValueError("evidence reference is not a supplied original message")
        if ref["task_id"] not in set(
            result["evidence_task_ids"] + result["protected_success_task_ids"]
        ):
            raise ValueError(
                "evidence reference must support a declared target or control"
            )
        observed_success = registry[tuple(ref[key] for key in fields)]
        expected_ids = (
            result["protected_success_task_ids"]
            if observed_success
            else result["evidence_task_ids"]
        )
        if ref["task_id"] not in expected_ids:
            raise ValueError("evidence cell outcome disagrees with target/control role")
        if not observed_success:
            cited.add(ref["task_id"])
    if result["operation"] != "no_op" and (
        not result["evidence_task_ids"] or not set(result["evidence_task_ids"]) <= cited
    ):
        raise ValueError(
            "non-NO_OP mutation requires cited failure evidence for every target"
        )
    return result


def validate_mutation(result):
    required = {
        "analysis",
        "semantic_family",
        "target_cluster_id",
        "operation",
        "target_skill_id",
        "skill",
        "evidence_task_ids",
        "protected_success_task_ids",
        "root_cause_hypothesis",
        "evidence_refs",
        "expected_effect",
        "regression_risk",
        "substantive_delta_from_prior",
    }
    if (
        not isinstance(result, dict)
        or not required <= set(result)
        or set(result) - required - {"children"}
    ):
        raise ValueError("invalid mutation fields")
    if result["operation"] not in V2_MUTATION_TYPES or any(
        not isinstance(result[k], str)
        for k in (
            "analysis",
            "semantic_family",
            "target_cluster_id",
            "substantive_delta_from_prior",
            "root_cause_hypothesis",
            "expected_effect",
            "regression_risk",
        )
    ):
        raise ValueError("invalid mutation identity or intent")
    for key in ("evidence_task_ids", "protected_success_task_ids"):
        if not isinstance(result[key], list) or any(
            not isinstance(x, str) for x in result[key]
        ):
            raise ValueError("fix/risk cases must be string arrays")
    if any(
        not result[key].strip()
        for key in (
            "root_cause_hypothesis",
            "expected_effect",
            "regression_risk",
            "target_cluster_id",
        )
    ):
        raise ValueError("mutation hypotheses/effect/risk must be explicit")
    if any(char in result["target_cluster_id"] for char in ("/", "\\", "\x00")):
        raise ValueError("candidate-owned cluster ID cannot contain path separators")
    if not isinstance(result["evidence_refs"], list):
        raise TypeError("evidence_refs must be an array")
    from .service_skills import ServiceSkillV2, _validate_v2_draft

    target = result["target_skill_id"]
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ValueError("target_skill_id must be a nonempty string or null")
    children = result.get("children", [])
    if not isinstance(children, list):
        raise TypeError("children must be an array")
    operation = result["operation"]
    if operation in ("delete", "no_op"):
        if result["skill"] is not None or children:
            raise ValueError("DELETE/NO_OP cannot carry skill payloads")
    elif operation == "split":
        if result["skill"] is not None or len(children) != 2:
            raise ValueError("SPLIT requires exactly two children and null skill")
    elif children:
        raise ValueError("only SPLIT may carry children")
    if operation in ("add", "no_op") and target is not None:
        raise ValueError("ADD/NO_OP cannot target a skill")
    if operation not in ("add", "no_op") and target is None:
        raise ValueError("an edit must name its target skill")
    payloads = (
        children
        if operation == "split"
        else ([result["skill"]] if operation not in ("delete", "no_op") else [])
    )
    for payload in payloads:
        _validate_v2_draft(payload)
        ServiceSkillV2.from_mapping({"skill_id": "skill-0001", **payload})
    return result
