"""Strict LLM research stages using the existing recorded provider dispatch chain."""

from functools import wraps

from .service_skills import V2_MUTATION_TYPES


class EvolverSchemaError(ValueError):
    """A parsed response violated its stage contract; never a scored rejection."""


def _legacy_schema_checked(method):
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


def _schema_checked(method):
    legacy = _legacy_schema_checked(method)

    @wraps(method)
    def checked(self, *args, **kwargs):
        if self.recovery is None or method.__name__ not in self.recovery.methods:
            return legacy(self, *args, **kwargs)
        return self.recovery.execute(self, method.__name__, args, kwargs,
                                     lambda: legacy(self, *args, **kwargs))
    return checked


DIRECT_MUTATOR_PROMPT = """You are the EvoTau Direct Service Skill Mutator. Analyze observed E failures and
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
Preserve the original factual world, business objective, identity, items/orders/reservations/flights/fares/payment constraints, policy/backend.
Change HOW information, corrections, persistence, timing, clarification and explanations are
communicated, never create a new objective, product, order, reservation, flight, fare, payment, identity, address or factual claim.
Review the challenge archive and probe unsaturated weakness families. Do not repeat semantic
strategies without a genuinely new weakness. Return JSON {"candidates":[{"strategy":"reusable text",
"semantic_family":"...","target_weakness_family":"...","substantive_delta_from_prior":"..."}]}.
Primary selection is native Service accuracy minimization; ties preserve incumbent."""
CUSTOMER_VALIDATOR_PROMPT = """You check Customer interaction policy semantic preservation before rollout.
For EVERY supplied underlying scenario, require the strategy preserve factual truth, business
objective, original identity/items/orders/reservations/flights/fares/payments and backend reality. It may only change interaction
pattern. Reject invented products/orders/reservations/flights/fares/payments/people/addresses/objectives or benchmark internals.
Do not infer reference targets. Return JSON {"preserves_facts":true,"preserves_objective":true,
"interaction_only":true,"no_benchmark_leakage":true,"reason":"..."}. If uncertain use false."""
LEGACY_SKILL_VALIDATOR_PROMPT = """Validate only whether the proposed runtime skill is reusable/domain-general,
subordinate to native policy, free of task IDs, named people, products, orders, reservation IDs, flight numbers, payment IDs, addresses, hidden
objectives and benchmark artifacts. Do not decide effectiveness. Return JSON {"reusable":true,
"policy_subordinate":true,"no_task_entities":true,"reason":"..."}. If uncertain use false."""

FAILURE_ANALYST_PROMPT = """You are the EvoTau Service Failure Analyst.
Your task is to identify distinct, evidence-supported failure hypotheses from observed E-panel agent trajectories. Do NOT generate or modify Skills in this stage.

The purpose of this stage is to determine WHAT might need fixing before deciding HOW to fix it.

### Evidence rules

Use only the supplied E-panel trajectories, observed tool calls and results, public native Airline policy, and permitted historical E-only evidence.

Never use hidden customer objectives, gold actions, evaluator reference solutions, or V/H task information.

A failed benchmark score does not by itself prove a behavioral root cause.

A problematic-looking behavior does not prove it caused the failure.

Separate:
- Directly observed facts.
- Plausible causal hypotheses.
- Unknown or missing information.

Do not infer missing dialogue or tool results from omitted or truncated messages.

Do not infer a root cause from the final tool name alone. Examine the relevant action sequence and observable state transitions.

### Analyze each failure

For each supported hypothesis, identify:

1. OBSERVED DEVIATION:
   The exact assistant decision, tool invocation, tool argument, incorrect assumption, missing prerequisite, or incorrect stopping decision.

2. SOURCE EVIDENCE:
   Cite original task_id, seed, trajectory_ref, projected_message_index, and message_sha256 from supplied retained evidence.
   Never fabricate a citation.

3. HYPOTHESIZED ROOT CAUSE:
   Explain why the observed deviation could prevent successful completion.
   Do not claim causality is established unless the available evidence justifies it.

4. COUNTERFACTUAL BEHAVIOR:
   Explain what the agent should do differently at the same decision point.

5. ALTERNATIVE EXPLANATIONS:
   Identify plausible competing explanations, including run-to-run stochasticity when relevant.

6. REPAIRABILITY:
   Determine whether an external natural-language Skill can plausibly change this behavior.
   Do not propose changes to model weights, native tools, benchmark state, scoring logic, or hidden task facts.

7. REGRESSION RISK:
   Identify successful behaviors or workflows that the proposed correction might damage.

### Diversity and prioritization

Group failures by underlying mechanism, not by wording, task identifier, or superficial symptom.

Do not produce multiple hypotheses that recommend the same operational correction with only different phrasing.

Prefer:
- Specific, observable mistakes.
- Concrete tool or policy misunderstandings.
- Incorrect operation order or missing prerequisites.
- Observable state-management errors.
- Premature stopping or unnecessary escalation, when supported by the trajectory.

Generic claims such as "the agent should be more careful", "follow policy", or "confirm before acting" are insufficient unless tied to a concrete observed deviation.

Output up to the requested number of distinct hypotheses. Fewer hypotheses, including zero, are valid.

Never invent hypotheses merely to fill the candidate budget.

Return strictly structured JSON. Each hypothesis must include:

- `mechanism_id`
- `target_task_ids`
- `protected_success_task_ids`
- `observed_deviation`
- `root_cause_hypothesis`
- `evidence_refs`
- `expected_behavior_change`
- `alternative_explanations`
- `regression_risk`
- `repairability`

Return them under a `hypotheses` array, together with an `insufficient_evidence_reason` string.

`repairability` must be one of:
- `skill`
- `not_skill`
- `uncertain`

Every `evidence_refs` entry must use the five-field reference format already defined by EvoTau.

Use only real references from the supplied context.

Do not output a Skill payload."""

MUTATOR_PROMPT = """You are the EvoTau Evidence-Grounded Service Skill Mutator.
You receive ONE failure hypothesis selected by the EvoTau Failure Analyst, together with its source evidence, successful control cases, current Service SkillMemory, historical E-only mutation outcomes, and native Airline policy.

Your task is to propose ONE minimal, reusable Skill that addresses THIS specific failure mechanism.

You must not choose a different root cause or silently broaden the assigned target.

The Analyst's hypothesis is not ground truth. Recheck its supporting evidence. If the evidence does not support an actionable correction, return NO_OP.

### Step 1: Identify the exact intervention

Before producing a Skill, determine:

- WHEN: At what observable interaction state should this Skill activate?
- WRONG: What specific agent behavior appears incorrect?
- RIGHT: What concrete alternative action should replace it?
- WHY: Why is that alternative consistent with the native Airline policy and observed tool contracts?
- UNLESS: In what conditions should the Skill not activate?
- EXPECT: What measurable tool invocation, argument, execution order, or observable response should change?

If you cannot identify the exact alternative behavior, do not produce a vague reminder.

### Step 2: Check regression risk

Read successful E cases and previous mutation history.

Identify the nearest successful workflows that might be harmed.

Preserve already-correct behavior whenever possible.

Avoid adding confirmations, lookups, or additional reasoning steps that are not necessary for the assigned mechanism.

Do not add a generic confirmation rule if the observed problem is unrelated to confirmation.

Do not repeat a previously rejected correction without a genuinely different operational mechanism supported by new evidence.

### Step 3: Write the smallest useful Skill

The Skill must have:

1. A narrow trigger based on observable state.
2. Concrete operational guidance.
3. Positive activation conditions.
4. Negative activation conditions.
5. Appropriate interaction phases.
6. One falsifiable expected behavioral change.
7. Explicit regression-risk reasoning.

A reusable Skill may reference domain-specific Airline business rules or public tool names.

Domain-specific knowledge is NOT automatically task overfitting.

However, never memorize exact task IDs, individual booking records, hidden evaluator targets, specific fixture answers, customer identities, or private benchmark information in the runtime Skill.

Do not alter native policies, tools, benchmark state, evaluator logic, model weights, or step limits.

Do not invent an authorization, payment method, reservation, or flight.

### Bad vs good repair

BAD:
"Always carefully verify customer requests before making changes."

This is too generic unless an exact observed missing-confirmation decision has been established.

GOOD FORM:
"When observable condition X occurs before operation Y, inspect field Z or perform prerequisite P; otherwise do not perform the action. Preserve successful workflow W."

This is a structural illustration, NOT a claim that X, Y, Z, or P occurred in the supplied task.

The correction must come from actual evidence.

### Output contract

Return exactly the EXISTING EvoTau mutation JSON schema.

Required fields remain:

- `analysis`
- `root_cause_hypothesis`
- `semantic_family`
- `target_cluster_id`
- `operation`
- `target_skill_id`
- `skill`
- `children`
- `evidence_task_ids`
- `protected_success_task_ids`
- `evidence_refs`
- `expected_effect`
- `regression_risk`
- `substantive_delta_from_prior`

Use the existing operation enum and mutation semantics.

The Skill payload must preserve the existing schema:

- `trigger`
- `guidance`
- `activation_signature.positive_conditions`
- `activation_signature.negative_conditions`
- `activation_signature.interaction_phase`

Evidence references must keep the original five fields:

- `task_id`
- `seed`
- `trajectory_ref`
- `projected_message_index`
- `message_sha256`

Every non-NO_OP target must cite an observed failed E condition using a retained original-message reference.

Do not cite omitted messages or invent message hashes.

`evidence_task_ids` must match the assigned hypothesis and observed failed E conditions.

`protected_success_task_ids` must describe actual successful E conditions.

If no safe, specific, evidence-supported intervention is available, return `operation: "no_op"` using the existing NO_OP contract.

The analyst-provided hypothesis and source evidence must constrain the mutation. Do not invent a replacement root cause in the final JSON.
For non-NO_OP, copy assigned_hypothesis.mechanism_id to target_cluster_id, copy its root_cause_hypothesis exactly, and preserve its full target_task_ids. Cite only its assigned evidence_refs. You may return NO_OP without claiming the hypothesis is true.
Operations: add, narrow_trigger, expand_trigger, rewrite_guidance, split, delete, no_op. Skill payload has exactly trigger, guidance, activation_signature with string arrays positive_conditions, negative_conditions, interaction_phase. SPLIT has exactly two children, no skill IDs; NO_OP uses null skill and target_skill_id, empty children and case/ref arrays. Keep root_cause_hypothesis, expected_effect, regression_risk and target_cluster_id nonempty even for NO_OP.
"""

SKILL_VALIDATOR_PROMPT = """A Skill is reusable if its rule applies across multiple possible instances of the same observable operational condition. Domain-specific Airline policies and public tool names are allowed. Reuse does not require domain independence. Validate structure, reuse, policy subordination and leakage only, never effectiveness. Reject hardcoded fixture identities, booking/flight/payment record identifiers, task IDs, gold answers, hidden goals and unsupported authorization/state assumptions. Public tool names and public business rules are permitted. Return exactly JSON {"reusable":true,"policy_subordinate":true,"no_task_entities":true,"reason":"..."}. If uncertain use false."""

LEGACY_MECHANISM_DEDUP_PROMPT = """Compare the supplied failure hypotheses by operational correction, not wording, identifiers or task names. For EVERY unordered pair return JSON {"comparisons":[{"left":"mechanism_id","right":"mechanism_id","same_mechanism":true,"reason":"..."}]}. Use false for distinct mechanisms, null when uncertain. Equivalent rephrasings of the same correction are duplicates. Do not generate skills or invent evidence. This is a semantic judgment, not deterministic proof. Only supplied E evidence is available."""


MECHANISM_DEDUP_PROMPT = """You are the EvoTau Failure Mechanism Deduplication Reviewer.
Compare supplied failure hypotheses according to their proposed operational corrections.
Determine whether they cause essentially the same behavioral change.
IMPORTANT OUTPUT CONTRACT: Return EXACTLY ONE valid top-level JSON object with
EXACTLY ONE field named "comparisons". Its value is ONE array containing ALL pairs.
Do not return a separate object for each pair. Do not concatenate objects, output JSONL,
Markdown, explanations or text outside the object. For EVERY unordered pair of supplied
mechanism IDs include exactly one item with left, right, same_mechanism and reason.
Use true for equivalent operational corrections, false for meaningfully different corrections,
and null when evidence is insufficient. Do not invent evidence or generate Skills.
Example (illustrative IDs only; always use actual supplied IDs):
{"comparisons":[{"left":"h1","right":"h2","same_mechanism":false,"reason":"Different corrections"},
{"left":"h1","right":"h3","same_mechanism":true,"reason":"Equivalent intervention"},
{"left":"h2","right":"h3","same_mechanism":null,"reason":"Insufficient evidence"}]}
If no pairs exist return {"comparisons":[]}."""


class V2Providers:
    def __init__(self, provider):
        self.provider = provider
        self.recovery = None
        self.recovery_attempt = 0
        self.last_recovery = None

    def configure_recovery(self, policy, root, manifest_sha):
        from .evolver_recovery import EvolverRecovery
        if policy.get("evolver_recovery") is not None:
            if policy["algorithm_version"] != "analyst_skill_recovery_v4":
                raise ValueError("automatic recovery cannot change a legacy frozen identity")
            self.recovery = EvolverRecovery(root, manifest_sha, policy["evolver_recovery"])
        else:
            self.recovery = None

    def call(self, prompt, context, name):
        # Recover a completed recorded call if the process died before stage publication.
        import json

        from .provider_diagnostics import safe_request_args
        from .tau_provenance import sha256_json

        if self.recovery_attempt:
            prompt += ("\nFORMAT RECOVERY ATTEMPT " + str(self.recovery_attempt)
                       + ": Return exactly ONE complete JSON object conforming to the ORIGINAL contract. "
                       "Use the same frozen input and scope. Do not add hypotheses, targets, evidence "
                       "or change the assigned root cause. This is a format recovery, not extra search.")
        directory = self.provider.output_directory
        if directory is not None:
            if self.recovery is not None and (directory / "evolver-calls").is_symlink():
                from .evolver_recovery import RecoveryIntegrityError
                raise RecoveryIntegrityError("unsafe evolver call root")
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
                if self.recovery is not None:
                    from .evolver_recovery import replay_failed_or_unknown
                    self.last_call_directory = input_path.parent
                    replay_failed_or_unknown(input_path.parent, name)
        from .alternating import _provider_call

        signal = getattr(self.provider, "stop_before_next_episode_file", None)
        if signal is not None:
            from pathlib import Path

            from .episode_execution import StopBeforeEpisodeDispatch

            if Path(signal).is_symlink():
                raise ValueError("unsafe pause signal")
            if Path(signal).exists():
                raise StopBeforeEpisodeDispatch("paused before next Evolver request")

        try:
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
        finally:
            self.last_call_directory = getattr(self.provider, "last_call_directory", None)
        return result

    @_schema_checked
    def propose_skill_mutation(self, context):
        prompt = MUTATOR_PROMPT if "assigned_hypothesis" in context else DIRECT_MUTATOR_PROMPT
        result = self.call(prompt, context, "evotau_service_skill_mutator")
        validate_mutation(result)
        validate_direct_evidence(result, context)
        from .failure_analysis import validate_assigned_mutation
        validate_assigned_mutation(result, context)
        return result

    @_schema_checked
    def analyze_service_failures(self, context):
        from .failure_analysis import validate_analysis
        result = self.call(FAILURE_ANALYST_PROMPT, context, "evotau_service_failure_analyst")
        return validate_analysis(result, context)

    @_schema_checked
    def deduplicate_failure_hypotheses(self, context):
        from .failure_analysis import validate_dedup_review
        result = self.call(MECHANISM_DEDUP_PROMPT if self.recovery is not None else LEGACY_MECHANISM_DEDUP_PROMPT, context, "evotau_failure_mechanism_dedup")
        return validate_dedup_review(result, context["hypotheses"])

    @_schema_checked
    def crossover(self, context):
        result = self.call(
            CROSSOVER_PROMPT + "\n" + DIRECT_MUTATOR_PROMPT, context, "evotau_skill_crossover"
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
            SKILL_VALIDATOR_PROMPT if context.get("algorithm_version") in ("analyst_skill_v_validation_v3", "analyst_skill_recovery_v4") else LEGACY_SKILL_VALIDATOR_PROMPT,
            context, "evotau_skill_semantic_validator"
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
