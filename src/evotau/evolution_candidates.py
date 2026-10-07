"""Strict LLM research stages using the existing recorded provider dispatch chain."""

from .service_skills import V2_MUTATION_TYPES

DIAGNOSER_PROMPT = """You are the EvoTau Service Failure Diagnoser. Your task is NOT to write a repair.
Group native failures by reusable behavioral/procedural ROOT CAUSE, not task identity or tool name.
Inspect structurally similar passing cases and explicitly identify regression risk. Prior mutation
fixes AND breaks and interaction costs are evidence. Do not convert every failure into a skill.
Return JSON {"clusters":[{"cluster_id":"...","root_cause":"...","evidence_task_ids":[],
"protected_success_task_ids":[],"recommended_surface":"skill|tool_boundary|runtime_protocol|stochastic_or_weak",
"recommended_mutation_types":[],"risk":"..."}]}.
Native policy, tools, backend, evaluation, facts and objectives are immutable. No hidden targets."""
MUTATOR_PROMPT = """You are the EvoTau Service Skill Mutator. Make ONE minimal attributable structural edit.
Use target failures AND protected passing cases, previous fixed/broken cases, native policy,
current accepted memory, failure matrix, and accepted AND rejected mutation effects.
Explain expected fix count and risk count. Rejected repairs with useful local fixes should first
be narrowed or simplified. Do not repeat an equivalent rejected mechanism without a substantive
new mechanism or applicability boundary. Keep edits local; guidance rewrite preserves applicability;
trigger edits preserve guidance. Prefer deletion/simplification over caveats when overhead/max_steps
is the problem. Never add task IDs, people, order IDs, product names, addresses, hidden targets or new
facts/objectives. Guidance is subordinate to native policy and visible backend evidence.
Bias A: narrow applicability; B: minimal behavior and fewer confirmations; C: structural decomposition.
On EXPLORE_ON_STAGNATION choose another cluster, different mutation type or distinct archive mechanism;
prefer SPLIT/DELETE over repeating the same semantic family. NO_OP is valid.
Return JSON {"analysis":"...","semantic_family":"...","target_cluster_id":"...",
"operation":"add|narrow_trigger|expand_trigger|rewrite_guidance|split|delete|no_op",
"target_skill_id":null,"skill":null,"children":[],"expected_fixes":[],"protected_cases_at_risk":[],
"substantive_delta_from_prior":"..."}.
A skill payload has exactly trigger, guidance, activation_signature (positive_conditions,
negative_conditions, interaction_phase arrays). SPLIT uses exactly two child payloads. No skill IDs."""
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
                    return saved["response"]
        from .alternating import _provider_call

        return _provider_call(
            self.provider.request_budget,
            lambda: self.provider._provider_json_call(
                self.provider.model,
                self.provider.model_args,
                prompt,
                context,
                call_name=name,
            ),
        )

    def diagnose(self, context):
        result = self.call(DIAGNOSER_PROMPT, context, "evotau_service_diagnoser")
        allowed = {x["task"]["task_id"] for x in context["task_interactions"]}
        clusters = result.get("clusters")
        if not isinstance(clusters, list):
            raise TypeError("diagnosis must contain clusters")
        identifiers = []
        for cluster in clusters:
            fields = {
                "cluster_id",
                "root_cause",
                "evidence_task_ids",
                "protected_success_task_ids",
                "recommended_surface",
                "recommended_mutation_types",
                "risk",
            }
            if set(cluster) != fields or not all(
                isinstance(cluster[k], str) and cluster[k]
                for k in ("cluster_id", "root_cause", "risk")
            ):
                raise ValueError("invalid root-cause cluster")
            if cluster["recommended_surface"] not in (
                "skill",
                "tool_boundary",
                "runtime_protocol",
                "stochastic_or_weak",
            ):
                raise ValueError("unknown repair surface")
            if not set(cluster["recommended_mutation_types"]) <= set(V2_MUTATION_TYPES):
                raise ValueError("unknown recommended mutation")
            if (
                not set(
                    cluster["evidence_task_ids"] + cluster["protected_success_task_ids"]
                )
                <= allowed
            ):
                raise ValueError("diagnosis references unseen or held-out evidence")
            identifiers.append(cluster["cluster_id"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate cluster IDs")
        return result

    def mutate(self, context):
        result = self.call(MUTATOR_PROMPT, context, "evotau_service_skill_mutator")
        return result

    def crossover(self, context):
        return self.call(
            CROSSOVER_PROMPT + "\n" + MUTATOR_PROMPT, context, "evotau_skill_crossover"
        )

    def customers(self, context, count):
        result = self.call(
            CUSTOMER_PROMPT,
            {**context, "requested_candidates": count},
            "evotau_customer_evolver",
        )
        proposals = result.get("candidates")
        if not isinstance(proposals, list) or len(proposals) != count:
            raise ValueError("wrong Customer candidate count")
        fields = {
            "strategy",
            "semantic_family",
            "target_weakness_family",
            "substantive_delta_from_prior",
        }
        if any(
            set(p) != fields
            or any(not isinstance(v, str) for v in p.values())
            or not p["strategy"].strip()
            for p in proposals
        ):
            raise ValueError("invalid Customer proposal metadata")
        return result

    def validate_customer(self, context):
        return self.call(
            CUSTOMER_VALIDATOR_PROMPT, context, "evotau_customer_semantic_validator"
        )

    def validate_skill(self, context):
        return self.call(
            SKILL_VALIDATOR_PROMPT, context, "evotau_skill_semantic_validator"
        )


def validate_mutation(result):
    required = {
        "analysis",
        "semantic_family",
        "target_cluster_id",
        "operation",
        "target_skill_id",
        "skill",
        "expected_fixes",
        "protected_cases_at_risk",
        "substantive_delta_from_prior",
    }
    if not required <= set(result) or set(result) - required - {"children"}:
        raise ValueError("invalid mutation fields")
    if result["operation"] not in V2_MUTATION_TYPES or any(
        not isinstance(result[k], str)
        for k in (
            "analysis",
            "semantic_family",
            "target_cluster_id",
            "substantive_delta_from_prior",
        )
    ):
        raise ValueError("invalid mutation identity or intent")
    for key in ("expected_fixes", "protected_cases_at_risk"):
        if not isinstance(result[key], list) or any(
            not isinstance(x, str) for x in result[key]
        ):
            raise ValueError("fix/risk cases must be string arrays")
    return result
