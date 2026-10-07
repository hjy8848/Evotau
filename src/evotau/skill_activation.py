"""Per-turn activation over Service-visible conversation, never hidden benchmark metadata."""

import math
from dataclasses import dataclass

from .service_skills import render_selected_service_skills, skill_token_count
from .tau_provenance import sha256_json

ACTIVATOR_PROMPT = """You are the EvoTau Service Skill Activator.
Decide which already-accepted procedural skills apply to the CURRENT observable interaction.
You are not the Service agent; do not answer the customer, modify policy, invent backend facts,
or infer hidden benchmark objectives. Only conversation, tool results and trigger/signature
catalogs are supplied. Select at most {max_skills} IDs. Negative conditions override positive
conditions. Broad domain similarity or a keyword such as exchange is insufficient. Prefer no
skill over weak matches. Return JSON only:
{{"active_skill_ids":[],"reason":"...","confidence":0.0}}."""


def project_observable_messages(messages):
    projected = []
    for message in messages:
        raw = (
            message.model_dump(mode="json")
            if hasattr(message, "model_dump")
            else message
        )
        if raw.get("role") == "multi_tool":
            projected.extend(project_observable_messages(raw["tool_messages"]))
            continue
        if raw.get("role") not in ("user", "assistant", "tool"):
            continue
        row = {
            k: raw[k] for k in ("role", "content", "name", "tool_call_id") if k in raw
        }
        if raw.get("tool_calls"):
            row["tool_calls"] = [
                {k: tool[k] for k in ("id", "name", "arguments") if k in tool}
                for tool in raw["tool_calls"]
            ]
        projected.append(row)
    return projected


@dataclass(frozen=True, slots=True)
class SkillActivationDecision:
    active_skill_ids: tuple[str, ...]
    reason: str = ""
    confidence: float | None = None

    def to_dict(self):
        return {
            "active_skill_ids": list(self.active_skill_ids),
            "reason": self.reason,
            "confidence": self.confidence,
        }


class SkillActivator:
    def __init__(self, *, model, model_args, request_budget=None, json_call=None):
        self.model, self.model_args = model, dict(model_args)
        self.request_budget, self.json_call = request_budget, json_call

    def select(self, observable_context, active_skill_catalog, max_active_skills=2):
        if set(observable_context) != {"messages"}:
            raise ValueError("activator context must contain only observable messages")
        context = {
            "messages": project_observable_messages(observable_context["messages"]),
            "catalog": [
                {
                    "skill_id": s.skill_id,
                    "trigger": s.trigger,
                    "activation_signature": s.activation_signature.to_dict(),
                }
                for s in active_skill_catalog
            ],
        }
        if not 1 <= max_active_skills <= 2:
            raise ValueError("invalid activation K")
        if not active_skill_catalog:
            return SkillActivationDecision(
                (), "empty active catalog; no provider request", 1.0
            ), context
        if self.json_call is None:
            from .alternating import LLMAlternatingEvolvers

            dispatcher = LLMAlternatingEvolvers(
                model=self.model,
                model_args=self.model_args,
                request_budget=self.request_budget,
            )
            result = dispatcher._provider_json_call(
                self.model,
                self.model_args,
                ACTIVATOR_PROMPT.format(max_skills=max_active_skills),
                context,
                call_name="evotau_skill_activator",
            )
        else:
            result = self.json_call(
                ACTIVATOR_PROMPT.format(max_skills=max_active_skills), context
            )
        if not isinstance(result, dict) or set(result) - {
            "active_skill_ids",
            "reason",
            "confidence",
        }:
            raise ValueError("invalid activation JSON")
        ids = result.get("active_skill_ids")
        if (
            not isinstance(ids, list)
            or any(not isinstance(x, str) for x in ids)
            or len(ids) > max_active_skills
            or len(set(ids)) != len(ids)
            or not set(ids) <= {s.skill_id for s in active_skill_catalog}
        ):
            raise ValueError("activation IDs violate catalog or K")
        confidence = result.get("confidence")
        if confidence is not None and (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid activation confidence")
        if not isinstance(result.get("reason", ""), str):
            raise TypeError("invalid activation reason")
        return SkillActivationDecision(
            tuple(ids), result.get("reason", ""), confidence
        ), context


class RenderAllSkillSelector:
    """No-provider observational selector for the explicitly frozen render-all ablation."""

    def select(self, observable_context, active_skill_catalog, max_active_skills=2):
        catalog = [
            {
                "skill_id": s.skill_id,
                "trigger": s.trigger,
                "activation_signature": s.activation_signature.to_dict(),
            }
            for s in active_skill_catalog
        ]
        return SkillActivationDecision(
            tuple(s.skill_id for s in active_skill_catalog),
            "render-all ablation; all accepted skills injected without applicability filtering",
            None,
        ), {
            "messages": project_observable_messages(observable_context["messages"]),
            "catalog": catalog,
        }


def activating_service_agent_class(base, memory, *, activator, max_active_skills, sink):
    """Subclass native LLMAgent; keep its message generation/tools/protocol unchanged."""
    from .prompts import append_strategy_block

    class EvoTauActivatedService(base):
        @property
        def system_prompt(self):
            return super().system_prompt  # Initial prompt contains no learned guidance.

        def generate_next_message(self, message, state):
            from tau2.data_model.message import SystemMessage

            current = project_observable_messages([message])
            decision, context = activator.select(
                {"messages": [*project_observable_messages(state.messages), *current]},
                memory.skills,
                max_active_skills,
            )
            block = render_selected_service_skills(memory, decision.active_skill_ids)
            original = self.system_prompt
            prompt = append_strategy_block(original, block)
            state.system_messages = [SystemMessage(role="system", content=prompt)]
            sink(
                {
                    "decision": decision.to_dict(),
                    "observable_context_sha256": sha256_json(context),
                    "catalog": context["catalog"],
                    "not_selected_skill_ids": [
                        s.skill_id
                        for s in memory.skills
                        if s.skill_id not in decision.active_skill_ids
                    ],
                    "activation_catalog_tokens": skill_token_count(
                        str(context["catalog"])
                    ),
                    "activated_guidance_tokens": skill_token_count(block),
                    "native_prompt_sha256": sha256_json(original),
                    "prompt_sha256": sha256_json(prompt),
                    "visible_message_count": len(context["messages"]),
                }
            )
            return super().generate_next_message(message, state)

    return EvoTauActivatedService
