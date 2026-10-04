"""Plain-language labels for EvoTau data already present in immutable artifacts."""

from __future__ import annotations

from typing import Any


def customer_strategy_view(strategy: Any) -> dict[str, Any]:
    if strategy is None:
        return {
            "available": True, "native": True,
            "summary": "τ-bench 原生 Customer；本场没有 EvoTau 策略覆盖。",
            "technical": None,
        }
    if not isinstance(strategy, dict):
        return {"available": False, "native": False, "summary": "Customer 策略定义不可用。", "technical": strategy}
    text = strategy.get("text")
    if isinstance(text, str):
        summary = text.strip() or "空策略覆盖；Customer 使用 τ-bench 原生提示。"
        return {
            "available": True,
            "native": not bool(text.strip()),
            "version": "open-text",
            "summary": summary,
            "text": text,
            "technical": strategy,
        }
    return {
        "available": True,
        "native": False,
        "version": "saved-legacy",
        "summary": "Historical serialized strategy; it is not used by the current evolution method.",
        "technical": strategy,
    }


def service_strategy_view(strategy: Any) -> dict[str, Any]:
    if isinstance(strategy, dict) and isinstance(strategy.get("text"), str):
        text = strategy["text"].strip()
        return {
            "available": True,
            "summary": text or "空策略覆盖；Service 完全使用 τ-bench 原生提示与 policy。",
            "text": text,
            "technical": strategy,
        }
    return {
        "available": isinstance(strategy, dict),
        "summary": "Historical serialized strategy; it is not used by the current evolution method.",
        "technical": strategy,
    }


def budget_view(budget: dict[str, Any]) -> dict[str, Any]:
    attempts, cap = budget.get("attempts"), budget.get("cap")
    prompt, completion = budget.get("prompt_tokens"), budget.get("completion_tokens")
    progress = (attempts / cap) if type(attempts) is int and type(cap) is int and cap > 0 else None
    return {
        **budget,
        "attempt_label": (
            f"{attempts} / {cap}" if type(attempts) is int and type(cap) is int
            else f"{attempts} requests · unbounded" if type(attempts) is int and cap is None
            else "Unavailable"
        ),
        "token_label": (
            f"{prompt} / {completion}"
            if type(prompt) is int and type(completion) is int else "Unavailable"
        ),
        "total_tokens": prompt + completion if type(prompt) is int and type(completion) is int else None,
        "progress": min(1.0, max(0.0, progress)) if progress is not None else None,
        "remaining": max(0, cap - attempts) if type(attempts) is int and type(cap) is int else None,
        "progress_label": f"{progress * 100:.1f}%" if progress is not None else "Unavailable",
        "cost_label": budget.get("cost", "Cost unavailable until a frozen price schedule is supplied."),
    }


def db_state_trace_view(trace: Any) -> dict[str, Any]:
    """Create restrained Simple and verbatim Research views of a trace sidecar."""

    if not isinstance(trace, dict) or trace.get("status") != "complete":
        return {
            "available": False,
            "reason_code": (trace or {}).get("reason_code", "trace_not_generated")
            if isinstance(trace, dict) else "trace_not_generated",
            "events": [],
            "summary": {},
        }
    entity_labels = {"order": "Order", "order_item": "Order item", "product": "Product", "user": "User"}
    field_labels = {
        "status": "Status",
        "return_items": "Return items",
        "return_payment_method_id": "Refund method",
        "cancel_reason": "Cancellation reason",
        "exchange_items": "Exchange items",
        "exchange_new_items": "Replacement items",
        "exchange_payment_method_id": "Exchange payment method",
        "exchange_price_difference": "Exchange price difference",
    }
    events = []
    for event in trace.get("events", ()):
        if not isinstance(event, dict):
            continue
        changes = []
        for change in event.get("changes", ()):
            if not isinstance(change, dict):
                continue
            entity_type = str(change.get("entity_type", ""))
            field_path = str(change.get("field_path", ""))
            leaf = field_path.rsplit(".", 1)[-1]
            changes.append({
                **change,
                "simple_entity_label": entity_labels.get(entity_type),
                "simple_field_label": field_labels.get(
                    leaf, " ".join(word.capitalize() for word in leaf.replace("_", " ").split()),
                ),
            })
        events.append({**event, "changes": changes})
    summary = trace.get("summary")
    summary = dict(summary) if isinstance(summary, dict) else {}
    comparison = summary.get("final_comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("differences"), list):
        comparison = dict(comparison)
        comparison["differences"] = [
            {
                **change,
                "simple_entity_label": entity_labels.get(str(change.get("entity_type", ""))),
                "simple_field_label": field_labels.get(
                    str(change.get("field_path", "")).rsplit(".", 1)[-1],
                    " ".join(
                        word.capitalize()
                        for word in str(change.get("field_path", "")).replace("_", " ").split()
                    ),
                ),
            }
            for change in comparison["differences"]
            if isinstance(change, dict)
        ]
        summary["final_comparison"] = comparison
    return {
        "available": True,
        "events": events,
        "summary": summary,
        "trace_sha256": trace.get("trace_sha256"),
        "provenance": trace.get("provenance", {}),
        "artifact_ref": "db-state-trace.json",
    }


def current_strategy_for_episode(run: dict[str, Any], episode: dict[str, Any]) -> tuple[Any, Any]:
    customer = episode.get("customer_strategy")
    service = episode.get("service_strategy")
    strategies = run.get("strategies", {})
    if customer is None:
        customer = strategies.get("customer", {}).get(episode.get("customer_strategy_id"))
    if service is None:
        service = strategies.get("service", {}).get(episode.get("service_strategy_id"))
    return customer, service


def generation_view(commit: dict[str, Any]) -> dict[str, Any]:
    if isinstance(commit.get("customer_phase"), dict):
        customer_before = commit.get("customer_before", {})
        customer_after = commit.get("customer_after", {})
        service_before = commit.get("service_before", {})
        service_after = commit.get("service_after", {})
        customer_changed = customer_before.get("strategy_id") != customer_after.get("strategy_id")
        service_changed = service_before.get("strategy_id") != service_after.get("strategy_id")
        customer_selection = commit["customer_phase"].get("selection", {})
        service_phase = commit.get("service_phase", {})
        service_decision = service_phase.get("selection", {})
        accepted = service_phase.get("accepted") is True
        proposed_service = service_phase.get("proposed_strategy") or {}
        return {
            **commit,
            "customer_evolved": customer_changed,
            "service_evolved": service_changed,
            "customer_id": customer_after.get("strategy_id"),
            "service_id": service_after.get("strategy_id"),
            "selection_label": (
                "Customer 策略已更新" if customer_changed else "Customer 保持 incumbent"
            ),
            "selection": {"reason": customer_selection.get("reason", "")},
            "service_accepted": accepted,
            "service_reason": service_decision.get("reason", ""),
            "proposed_service_text": proposed_service.get(
                "text", proposed_service.get("strategy", ""),
            ),
            "decision_record": commit,
            "narrative": (
                f"Customer: {customer_before.get('strategy', '')}",
                f"→ {customer_after.get('strategy', '')}",
                f"Service: {service_before.get('strategy', '')}",
                f"→ {service_after.get('strategy', '')}",
                f"Service acceptance: {service_decision.get('reason', 'not accepted')}",
            ),
            "customer_diff": strategy_diff_rows(customer_before, customer_after, "customer"),
            "service_diff": strategy_diff_rows(service_before, service_after, "service"),
        }
    return {
        **commit,
        "selection_label": "Historical generation artifact",
        "narrative": (),
        "customer_diff": {"available": False, "rows": (), "added": (), "removed": ()},
        "service_diff": {"available": False, "rows": (), "added": (), "removed": ()},
    }


def strategy_diff_rows(before: Any, after: Any, side: str) -> dict[str, Any]:
    """Build a display-only diff from two snapshots in one committed decision."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"available": False, "rows": (), "added": (), "removed": ()}
    before_text = before.get("text", before.get("strategy"))
    after_text = after.get("text", after.get("strategy"))
    if isinstance(before_text, str) and isinstance(after_text, str):
        return {
            "available": True,
            "rows": ({
                "field": "strategy",
                "before": before_text,
                "after": after_text,
                "changed": before_text != after_text,
            },),
            "added": (),
            "removed": (),
        }
    keys = sorted(set(before) | set(after))
    rows = tuple({
        "field": key,
        "before": before.get(key),
        "after": after.get(key),
        "changed": before.get(key) != after.get(key),
    } for key in keys)
    return {"available": True, "rows": rows, "added": (), "removed": ()}
