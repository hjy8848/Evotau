"""Plain-language labels for EvoTau data already present in immutable artifacts."""

from __future__ import annotations

from typing import Any

_DISCLOSURE = {
    "minimal_on_request": "按客服询问逐步提供必要信息",
    "progressive": "随对话进展逐步披露已知信息",
    "related_on_request": "回答具体问题，并补充相关信息",
}
_REQUEST_ORDER = {
    "scenario_order": "按任务给定顺序提出独立请求",
    "reverse_independent": "反向提出独立请求（保留必要依赖顺序）",
    "dependency_first": "先提出依赖事项，再提出后续请求",
    "high_risk_first": "独立请求中优先提出需核验或承诺的事项",
}
_DECOMPOSITION = {
    "bundled": "关联请求一并提出",
    "one_by_one": "独立请求逐个提出",
    "dependency_grouped": "按依赖关系分组提出",
}
_PREFERENCE_REVISION = {
    "fixed": "保持原有偏好",
    "revise_before_commit": "承诺前仅调整任务允许变动的偏好",
    "narrow_after_options": "了解选项后仅收窄任务允许变动的偏好",
}
_CORRECTION = {
    "accept_if_correct": "理解准确时接受；必要时如实纠正",
    "correct_once": "对实质误解如实纠正一次",
    "correct_and_restate_constraint": "如实纠正并重申原有约束",
}
_CHALLENGE = {
    "none": "不额外追问",
    "ask_reason": "遇到拒绝或验证要求时询问原因",
    "ask_policy_boundary": "询问适用的政策边界",
    "rephrase_request": "遇到阻碍时重述一次相同请求",
}


def customer_strategy_view(strategy: Any) -> dict[str, Any]:
    if strategy is None:
        return {
            "available": True, "native": True,
            "summary": "τ-bench 原生 Customer；本场没有 EvoTau 策略覆盖。",
            "technical": None,
        }
    if not isinstance(strategy, dict):
        return {"available": False, "native": False, "summary": "Customer 策略定义不可用。", "technical": strategy}
    challenge_budget = strategy.get("challenge_budget")
    if "request_decomposition" in strategy:
        fields = {
            "disclosure": _DISCLOSURE.get(str(strategy.get("disclosure")), "未知设置"),
            "request_order": _REQUEST_ORDER.get(str(strategy.get("request_order")), "未知设置"),
            "request_decomposition": _DECOMPOSITION.get(str(strategy.get("request_decomposition")), "未知设置"),
            "preference_revision": _PREFERENCE_REVISION.get(str(strategy.get("preference_revision")), "未知设置"),
            "correction_behavior": _CORRECTION.get(str(strategy.get("correction_behavior")), "未知设置"),
            "challenge_behavior": _CHALLENGE.get(str(strategy.get("challenge_behavior")), "未知设置"),
        }
        return {
            "available": True, "native": False, "version": 2,
            **fields,
            "challenge_budget": challenge_budget if type(challenge_budget) is int else None,
            "summary": "；".join(fields.values()),
            "technical": strategy,
        }
    return {
        "available": True,
        "native": False,
        "version": 1,
        "disclosure": _DISCLOSURE.get(str(strategy.get("disclosure")), "未知设置"),
        "request_order": _REQUEST_ORDER.get(str(strategy.get("request_order")), "未知设置"),
        "challenge_style": _CHALLENGE.get(str(strategy.get("challenge_style")), "未知设置"),
        "challenge_budget": challenge_budget if type(challenge_budget) is int else None,
        "summary": "；".join((
            _DISCLOSURE.get(str(strategy.get("disclosure")), "未知设置"),
            _REQUEST_ORDER.get(str(strategy.get("request_order")), "未知设置"),
            _CHALLENGE.get(str(strategy.get("challenge_style")), "未知设置"),
        )),
        "technical": strategy,
    }


def service_strategy_view(strategy: Any) -> dict[str, Any]:
    if not isinstance(strategy, dict) or not isinstance(strategy.get("rules", []), list):
        return {"available": False, "rules": [], "technical": strategy}
    return {"available": True, "rules": strategy.get("rules", []), "technical": strategy}


def gate_status(gate: Any) -> dict[str, str]:
    if not isinstance(gate, dict):
        return {"label": "未运行", "tone": "muted", "detail": "本代没有可查看的 repair gate。"}
    if gate.get("inconclusive") is True:
        return {"label": "无法判断", "tone": "warning", "detail": "现有 gate 证据不足，不能判定接受或拒绝。"}
    if gate.get("accepted") is True:
        return {"label": "已接受", "tone": "success", "detail": "候选 Service 通过已记录的门控。"}
    return {"label": "已拒绝", "tone": "danger", "detail": "候选 Service 未通过门控。"}


def failure_status(failure: Any) -> dict[str, str]:
    if isinstance(failure, dict) and failure.get("verified") is True:
        return {"label": "已验证的 Service 失败", "tone": "danger"}
    if isinstance(failure, dict) and failure.get("provisional") is True:
        return {"label": "尚未验证，不计入 fitness", "tone": "warning"}
    return {"label": "任务结果 / 失败候选", "tone": "muted"}


def budget_view(budget: dict[str, Any]) -> dict[str, Any]:
    attempts, cap = budget.get("attempts"), budget.get("cap")
    prompt, completion = budget.get("prompt_tokens"), budget.get("completion_tokens")
    progress = (attempts / cap) if type(attempts) is int and type(cap) is int and cap > 0 else None
    return {
        **budget,
        "attempt_label": f"{attempts} / {cap}" if type(attempts) is int and type(cap) is int else "Unavailable",
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
    decision = commit.get("decision_record") or {}
    customer = decision.get("customer", {})
    evaluations = customer.get("evaluations", ())
    service = decision.get("service", {})
    gate = service.get("gate")
    return {
        **commit,
        "customer_evaluations": evaluations,
        "selection": customer.get("selection"),
        "proposals": customer.get("proposals", ()),
        "service_decision": service,
        "gate_view": gate_status(gate),
        "gate": gate,
        "selection_label": (
            "Customer 策略已更新" if commit.get("customer_evolved")
            else "Customer 保持不变：没有严格胜出的确认挑战者"
        ),
        "narrative": generation_narrative(commit),
        "customer_diff": strategy_diff_rows(customer.get("incumbent_before"), customer.get("incumbent_after"), "customer"),
        "service_diff": strategy_diff_rows(service.get("incumbent_before"), service.get("incumbent_after"), "service"),
    }


def generation_narrative(commit: dict[str, Any]) -> tuple[str, ...]:
    """Explain only decisions recorded in the committed generation artifact."""
    decision = commit.get("decision_record") or {}
    customer = decision.get("customer") or {}
    service = decision.get("service") or {}
    lines = []
    if commit.get("customer_evolved") is True:
        lines.append("Customer 策略已更新：提交记录显示本代选择了新策略。")
    elif commit.get("customer_evolved") is False:
        selection = customer.get("selection") or {}
        reason = selection.get("reason")
        lines.append("Customer 保持不变：本代没有提交策略更新。")
        if isinstance(reason, str) and reason:
            lines.append(f"记录的选择原因：{reason}")
    else:
        lines.append("Customer 演变状态在提交记录中不可用。")
    if commit.get("service_evolved") is True:
        lines.append("Service 策略已更新：提交记录显示 repair gate 接受了新策略。")
    elif commit.get("service_evolved") is False:
        gate = service.get("gate")
        if isinstance(gate, dict):
            status = gate_status(gate)["label"]
            lines.append(f"Service 未更新：repair gate 状态为“{status}”。")
        else:
            lines.append("Service 未更新：本代没有已记录的 repair gate 结果。")
    else:
        lines.append("Service 演变状态在提交记录中不可用。")
    return tuple(lines)


_CUSTOMER_FIELDS = (
    ("disclosure", "信息披露"), ("request_order", "请求顺序"),
    ("challenge_style", "挑战方式"), ("challenge_budget", "挑战次数"),
)
_CUSTOMER_FIELDS_V2 = (
    ("disclosure", "信息披露"), ("request_order", "请求顺序"),
    ("request_decomposition", "请求拆分"), ("preference_revision", "偏好调整"),
    ("correction_behavior", "纠正方式"), ("challenge_behavior", "追问方式"),
    ("challenge_budget", "追问次数"),
)


def strategy_diff_rows(before: Any, after: Any, side: str) -> dict[str, Any]:
    """Build a display-only diff from two snapshots in one committed decision."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"available": False, "rows": (), "added": (), "removed": ()}
    if side == "customer":
        before_view = customer_strategy_view(before)
        after_view = customer_strategy_view(after)
        fields = (
            _CUSTOMER_FIELDS_V2
            if before_view.get("version") == 2 or after_view.get("version") == 2
            else _CUSTOMER_FIELDS
        )
        rows = tuple({
            "field": label,
            "before": before_view.get(key, "Unavailable"),
            "after": after_view.get(key, "Unavailable"),
            "changed": before_view.get(key) != after_view.get(key),
        } for key, label in fields)
        return {"available": True, "rows": rows, "added": (), "removed": ()}

    before_rules = before.get("rules", [])
    after_rules = after.get("rules", [])
    if not isinstance(before_rules, list) or not isinstance(after_rules, list):
        return {"available": False, "rows": (), "added": (), "removed": ()}
    before_by_ref = {str(rule.get("policy_ref")): rule for rule in before_rules if isinstance(rule, dict)}
    after_by_ref = {str(rule.get("policy_ref")): rule for rule in after_rules if isinstance(rule, dict)}
    added = tuple(after_by_ref[key] for key in sorted(after_by_ref.keys() - before_by_ref.keys()))
    removed = tuple(before_by_ref[key] for key in sorted(before_by_ref.keys() - after_by_ref.keys()))
    common = sorted(before_by_ref.keys() & after_by_ref.keys())
    rows = tuple({"field": key, "before": before_by_ref[key], "after": after_by_ref[key],
                  "changed": before_by_ref[key] != after_by_ref[key]}
                 for key in common)
    return {"available": True, "rows": rows, "added": added, "removed": removed}
