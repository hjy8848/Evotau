"""Plain-language labels for EvoTau data already present in immutable artifacts."""

from __future__ import annotations

from typing import Any

_DISCLOSURE = {
    "minimal_on_request": "按客服询问逐步提供必要信息",
    "related_on_request": "回答具体问题，并补充相关信息",
}
_REQUEST_ORDER = {
    "scenario_order": "按任务给定顺序提出独立请求",
    "reverse_independent": "反向提出独立请求（保留必要依赖顺序）",
}
_CHALLENGE = {
    "none": "不额外追问",
    "ask_reason": "遇到拒绝或验证要求时询问原因",
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
    return {
        "available": True,
        "native": False,
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
    return {
        **budget,
        "attempt_label": f"{attempts} / {cap}" if type(attempts) is int and type(cap) is int else "Unavailable",
        "token_label": (
            f"{prompt} / {completion}"
            if type(prompt) is int and type(completion) is int else "Unavailable"
        ),
        "total_tokens": prompt + completion if type(prompt) is int and type(completion) is int else None,
        "cost_label": budget.get("cost", "Cost unavailable until a frozen price schedule is supplied."),
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
    }
