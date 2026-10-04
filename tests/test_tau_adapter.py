from __future__ import annotations

from evotau.prompts import append_strategy_block
from evotau.strategies import (
    PromptStrategy,
    render_customer_strategy,
    render_service_strategy,
)
from evotau.tau_adapter import customer_user_class, service_agent_class


class NativeUser:
    def __init__(self, prompt: str = "τ-bench user scenario and guidelines") -> None:
        self.prompt = prompt

    @property
    def system_prompt(self) -> str:
        return self.prompt


class NativeAgent:
    def __init__(self, prompt: str = "τ-bench agent policy") -> None:
        self.prompt = prompt

    @property
    def system_prompt(self) -> str:
        return self.prompt


def test_empty_open_strategies_leave_native_prompts_unchanged() -> None:
    user = NativeUser()
    agent = NativeAgent()
    adapted_user = customer_user_class(NativeUser, PromptStrategy(""))(prompt=user.system_prompt)
    adapted_agent = service_agent_class(NativeAgent, PromptStrategy(" "))(prompt=agent.system_prompt)

    assert adapted_user.system_prompt == user.system_prompt
    assert adapted_agent.system_prompt == agent.system_prompt
    assert append_strategy_block(user.system_prompt, "") == user.system_prompt


def test_natural_language_strategies_are_appended_as_separate_prompt_sections() -> None:
    customer = PromptStrategy("Pursue the scenario goal and respond naturally to what the agent does.")
    service = PromptStrategy("Check the requested scope before using a write tool.")
    user = customer_user_class(NativeUser, customer)(prompt=NativeUser().system_prompt)
    agent = service_agent_class(NativeAgent, service)(prompt=NativeAgent().system_prompt)

    assert user.system_prompt.startswith("τ-bench user scenario and guidelines")
    assert render_customer_strategy(customer) in user.system_prompt
    assert agent.system_prompt.startswith("τ-bench agent policy")
    assert render_service_strategy(service) in agent.system_prompt
