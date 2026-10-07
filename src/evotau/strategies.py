"""Open natural-language strategy carriers used by the alternating loop."""

from __future__ import annotations

from dataclasses import dataclass

from .service_skills import (
    ServiceSkillMemory,
    ServiceSkillMemoryV2,
    render_service_skill_memory,
)


@dataclass(frozen=True, slots=True)
class PromptStrategy:
    """A strategy is a free-form prompt overlay; no behavior schema is imposed."""

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("strategy text must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text}


ServiceCarrier = PromptStrategy | ServiceSkillMemory | ServiceSkillMemoryV2


def render_customer_strategy(strategy: PromptStrategy | None) -> str:
    return _render("evotau_customer_strategy", strategy)


def render_service_strategy(strategy: ServiceCarrier | None) -> str:
    if isinstance(strategy, ServiceSkillMemoryV2):
        return ""  # V2 guidance is injected only by per-turn activation.
    if isinstance(strategy, ServiceSkillMemory):
        return render_service_skill_memory(strategy)
    return _render("evotau_service_strategy", strategy)


def _render(tag: str, strategy: PromptStrategy | None) -> str:
    if strategy is None or not strategy.text.strip():
        return ""
    return f"<{tag}>\n{strategy.text.strip()}\n</{tag}>"
