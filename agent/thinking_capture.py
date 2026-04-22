"""Capture and store model thinking/reasoning tokens from aider runs."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SummarizerCost:
    """Cost info from a single summarizer LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class SummarizerCostTracker:
    """Accumulates costs from all summarizer LLM calls (spec + test output)."""

    costs: list[SummarizerCost] = field(default_factory=list)

    def add(self, cost: SummarizerCost) -> None:
        self.costs.append(cost)

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.costs)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.costs)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.costs)

    def to_dict(self) -> dict:
        return {
            "summarizer_cost": self.total_cost,
            "summary_input_tokens": self.total_prompt_tokens,
            "summary_output_tokens": self.total_completion_tokens,
            "summarizer_call_count": len(self.costs),
        }


@dataclass
class Turn:
    """A single conversation turn (one user message + one assistant response)."""

    role: str  # "user" or "assistant"
    content: str  # The actual message/response text
    thinking: Optional[str] = None  # Reasoning content (assistant only)
    thinking_tokens: int = 0  # Token count for thinking
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0
    stage: str = ""  # "draft", "lint", or "test"
    module: str = ""  # e.g., "src__itsdangerous___json"
    turn_number: int = 0
    edit_error: str | None = None


@dataclass
class ThinkingCapture:
    """Accumulates turns with thinking across the entire pipeline run."""

    turns: list[Turn] = field(default_factory=list)
    summarizer_costs: SummarizerCostTracker = field(
        default_factory=SummarizerCostTracker
    )

    def add_user_turn(
        self,
        content: str,
        stage: str,
        module: str,
        turn_number: int,
    ) -> None:
        """Record a user message turn."""
        self.turns.append(
            Turn(
                role="user",
                content=content,
                stage=stage,
                module=module,
                turn_number=turn_number,
            )
        )

    def add_assistant_turn(
        self,
        content: str,
        thinking: Optional[str],
        thinking_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int,
        cache_write_tokens: int,
        cost: float,
        stage: str,
        module: str,
        turn_number: int,
    ) -> None:
        """Record an assistant response turn with optional thinking content."""
        self.turns.append(
            Turn(
                role="assistant",
                content=content,
                thinking=thinking,
                thinking_tokens=thinking_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_hit_tokens=cache_hit_tokens,
                cache_write_tokens=cache_write_tokens,
                cost=cost,
                stage=stage,
                module=module,
                turn_number=turn_number,
            )
        )

    def to_history(self) -> list[dict]:
        """Convert to output.jsonl history format."""
        return [
            {
                "role": t.role,
                "content": t.content,
                **({"thinking": t.thinking} if t.thinking else {}),
                **({"thinking_tokens": t.thinking_tokens} if t.thinking_tokens else {}),
                "stage": t.stage,
                "module": t.module,
                "turn": t.turn_number,
            }
            for t in self.turns
        ]

    def get_module_turns(self, module: str) -> list[Turn]:
        """Return turns belonging to a specific module."""
        return [t for t in self.turns if t.module == module]

    def get_module_metrics(self, module: str) -> dict:
        """Aggregate metrics for a single module."""
        module_turns = [
            t for t in self.turns if t.role == "assistant" and t.module == module
        ]
        return {
            "total_cost": sum(t.cost for t in module_turns),
            "total_prompt_tokens": sum(t.prompt_tokens for t in module_turns),
            "total_completion_tokens": sum(t.completion_tokens for t in module_turns),
            "total_thinking_tokens": sum(t.thinking_tokens for t in module_turns),
            "cache_hit_tokens": sum(t.cache_hit_tokens for t in module_turns),
            "cache_write_tokens": sum(t.cache_write_tokens for t in module_turns),
            "num_turns": len(module_turns),
        }

    def get_metrics(self) -> dict:
        """Aggregate metrics across all turns."""
        total_cost = sum(t.cost for t in self.turns if t.role == "assistant")
        total_prompt = sum(t.prompt_tokens for t in self.turns if t.role == "assistant")
        total_completion = sum(
            t.completion_tokens for t in self.turns if t.role == "assistant"
        )
        total_thinking = sum(
            t.thinking_tokens for t in self.turns if t.role == "assistant"
        )

        per_stage: dict = {}
        for t in self.turns:
            if t.role != "assistant":
                continue
            if t.stage not in per_stage:
                per_stage[t.stage] = {
                    "cost": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thinking_tokens": 0,
                }
            per_stage[t.stage]["cost"] += t.cost
            per_stage[t.stage]["prompt_tokens"] += t.prompt_tokens
            per_stage[t.stage]["completion_tokens"] += t.completion_tokens
            per_stage[t.stage]["thinking_tokens"] += t.thinking_tokens

        return {
            "total_cost": total_cost + self.summarizer_costs.total_cost,
            "total_prompt_tokens": total_prompt
            + self.summarizer_costs.total_prompt_tokens,
            "total_completion_tokens": total_completion
            + self.summarizer_costs.total_completion_tokens,
            "total_thinking_tokens": total_thinking,
            "per_stage": per_stage,
            **self.summarizer_costs.to_dict(),
        }
