from typing import Any

from churn_mcp.config import Effort, ModelTier
from churn_mcp.llm.base import LLM
from churn_mcp.models import Usage


class MockLLM(LLM):
    """Replays a scripted list of payloads, so pipeline tests never touch the network."""

    def __init__(self, replies: list[dict[str, Any] | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        *,
        model: ModelTier,
        effort: Effort,
        developer: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], Usage]:
        self.calls.append((schema_name, user))
        if not self.replies:
            raise AssertionError(f"MockLLM ran out of replies at '{schema_name}'")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, Usage(input_tokens=100, cached_tokens=0, output_tokens=20)
