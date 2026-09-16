from abc import ABC, abstractmethod
from typing import Any

from churn_mcp.config import Effort, ModelTier
from churn_mcp.models import Usage


class LLM(ABC):
    """Swap providers by adding a subclass; nothing else depends on OpenAI directly."""

    @abstractmethod
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
    ) -> tuple[dict[str, Any], Usage]: ...
