from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.llm.base import LLM
from churn_mcp.llm.mock import MockLLM
from churn_mcp.llm.openai_client import OpenAILLM
from churn_mcp.models import Usage

__all__ = ["LLM", "MockLLM", "OpenAILLM", "Usage", "build"]


def build(config: TelcoChurnMcpConfig) -> LLM | None:
    key = config.api_key()
    return OpenAILLM(key) if key else None
