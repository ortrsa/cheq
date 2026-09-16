import json
from typing import Any

from openai import OpenAI

from churn_mcp.config import Effort, ModelTier
from churn_mcp.exceptions import LLMRefused
from churn_mcp.llm.base import LLM
from churn_mcp.models import Usage


class OpenAILLM(LLM):
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

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
        response = self._client.responses.create(
            model=model.value,
            input=[
                {"role": "developer", "content": developer},
                {"role": "user", "content": user},
            ],
            reasoning={"effort": effort.value},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            max_output_tokens=max_output_tokens,
        )

        reported = response.usage
        details = getattr(reported, "input_tokens_details", None)
        usage = Usage(
            input_tokens=reported.input_tokens if reported else 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            output_tokens=reported.output_tokens if reported else 0,
        )

        for item in response.output:
            for content in getattr(item, "content", []):
                if content.type == "refusal":
                    raise LLMRefused(content.refusal)

        if response.status != "completed" or not response.output_text:
            raise LLMRefused(f"response was {response.status} with no text")

        return json.loads(response.output_text), usage
