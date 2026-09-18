from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from churn_mcp.models.llm import Usage


@dataclass
class Result:
    question: str
    answered_by: str = "t2sql"
    interpretation: str = ""
    sql: str = ""
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    answer: str = ""
    assumptions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    confidence: str = ""
    attempts: int = 0
    grounding: str = "skipped"
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


class Intent(StrEnum):
    DATA_QUERY = "data_query"
    MODEL_INFO = "model_info"
    UNSAFE = "unsafe"
    OUT_OF_SCOPE = "out_of_scope"


class LLMOutput(BaseModel):
    """Structured output the LLM must return; strict mode needs every field required
    and no extras, which is what a defaults-free model with extra='forbid' emits."""

    model_config = ConfigDict(extra="forbid")


class Route(LLMOutput):
    intent: Intent
    normalized_question: str


class Draft(LLMOutput):
    sql: str
    interpretation: str
    assumptions: list[str]
    confidence: Literal["high", "medium", "low"]


class Narrative(LLMOutput):
    answer: str
    caveats: list[str]
