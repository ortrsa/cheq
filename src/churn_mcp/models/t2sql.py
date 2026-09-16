from dataclasses import dataclass, field
from typing import Any

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
