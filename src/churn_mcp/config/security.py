from pydantic import Field

from churn_mcp.config.base import Section


class SecurityConfig(Section):
    allowed_tables: frozenset[str] = frozenset({"customers"})
    max_rows: int = Field(default=200, ge=1, le=10_000)
