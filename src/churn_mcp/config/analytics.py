from pydantic import Field

from churn_mcp.config.base import Section


class AnalyticsConfig(Section):
    min_segment_size: int = Field(default=30, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    fdr_alpha: float = Field(default=0.05, gt=0, lt=1)
