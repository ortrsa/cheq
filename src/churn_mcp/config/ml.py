from pydantic import Field

from churn_mcp.config.base import Section


class ModelConfig(Section):
    cv_folds: int = Field(default=5, ge=2)
    seed: int = 42
    exclude_protected: bool = True
