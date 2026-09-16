from pydantic import Field

from churn_mcp.config.base import Section
from churn_mcp.config.enums import Effort, ModelTier


class T2SQLConfig(Section):
    router_model: ModelTier = ModelTier.LUNA
    router_effort: Effort = Effort.NONE
    generator_model: ModelTier = ModelTier.TERRA
    generator_effort: Effort = Effort.MEDIUM
    synthesizer_model: ModelTier = ModelTier.LUNA
    synthesizer_effort: Effort = Effort.LOW
    include_value_domains: bool = True
    include_traps: bool = True
    include_examples: bool = True
    max_repairs: int = Field(default=2, ge=0, le=5)
    max_result_rows_to_llm: int = 50
