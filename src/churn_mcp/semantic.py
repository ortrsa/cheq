from functools import lru_cache

import yaml

from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.models import Column, SemanticLayer


def load(config: TelcoChurnMcpConfig) -> SemanticLayer:
    raw = yaml.safe_load(config.data.semantic_path.read_text())
    columns = {
        name: Column(
            name=name,
            role=spec["role"],
            desc=spec["desc"],
            values=tuple(spec.get("values", ())),
        )
        for name, spec in raw["columns"].items()
    }
    return SemanticLayer(
        table=raw["table"],
        grain=raw["grain"],
        columns=columns,
        metrics=raw["metrics"],
        traps=tuple(raw["traps"]),
        defaults=raw["defaults"],
        examples=tuple(raw["examples"]),
    )


@lru_cache(maxsize=1)
def get_semantic_layer(config: TelcoChurnMcpConfig) -> SemanticLayer:
    return load(config)
