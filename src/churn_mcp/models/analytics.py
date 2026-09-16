from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    group: tuple[Any, ...]
    n: int
    churned: int
    churn_rate: float
    ci_low: float
    ci_high: float
    lift: float
    q_value: float
    mrr_lost: float
    suppressed: bool
