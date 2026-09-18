from math import sqrt

import duckdb
from scipy.stats import norm

from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.exceptions import UnknownColumn, UnknownValue
from churn_mcp.models import Segment, SemanticLayer

FilterValue = str | int | float | list[str]


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    spread = z * sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((center - spread) / denom, (center + spread) / denom)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """p-value for H0: the two proportions are equal, two-sided."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def bh_qvalues(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values: the smallest FDR at which each test is significant."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    qvalues = [0.0] * n
    running_min = 1.0
    for rank, index in zip(reversed(range(1, n + 1)), reversed(order), strict=True):
        candidate = pvalues[index] * n / rank
        running_min = min(running_min, candidate)
        qvalues[index] = running_min
    return qvalues


MAX_GROUP_BY = 3


def _build_filter_sql(
    filters: dict[str, FilterValue], layer: SemanticLayer
) -> tuple[str, list[str | int | float]]:
    clauses = []
    params: list[str | int | float] = []
    for column, value in filters.items():
        if column not in layer.names():
            raise UnknownColumn(column, layer.names())
        domain = layer.domain(column)
        values = value if isinstance(value, list) else [value]
        if domain:
            for v in values:
                if str(v) not in domain:
                    raise UnknownValue(column, str(v), domain)
        placeholders = ", ".join("?" for _ in values)
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
    return (" AND ".join(clauses) if clauses else "TRUE"), params


def segment_churn(
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    group_by: list[str],
    filters: dict[str, FilterValue] | None = None,
    top_n: int = 20,
) -> list[Segment]:
    if not 1 <= len(group_by) <= MAX_GROUP_BY:
        raise ValueError(f"group_by needs 1 to {MAX_GROUP_BY} columns, got {len(group_by)}")
    for column in group_by:
        if column not in layer.names():
            raise UnknownColumn(column, layer.names())

    where, params = _build_filter_sql(filters or {}, layer)
    dims = ", ".join(group_by)
    rows = con.execute(
        f"SELECT {dims}, COUNT(*) AS n, SUM(churn) AS churned, "
        f"SUM(monthly_charge) FILTER (WHERE churn = 1) AS mrr_lost "
        f"FROM customers WHERE {where} GROUP BY {dims}",
        params,
    ).fetchall()

    # A scalar aggregate always returns exactly one row, so fetchone() is never None.
    totals = con.execute(
        f"SELECT COUNT(*), SUM(churn) FROM customers WHERE {where}", params
    ).fetchone()
    assert totals is not None
    total_n, total_churned = totals
    baseline_rate = total_churned / total_n if total_n else 0.0

    pvalues = [two_prop_z(row[-2], row[-3], total_churned, total_n) for row in rows]
    qvalues = bh_qvalues(pvalues)

    segments = []
    for row, qvalue in zip(rows, qvalues, strict=True):
        *group, n, churned, mrr_lost = row
        rate = churned / n if n else 0.0
        low, high = wilson_ci(churned, n, config.analytics.confidence_level)
        segments.append(
            Segment(
                group=tuple(group),
                n=n,
                churned=churned,
                churn_rate=rate,
                ci_low=low,
                ci_high=high,
                lift=rate / baseline_rate if baseline_rate else 0.0,
                q_value=qvalue,
                mrr_lost=mrr_lost or 0.0,
                suppressed=n < config.analytics.min_segment_size,
            )
        )
    segments.sort(key=lambda s: s.churn_rate, reverse=True)
    return segments[:top_n]
