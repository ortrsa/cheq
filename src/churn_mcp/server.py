import functools
from dataclasses import asdict
from typing import Any

import duckdb
from mcp.server import MCPServer

from churn_mcp import analytics, model, sql_guard, t2sql
from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.exceptions import ModelCardMissing, UnknownColumn, UnknownValue
from churn_mcp.llm import LLM
from churn_mcp.models import RiskRanking, Segment, SemanticLayer, Trained

DOMAIN_ERRORS = (UnknownColumn, UnknownValue)
# Bad arguments (including the domain errors) and queries that parse but fail at runtime.
TOOL_ERRORS = (ValueError, duckdb.Error)


def envelope(result: Any, caveats: tuple[str, ...] = (), **meta: Any) -> dict[str, Any]:
    return {"result": result, "caveats": list(caveats), "meta": meta}


def error_envelope(message: str) -> dict[str, Any]:
    return envelope(None, error=message)


def _table(columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> dict[str, Any]:
    return {"columns": list(columns), "rows": [list(row) for row in rows], "row_count": len(rows)}


def _segment_dict(segment: Segment) -> dict[str, Any]:
    return {
        "group": list(segment.group),
        "n": segment.n,
        "churn_rate": segment.churn_rate,
        "ci": [segment.ci_low, segment.ci_high],
        "lift": segment.lift,
        "q_value": segment.q_value,
        "mrr_lost": segment.mrr_lost,
        "suppressed": segment.suppressed,
    }


def _risk_dict(ranking: RiskRanking) -> dict[str, Any]:
    return {
        "customer_id": ranking.customer_id,
        "probability": ranking.probability,
        "arr": ranking.arr,
        "expected_loss": ranking.expected_loss,
        "reasons": [{"feature": name, "contribution": value} for name, value in ranking.reasons],
    }


def build_server(
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    llm: LLM | None,
) -> MCPServer:
    mcp = MCPServer("telco-churn")

    # Training runs 5-fold cross-validation; do it on first use, not on every call.
    @functools.cache
    def trained_model() -> Trained:
        with con.cursor() as cur:
            return model.train(cur, layer, config)

    @mcp.tool(
        description="Answer a natural-language question about telco customer churn data "
        "with SQL, a result table and a grounded narrative."
    )
    def ask_data(question: str, synthesize: bool = True) -> dict[str, Any]:
        with con.cursor() as cur:
            r = t2sql.ask(question, cur, layer, config, llm, synthesise=synthesize)
        return envelope(
            {
                "question": r.question,
                "answered_by": r.answered_by,
                "interpretation": r.interpretation,
                "sql": r.sql,
                "table": _table(r.columns, r.rows),
                "answer": r.answer,
                "assumptions": list(r.assumptions),
                "confidence": r.confidence,
                "attempts": r.attempts,
                "grounding": r.grounding,
                "error": r.error,
            },
            caveats=r.caveats,
            input_tokens=r.usage.input_tokens,
            cached_tokens=r.usage.cached_tokens,
            output_tokens=r.usage.output_tokens,
        )

    @mcp.tool(
        description="Run your own read-only DuckDB SELECT against the customers table, "
        "through the same validator and sandbox as ask_data. Use when you already know the "
        "exact SQL; otherwise use ask_data. Call describe_dataset first for exact column "
        "names and values."
    )
    def run_sql(sql: str, max_rows: int | None = None) -> dict[str, Any]:
        try:
            scoped = sql_guard.with_max_rows(config, max_rows)
            verdict = sql_guard.validate(sql, layer, scoped)
            if not verdict.ok:
                return error_envelope(verdict.error or "invalid query")
            with con.cursor() as cur:
                columns, rows = t2sql.execute(verdict.sql, cur)
        except TOOL_ERRORS as error:
            return error_envelope(str(error))
        return envelope(
            {"sql": verdict.sql, "table": _table(columns, rows)},
            caveats=verdict.warnings,
            rewrites=list(verdict.rewrites),
        )

    @mcp.tool(
        description="Look up columns: role, description and exact allowed values, plus "
        "metric definitions and known data traps. Pass column names to limit the output. "
        "Use before writing SQL or filters."
    )
    def describe_dataset(columns: list[str] | None = None) -> dict[str, Any]:
        names = columns or sorted(layer.names())
        try:
            for name in names:
                if name not in layer.names():
                    raise UnknownColumn(name, layer.names())
        except DOMAIN_ERRORS as error:
            return error_envelope(str(error))

        described = {
            name: {
                "role": layer.columns[name].role,
                "desc": layer.columns[name].desc,
                "values": list(layer.columns[name].values),
            }
            for name in names
        }
        return envelope(
            {
                "table": layer.table,
                "grain": layer.grain,
                "columns": described,
                "metrics": layer.metrics,
                "traps": list(layer.traps),
                "defaults": layer.defaults,
            }
        )

    @mcp.tool(
        description="Churn rate by 1-3 dimensions (e.g. contract, internet_type) with 95% "
        "confidence intervals, lift over the baseline, Benjamini-Hochberg q-values and MRR "
        "lost. Use for 'which segments are significantly worse' and any comparison where "
        "significance matters; ask_data does not test significance. Filters take exact "
        'values, e.g. {"internet_type": "Fiber Optic"}. Small segments are flagged.'
    )
    def segment_churn(
        group_by: list[str], filters: dict[str, Any] | None = None, top_n: int = 20
    ) -> dict[str, Any]:
        try:
            with con.cursor() as cur:
                segments = analytics.segment_churn(cur, layer, config, group_by, filters, top_n)
        except TOOL_ERRORS as error:
            return error_envelope(str(error))

        caveats = sql_guard.leak_warning(set(group_by) | set(filters or {}), layer)
        if any(s.suppressed for s in segments):
            caveats += (f"Segments below {config.analytics.min_segment_size} rows are suppressed.",)
        return envelope([_segment_dict(s) for s in segments], caveats=caveats)

    @mcp.tool(
        description="Rank active customers by expected revenue loss (churn probability "
        "x ARR) with the top reasons for each. For how accurate the model is, use "
        "model_card."
    )
    def at_risk_customers(top_n: int = 20) -> dict[str, Any]:
        with con.cursor() as cur:
            rankings = model.at_risk_customers(cur, layer, config, top_n, trained_model())
        return envelope(
            [_risk_dict(r) for r in rankings],
            caveats=("Scores are correlational; validate with a holdout before acting on them.",),
        )

    @mcp.tool(
        description="Show the churn model's performance: out-of-fold AUC, features used and "
        "CV folds. Use for any question about the model's score, accuracy or quality."
    )
    def model_card() -> dict[str, Any]:
        try:
            card = model.load_model_card(config)
        except ModelCardMissing as missing:
            return error_envelope(str(missing))
        return envelope(asdict(card), caveats=card.caveats())

    @mcp.resource(
        "churn://schema",
        description="Full customers table schema: every column's role, description and "
        "allowed values, plus metric definitions.",
    )
    def schema() -> dict[str, Any]:
        return {
            "table": layer.table,
            "grain": layer.grain,
            "columns": {
                name: {"role": c.role, "desc": c.desc, "values": list(c.values)}
                for name, c in layer.columns.items()
            },
            "metrics": layer.metrics,
        }

    @mcp.resource(
        "churn://model-card",
        description="Churn model performance: out-of-fold AUC, features used, CV folds.",
    )
    def model_card_resource() -> dict[str, Any]:
        try:
            return {"available": True, **asdict(model.load_model_card(config))}
        except ModelCardMissing as missing:
            return {"available": False, "message": str(missing)}

    return mcp
