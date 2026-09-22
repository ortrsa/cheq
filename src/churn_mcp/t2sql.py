import re
from typing import Any

import duckdb

from churn_mcp import model, prompts, sql_guard
from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.exceptions import LLMRefused, ModelCardMissing
from churn_mcp.llm import LLM
from churn_mcp.models import Draft, Narrative, Result, Route, SemanticLayer, Usage

# Lookbehind excludes digits glued to letters, e.g. "Q3", from matching as a number.
NUMBER = re.compile(r"(?<![A-Za-z0-9.])-?\d[\d,]*(?:\.\d+)?%?")


def route(question: str, llm: LLM, config: TelcoChurnMcpConfig) -> tuple[dict[str, Any], Usage]:
    return llm.complete(
        model=config.t2sql.router_model,
        effort=config.t2sql.router_effort,
        developer=prompts.ROUTER,
        user=question,
        schema=Route.model_json_schema(),
        schema_name="route",
        max_output_tokens=256,
    )


def generate(
    question: str,
    layer: SemanticLayer,
    llm: LLM,
    config: TelcoChurnMcpConfig,
    feedback: str | None = None,
) -> tuple[dict[str, Any], Usage]:
    user = (
        question if feedback is None else f"{question}\n\nYour previous attempt failed.\n{feedback}"
    )
    return llm.complete(
        model=config.t2sql.generator_model,
        effort=config.t2sql.generator_effort,
        developer=prompts.GENERATOR + layer.context_block(config),
        user=user,
        schema=Draft.model_json_schema(),
        schema_name="generate_sql",
        max_output_tokens=2048,
    )


def execute(
    sql: str, con: duckdb.DuckDBPyConnection
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
    cursor = con.execute(sql)
    columns = tuple(d[0] for d in cursor.description or ())
    return columns, tuple(tuple(row) for row in cursor.fetchall())


def synthesize(
    question: str,
    interpretation: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    llm: LLM,
    config: TelcoChurnMcpConfig,
    retry_note: str = "",
) -> tuple[dict[str, Any], Usage]:
    shown = rows[: config.t2sql.max_result_rows_to_llm]
    table = "\n".join(str(dict(zip(columns, row, strict=False))) for row in shown)
    user = (
        f"Question: {question}\nInterpretation: {interpretation}\n"
        f"Rows ({len(rows)} total, showing {len(shown)}):\n{table}{retry_note}"
    )
    return llm.complete(
        model=config.t2sql.synthesizer_model,
        effort=config.t2sql.synthesizer_effort,
        developer=prompts.SYNTHESIZER,
        user=user,
        schema=Narrative.model_json_schema(),
        schema_name="synthesize",
        max_output_tokens=1024,
    )


def _cell_values(rows: tuple[tuple[Any, ...], ...]) -> list[float]:
    values = []
    for row in rows:
        for cell in row:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, int | float):
                values.append(float(cell))
    return values


def ungrounded(answer: str, rows: tuple[tuple[Any, ...], ...], question: str = "") -> list[str]:
    """Numbers in the narrative that neither a result cell nor the question supports."""
    cells = _cell_values(rows)
    # Numbers the user wrote, like "under 12 months", are restated, not computed.
    asked = [float(token.rstrip("%").replace(",", "")) for token in NUMBER.findall(question)]
    # Absolute values too: "14.1 points below" is how a narrative reads a -0.141 difference.
    allowed = cells + [abs(cell) for cell in cells] + asked + [float(len(rows))]
    missing = []
    for token in NUMBER.findall(answer):
        cleaned = token.rstrip("%").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        candidates = [value, value / 100] if token.endswith("%") else [value]
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        if any(
            any(
                abs(cell - c) < 1e-9 or round(cell, decimals) == round(c, decimals)
                for c in candidates
            )
            or any(abs(cell * 100 - c) < 0.05 for c in candidates)
            for cell in allowed
        ):
            continue
        missing.append(token)
    return missing


def _answer_from_model_card(result: Result, config: TelcoChurnMcpConfig) -> Result:
    result.answered_by = "model_card"
    try:
        card = model.load_model_card(config)
    except ModelCardMissing as missing:
        result.error = str(missing)
        return result
    result.columns, result.rows = card.table()
    result.answer = card.summary()
    result.caveats = card.caveats()
    result.grounding = "failed" if ungrounded(result.answer, result.rows) else "passed"
    return result


def _reject(result: Result, intent: str) -> Result:
    result.answered_by = intent
    result.error = f"Question rejected as {intent}."
    return result


def _answer_with_sql(
    question: str,
    result: Result,
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    llm: LLM,
    synthesise: bool,
) -> Result:
    feedback: str | None = None
    for attempt in range(config.t2sql.max_repairs + 1):
        result.attempts = attempt + 1
        drafted, usage = generate(question, layer, llm, config, feedback)
        result.usage += usage
        result.interpretation = drafted["interpretation"]
        result.assumptions = tuple(drafted["assumptions"])
        result.confidence = drafted["confidence"]

        verdict = sql_guard.validate(drafted["sql"], layer, config)
        if not verdict.ok:
            feedback = verdict.error
            continue

        result.sql = verdict.sql
        result.caveats = verdict.warnings
        try:
            result.columns, result.rows = execute(verdict.sql, con)
        except duckdb.Error as error:
            feedback = f"The query failed to run: {error}"
            continue
        break
    else:
        result.error = f"Could not produce a valid query. Last problem: {feedback}"
        return result

    if not synthesise:
        return result

    narrated, usage = synthesize(
        question, result.interpretation, result.columns, result.rows, llm, config
    )
    result.usage += usage
    result.answer = narrated["answer"]
    result.caveats += tuple(narrated["caveats"])

    missing = ungrounded(result.answer, result.rows, question)
    if missing:
        narrated, usage = synthesize(
            question,
            result.interpretation,
            result.columns,
            result.rows,
            llm,
            config,
            retry_note=f"\n\nThese numbers were not in the rows: {', '.join(missing)}. "
            "Rewrite using only values present above.",
        )
        result.usage += usage
        if ungrounded(narrated["answer"], result.rows, question):
            result.answer = ""
            result.grounding = "failed"
            result.caveats += ("Narrative withheld: numbers could not be verified.",)
            return result
        result.answer = narrated["answer"]
    result.grounding = "passed"
    return result


def ask(
    question: str,
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    llm: LLM | None,
    synthesise: bool = True,
) -> Result:
    result = Result(question=question)
    if llm is None:
        result.error = (
            "LLM disabled: set OPENAI_API_KEY to use ask_data. "
            "run_sql, describe_dataset and the analytic tools work without a key."
        )
        result.answered_by = "disabled"
        return result

    try:
        routed, usage = route(question, llm, config)
        result.usage += usage
        match routed["intent"]:
            case "data_query":
                normalized = routed["normalized_question"]
                return _answer_with_sql(normalized, result, con, layer, config, llm, synthesise)
            case "model_info":
                return _answer_from_model_card(result, config)
            case "unsafe" | "out_of_scope":
                return _reject(result, routed["intent"])
            case unknown:
                raise ValueError(f"intent '{unknown}' is in INTENTS but has no handler in ask()")
    except LLMRefused as refusal:
        result.error = f"The model declined to answer: {refusal}"
        result.answered_by = "refused"
        return result
