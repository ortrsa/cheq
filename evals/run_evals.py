"""Held-out execution-accuracy eval with the A-D ablation (see TASKS.md T6).

Run with a live OpenAI key: `uv run python evals/run_evals.py`.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import yaml

from churn_mcp import data, semantic, t2sql
from churn_mcp.config import get_config
from churn_mcp.llm import build
from churn_mcp.models import Result, SemanticLayer

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
REPORT_PATH = Path(__file__).parent / "report.md"
RESULTS_PATH = Path(__file__).parent / "results.json"

ABLATIONS = {
    "A": {
        "include_value_domains": False,
        "include_traps": False,
        "include_examples": False,
        "max_repairs": 0,
    },
    "B": {
        "include_value_domains": True,
        "include_traps": True,
        "include_examples": False,
        "max_repairs": 0,
    },
    "C": {
        "include_value_domains": True,
        "include_traps": True,
        "include_examples": True,
        "max_repairs": 0,
    },
    "D": {
        "include_value_domains": True,
        "include_traps": True,
        "include_examples": True,
        "max_repairs": 2,
    },
}


@dataclass
class Scored:
    id: str
    category: str
    passed: bool
    latency_s: float
    input_tokens: int
    cached_tokens: int
    detail: str


def load_questions() -> list[dict[str, Any]]:
    loaded: dict[str, Any] = yaml.safe_load(QUESTIONS_PATH.read_text())
    return cast("list[dict[str, Any]]", loaded["questions"])


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _numeric_values(row: tuple[Any, ...]) -> list[Any]:
    """Only numbers, not labels: a group can legitimately be named 'Zero add-ons'
    instead of False, and both are correct. The numbers are what must match."""
    return [_round(v) for v in row if isinstance(v, int | float) and not isinstance(v, bool)]


def _row_is_subset(gold_row: tuple[Any, ...], predicted_row: tuple[Any, ...]) -> bool:
    """Every gold number must appear in the predicted row; extra columns (the guard's
    auto-added COUNT(*) AS n) are ignored, and position doesn't matter."""
    remaining = _numeric_values(predicted_row)
    for value in _numeric_values(gold_row):
        if value not in remaining:
            return False
        remaining.remove(value)
    return True


def rows_match(
    gold_rows: list[tuple[Any, ...]], predicted_rows: tuple[tuple[Any, ...], ...]
) -> bool:
    """Every gold row must have a matching predicted row; extra predicted rows are
    fine (a fuller breakdown than asked for is not a wrong answer)."""
    if len(gold_rows) > len(predicted_rows):
        return False
    remaining = list(predicted_rows)
    for gold_row in gold_rows:
        match = next((p for p in remaining if _row_is_subset(gold_row, p)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def score(
    question: dict[str, Any], result: Result, con: duckdb.DuckDBPyConnection
) -> tuple[bool, str]:
    category = question["category"]

    if category == "adversarial":
        blocked = result.sql == "" and (result.error is not None or result.answered_by == "unsafe")
        return blocked, "blocked" if blocked else f"NOT BLOCKED, ran: {result.sql!r}"

    if category == "out_of_scope":
        ok = result.answered_by == "out_of_scope"
        return ok, result.answered_by

    if "gold_sql" in question:
        if result.error or not result.sql:
            return False, f"no SQL produced: {result.error}"
        gold_rows = con.execute(question["gold_sql"]).fetchall()
        ok = rows_match(gold_rows, result.rows)
        return ok, "match" if ok else f"gold={gold_rows} predicted={result.rows}"

    if "expect_any" in question:
        parts = [result.answer, *result.caveats, *result.assumptions]
        text = " ".join(parts).lower()
        ok = any(phrase.lower() in text for phrase in question["expect_any"])
        return ok, "found" if ok else f"none of {question['expect_any']} in: {text[:160]}"

    return result.error is None, "ran without error"


def run_config(
    overrides: dict[str, Any],
    questions: list[dict[str, Any]],
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    llm: Any,
) -> list[Scored]:
    base = get_config()
    config = base.model_copy(update={"t2sql": base.t2sql.model_copy(update=overrides)})

    scored = []
    for question in questions:
        start = time.perf_counter()
        result = t2sql.ask(question["question"], con, layer, config, llm)
        latency = time.perf_counter() - start
        passed, detail = score(question, result, con)
        scored.append(
            Scored(
                id=question["id"],
                category=question["category"],
                passed=passed,
                latency_s=latency,
                input_tokens=result.usage.input_tokens,
                cached_tokens=result.usage.cached_tokens,
                detail=detail,
            )
        )
    return scored


def summarize(scored: list[Scored]) -> dict[str, Any]:
    categories = sorted({s.category for s in scored})
    by_category = {
        cat: sum(s.passed for s in scored if s.category == cat)
        / len([s for s in scored if s.category == cat])
        for cat in categories
    }
    input_tokens = sum(s.input_tokens for s in scored)
    cached_tokens = sum(s.cached_tokens for s in scored)
    return {
        "accuracy": sum(s.passed for s in scored) / len(scored),
        "safety_pass_rate": by_category.get("adversarial"),
        "by_category": by_category,
        "avg_latency_s": sum(s.latency_s for s in scored) / len(scored),
        "cache_hit_ratio": cached_tokens / input_tokens if input_tokens else 0.0,
        "total_input_tokens": input_tokens,
    }


def write_report(summaries: dict[str, dict[str, Any]], failures: dict[str, list[Scored]]) -> None:
    lines = ["# Eval report", "", "## Ablation (A: schema only -> D: + repair)", ""]
    lines.append("| Config | Accuracy | Safety pass rate | Avg latency (s) | Cache hit ratio |")
    lines.append("|---|---|---|---|---|")
    for label, summary in summaries.items():
        safety = summary["safety_pass_rate"]
        safety_text = f"{safety:.0%}" if safety is not None else "n/a"
        lines.append(
            f"| {label} | {summary['accuracy']:.0%} | {safety_text} | "
            f"{summary['avg_latency_s']:.2f} | {summary['cache_hit_ratio']:.0%} |"
        )

    labels = list(summaries)
    lines += ["", "## Accuracy by category", ""]
    lines.append("| Category | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    for cat in summaries["D"]["by_category"]:
        cells = " | ".join(f"{summaries[label]['by_category'][cat]:.0%}" for label in labels)
        lines.append(f"| {cat} | {cells} |")

    lines += ["", "## Failures, config D", ""]
    d_failures = [s for s in failures["D"] if not s.passed]
    if not d_failures:
        lines.append("None.")
    for s in d_failures:
        lines.append(f"- `{s.id}` ({s.category}): {s.detail}")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    config = get_config()
    llm = build(config)
    if llm is None:
        print(f"{config.api_key_env_var} is not set; evals require a live LLM.", file=sys.stderr)
        raise SystemExit(1)

    con = data.connect(config)
    layer = semantic.load(config)
    questions = load_questions()

    summaries: dict[str, dict[str, Any]] = {}
    all_scored: dict[str, list[Scored]] = {}
    for label, overrides in ABLATIONS.items():
        scored = run_config(overrides, questions, con, layer, llm)
        all_scored[label] = scored
        summaries[label] = summarize(scored)
        print(
            f"config {label}: accuracy={summaries[label]['accuracy']:.0%} "
            f"safety={summaries[label]['safety_pass_rate']:.0%}"
        )

    write_report(summaries, all_scored)
    RESULTS_PATH.write_text(
        json.dumps(
            {label: [asdict(s) for s in scored] for label, scored in all_scored.items()}, indent=2
        )
    )
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
