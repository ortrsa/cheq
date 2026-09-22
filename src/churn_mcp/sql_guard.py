import difflib
import re

import sqlglot
from sqlglot import exp

from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.models import SemanticLayer, Verdict

DENIED_FUNCTIONS = frozenset(
    {
        "pragma",
        "glob",
        "attach",
        "detach",
        "copy",
        "install",
        "load",
        "system",
        "shell",
        "current_setting",
    }
)
DENIED_PREFIXES = ("read_", "write_", "parquet_", "sniff_", "duckdb_")

AGGREGATES = (exp.Avg, exp.Sum, exp.Min, exp.Max)


def _blocked(sql: str, message: str) -> Verdict:
    return Verdict(sql=sql, error=message)


def _suggest(name: str, candidates: frozenset[str]) -> str:
    close = difflib.get_close_matches(name, sorted(candidates), n=1, cutoff=0.6)
    return f" Did you mean '{close[0]}'?" if close else ""


def _local_names(tree: exp.Expression) -> set[str]:
    """Names the query defines for itself: CTEs, table aliases and output aliases."""
    names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    names |= {alias.alias_or_name for alias in tree.find_all(exp.Alias)}
    names |= {ta.name for ta in tree.find_all(exp.TableAlias)}
    return {n for n in names if n}


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this or "").lower()
    return node.sql_name().lower()


def _check_functions(tree: exp.Expression) -> str | None:
    for node in tree.find_all(exp.Func):
        name = _function_name(node)
        if name in DENIED_FUNCTIONS or name.startswith(DENIED_PREFIXES):
            return f"Function '{name}' is not allowed. Query the customers table directly."
    return None


def _check_tables(tree: exp.Expression, allowed: frozenset[str], local: set[str]) -> str | None:
    for table in tree.find_all(exp.Table):
        name = table.name
        if name and name not in allowed and name not in local:
            return (
                f"Table '{name}' does not exist. "
                f"Only these are queryable: {', '.join(sorted(allowed))}."
            )
    return None


def _check_columns(tree: exp.Expression, layer: SemanticLayer, local: set[str]) -> str | None:
    known = layer.names()
    for column in tree.find_all(exp.Column):
        if column.is_star:
            continue
        name = column.name
        if name and name not in known and name not in local:
            return f"Column '{name}' does not exist.{_suggest(name, known)}"
    return None


def _check_literals(tree: exp.Expression, layer: SemanticLayer) -> str | None:
    def verify(column: exp.Expression, literal: exp.Expression) -> str | None:
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal):
            return None
        if not literal.is_string:
            return None
        domain = layer.domain(column.name)
        if not domain or literal.this in domain:
            return None
        lowered = {v.lower(): v for v in domain}
        exact = lowered.get(literal.this.lower())
        fix = f" Use '{exact}'." if exact else ""
        return (
            f"'{literal.this}' is not a value of {column.name}.{fix} "
            f"Valid values: {', '.join(domain)}."
        )

    for comparison in tree.find_all(exp.EQ, exp.NEQ):
        problem = verify(comparison.this, comparison.expression)
        if problem:
            return problem
    for member in tree.find_all(exp.In):
        for candidate in member.expressions:
            problem = verify(member.this, candidate)
            if problem:
                return problem
    return None


def _like_regex(pattern: str, ignore_case: bool) -> re.Pattern[str]:
    regex = re.escape(pattern).replace("%", ".*").replace("_", ".")
    return re.compile(f"^{regex}$", re.IGNORECASE if ignore_case else 0)


def _check_patterns(tree: exp.Expression, layer: SemanticLayer) -> str | None:
    for match in tree.find_all(exp.Like, exp.ILike):
        column, pattern = match.this, match.expression
        if not isinstance(column, exp.Column) or not isinstance(pattern, exp.Literal):
            continue
        if not pattern.is_string:
            continue
        domain = layer.domain(column.name)
        if not domain:
            continue
        regex = _like_regex(pattern.this, isinstance(match, exp.ILike))
        if not any(regex.match(v) for v in domain):
            return (
                f"'{pattern.this}' matches no value of {column.name}. "
                f"Valid values: {', '.join(domain)}."
            )
    return None


def _check_limit(tree: exp.Select) -> str | None:
    limit = tree.args.get("limit")
    if limit is None:
        return None
    current = limit.expression
    if isinstance(current, exp.Literal) and current.this.isdigit() and int(current.this) > 0:
        return None
    return f"LIMIT must be a positive integer, got {current.sql(dialect='duckdb')}."


def _enforce_limit(tree: exp.Select, max_rows: int) -> tuple[exp.Select, str | None]:
    limit = tree.args.get("limit")
    if limit is None:
        return tree.limit(max_rows), f"added LIMIT {max_rows}"
    current = limit.expression
    if isinstance(current, exp.Literal) and int(current.this) > max_rows:
        return tree.limit(max_rows), f"lowered LIMIT to {max_rows}"
    return tree, None


def _add_group_count(tree: exp.Select) -> tuple[exp.Select, str | None]:
    """A grouped rate without its denominator hides tiny segments, so add one."""
    if tree.find(exp.Group) is None:
        return tree, None
    if tree.find(exp.Count) is not None:
        return tree, None
    if tree.find(*AGGREGATES) is None:
        return tree, None
    return tree.select(exp.alias_(exp.Count(this=exp.Star()), "n")), "added COUNT(*) AS n"


def _selects_star(tree: exp.Expression) -> bool:
    # A star over a CTE only re-exposes columns the CTE already named, and those are
    # checked as columns; only a star read straight from a real table exposes everything.
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for select in tree.find_all(exp.Select):
        if not any(e.is_star for e in select.expressions):
            continue
        sources = {t.name for t in select.find_all(exp.Table) if t.parent_select is select}
        if sources - ctes:
            return True
    return False


def leak_warning(used: set[str], layer: SemanticLayer) -> tuple[str, ...]:
    leaking = sorted(used & set(layer.leaky()))
    if not leaking:
        return ()
    return (
        f"{', '.join(leaking)} is recorded at exit and separates churners almost perfectly, "
        "so it describes churn rather than explaining it.",
    )


def _leak_warning(tree: exp.Expression, layer: SemanticLayer) -> tuple[str, ...]:
    used = {c.name for c in tree.find_all(exp.Column)}
    if _selects_star(tree):
        used |= set(layer.names())
    return leak_warning(used, layer)


def validate(sql: str, layer: SemanticLayer, config: TelcoChurnMcpConfig) -> Verdict:
    try:
        statements = sqlglot.parse(sql, dialect="duckdb")
    except sqlglot.ParseError as error:
        return _blocked(sql, f"SQL does not parse: {error}")

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        return _blocked(sql, f"Expected exactly one statement, got {len(statements)}.")

    tree = statements[0]
    if not isinstance(tree, exp.Select):
        return _blocked(sql, f"Only SELECT queries are allowed, got {type(tree).__name__.upper()}.")

    local = _local_names(tree)
    for problem in (
        _check_functions(tree),
        _check_tables(tree, config.security.allowed_tables, local),
        _check_columns(tree, layer, local),
        _check_literals(tree, layer),
        _check_patterns(tree, layer),
        _check_limit(tree),
    ):
        if problem:
            return _blocked(sql, problem)

    warnings = _leak_warning(tree, layer)
    rewrites: list[str] = []
    tree, counted = _add_group_count(tree)
    tree, limited = _enforce_limit(tree, config.security.max_rows)
    rewrites += [note for note in (counted, limited) if note]

    return Verdict(sql=tree.sql(dialect="duckdb"), warnings=warnings, rewrites=tuple(rewrites))


def with_max_rows(config: TelcoChurnMcpConfig, max_rows: int | None) -> TelcoChurnMcpConfig:
    if max_rows is None:
        return config
    if max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    security = config.security.model_copy(
        update={"max_rows": min(max_rows, config.security.max_rows)}
    )
    return config.model_copy(update={"security": security})
