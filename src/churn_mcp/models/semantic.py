from dataclasses import dataclass

from churn_mcp.config import TelcoChurnMcpConfig

LEAKY_ROLES = frozenset({"leaky"})
PROTECTED_ROLES = frozenset({"protected"})


@dataclass(frozen=True)
class Column:
    name: str
    role: str
    desc: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticLayer:
    table: str
    grain: str
    columns: dict[str, Column]
    metrics: dict[str, str]
    traps: tuple[str, ...]
    defaults: dict[str, str]
    examples: tuple[dict[str, str], ...]

    def names(self) -> frozenset[str]:
        return frozenset(self.columns)

    def leaky(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.role in LEAKY_ROLES)

    def protected(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns.values() if c.role in PROTECTED_ROLES)

    def domain(self, column: str) -> tuple[str, ...]:
        found = self.columns.get(column)
        return found.values if found else ()

    def categorical(self) -> dict[str, tuple[str, ...]]:
        return {c.name: c.values for c in self.columns.values() if c.values}

    def context_block(self, config: TelcoChurnMcpConfig) -> str:
        t2sql = config.t2sql
        parts = [f"TABLE {self.table} -- {self.grain}", "", "COLUMNS"]
        for col in self.columns.values():
            line = f"  {col.name} [{col.role}] {col.desc}"
            if col.values and t2sql.include_value_domains:
                line += f" | exact values: {', '.join(col.values)}"
            parts.append(line)

        parts += ["", "METRICS"]
        parts += [f"  {name} := {sql}" for name, sql in self.metrics.items()]

        if t2sql.include_traps:
            parts += ["", "TRAPS"] + [f"  - {t}" for t in self.traps]
            parts += ["", "DEFAULT INTERPRETATIONS"]
            parts += [f"  {term} -> {rule}" for term, rule in self.defaults.items()]

        if t2sql.include_examples:
            parts += ["", "VERIFIED EXAMPLES"]
            for ex in self.examples:
                parts += [f"  Q: {ex['q']}", f"  A: {ex['sql']}"]

        return "\n".join(parts)
