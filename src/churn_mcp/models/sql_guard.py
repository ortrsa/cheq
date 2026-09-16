from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    sql: str
    error: str | None = None
    warnings: tuple[str, ...] = ()
    rewrites: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.error is None
