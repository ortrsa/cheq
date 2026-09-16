class LLMRefused(RuntimeError):
    """A safety refusal or empty output: a normal pipeline outcome, not a crash."""


class UnknownColumn(ValueError):
    def __init__(self, column: str, known: frozenset[str]) -> None:
        super().__init__(f"Unknown column '{column}'. Known columns: {', '.join(sorted(known))}")
        self.column = column


class UnknownValue(ValueError):
    def __init__(self, column: str, value: str, domain: tuple[str, ...]) -> None:
        super().__init__(f"'{value}' is not a value of {column}. Valid values: {', '.join(domain)}")
        self.column = column
        self.value = value


class LeakyFeature(ValueError):
    def __init__(self, column: str, reason: str) -> None:
        super().__init__(f"'{column}' cannot be used as a model feature: {reason}")
        self.column = column


class ModelCardMissing(FileNotFoundError):
    def __init__(self) -> None:
        super().__init__(
            "No model card yet. Run `churn-mcp prepare` to train the model and write one; "
            "at_risk_customers still works without it."
        )
