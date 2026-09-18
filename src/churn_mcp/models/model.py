from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Trained:
    pipeline: Pipeline
    oof_proba: np.ndarray
    auc: float
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RiskRanking:
    customer_id: str
    probability: float
    arr: float
    expected_loss: float
    reasons: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ModelCard:
    model: str
    auc: float
    features: tuple[str, ...]
    cv_folds: int
    rows: int
    protected_excluded: bool

    def table(self) -> tuple[tuple[str, ...], tuple[tuple[object, ...], ...]]:
        return ("metric", "value"), (
            ("model", self.model),
            ("auc", self.auc),
            ("cv_folds", self.cv_folds),
            ("rows", self.rows),
            ("feature_count", len(self.features)),
            ("features", ", ".join(self.features)),
        )

    def summary(self) -> str:
        return (
            f"The churn model is {self.model} with an out-of-fold AUC of {self.auc:.3f} "
            f"({self.cv_folds}-fold cross-validation over {self.rows:,} customers, "
            f"{len(self.features)} encoded features)."
        )

    def caveats(self) -> tuple[str, ...]:
        return (
            f"AUC is out-of-fold over all {self.rows:,} customers, not the dataset's own "
            "train/test split.",
        )
