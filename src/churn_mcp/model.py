import json
from dataclasses import asdict

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_mcp.config import TelcoChurnMcpConfig
from churn_mcp.exceptions import LeakyFeature, ModelCardMissing
from churn_mcp.models import ModelCard, RiskRanking, SemanticLayer, Trained

FEATURE_ROLES = frozenset({"account", "service", "billing", "derived"})

ALWAYS_EXCLUDED = {
    "is_active": "derived from the outcome (is_active = 1 - churn)",
    "arr": "exactly monthly_charge x 12, so it would count one signal twice",
    "total_revenue": "exactly total_charges + long distance + extra data - refunds",
    "total_charges": "tracks tenure x monthly_charge (r=0.9996), a duplicate of both",
    "num_addons": "exactly the sum of the seven add-on flags",
    "internet_service": "exactly the inverse of internet_type = 'No Internet'",
    "tenure_in_months": "duplicates tenure_bucket; the bucket keeps the non-linear shape",
    "number_of_referrals": "linear in a non-linear effect; referral_bucket replaces it",
    "referred_a_friend": "exactly number_of_referrals > 0; referral_bucket replaces it",
}

CATEGORICAL_FEATURES = frozenset(
    {"contract", "offer", "payment_method", "internet_type", "tenure_bucket", "referral_bucket"}
)


def check_firewall(columns: list[str], layer: SemanticLayer, config: TelcoChurnMcpConfig) -> None:
    for column in columns:
        role = layer.columns[column].role
        if role in ("outcome", "leaky"):
            raise LeakyFeature(column, f"role is '{role}', which encodes the outcome")
        if role == "protected" and config.ml.exclude_protected:
            raise LeakyFeature(column, "protected attribute, excluded by default")
        if column in ALWAYS_EXCLUDED:
            raise LeakyFeature(column, ALWAYS_EXCLUDED[column])


def feature_columns(layer: SemanticLayer, config: TelcoChurnMcpConfig) -> list[str]:
    allowed_roles = set(FEATURE_ROLES)
    if not config.ml.exclude_protected:
        allowed_roles.add("protected")
    return [
        c.name
        for c in layer.columns.values()
        if c.role in allowed_roles and c.name not in ALWAYS_EXCLUDED
    ]


def _pipeline(columns: list[str], seed: int) -> Pipeline:
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def _frame(con: duckdb.DuckDBPyConnection, columns: list[str]) -> pd.DataFrame:
    cols = ", ".join([*columns, "churn", "arr", "is_active", "customer_id"])
    return con.execute(f"SELECT {cols} FROM customers").fetchdf()


def train(
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    columns: list[str] | None = None,
) -> Trained:
    columns = columns if columns is not None else feature_columns(layer, config)
    check_firewall(columns, layer, config)
    return _fit(con, columns, config)


def _fit(
    con: duckdb.DuckDBPyConnection, columns: list[str], config: TelcoChurnMcpConfig
) -> Trained:
    frame = _frame(con, columns)
    x, y = frame[columns], frame["churn"]
    cv = StratifiedKFold(n_splits=config.ml.cv_folds, shuffle=True, random_state=config.ml.seed)
    pipeline = _pipeline(columns, config.ml.seed)

    oof_proba = cross_val_predict(pipeline, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof_proba)

    pipeline.fit(x, y)
    names = tuple(pipeline.named_steps["preprocess"].get_feature_names_out())
    return Trained(pipeline=pipeline, oof_proba=oof_proba, auc=auc, feature_names=names)


def _source_column(encoded_name: str) -> str:
    kind, name = encoded_name.split("__", 1)
    if kind == "num":
        return name
    return next(c for c in CATEGORICAL_FEATURES if name.startswith(f"{c}_"))


def at_risk_customers(
    con: duckdb.DuckDBPyConnection,
    layer: SemanticLayer,
    config: TelcoChurnMcpConfig,
    top_n: int = 20,
    trained: Trained | None = None,
) -> list[RiskRanking]:
    columns = feature_columns(layer, config)
    if trained is None:
        trained = train(con, layer, config, columns)

    frame = _frame(con, columns)
    preprocess = trained.pipeline.named_steps["preprocess"]
    model = trained.pipeline.named_steps["model"]
    transformed = preprocess.transform(frame[columns])
    transformed = np.asarray(
        transformed.todense() if hasattr(transformed, "todense") else transformed
    )
    # Centre every encoded column, one-hots included, so each contribution is measured
    # against the average customer rather than against an arbitrary dropped category.
    contributions = pd.DataFrame(
        (transformed - transformed.mean(axis=0)) * model.coef_[0],
        columns=[_source_column(name) for name in trained.feature_names],
    )
    # Sum the one-hots back into their source column: one reason per business feature.
    by_column = contributions.T.groupby(level=0).sum().T

    active = frame[frame["is_active"] == 1].copy()
    active["proba"] = trained.oof_proba[active.index]
    active["expected_loss"] = active["proba"] * active["arr"]
    ranked = active.sort_values("expected_loss", ascending=False).head(top_n)

    rankings = []
    for row_position in ranked.index:
        risk_raising = by_column.loc[row_position]
        top = risk_raising[risk_raising > 0].nlargest(3)
        customer = ranked.loc[row_position]
        rankings.append(
            RiskRanking(
                customer_id=str(customer["customer_id"]),
                probability=float(customer["proba"]),
                arr=float(customer["arr"]),
                expected_loss=float(customer["expected_loss"]),
                reasons=tuple((f"{c}={customer[c]}", float(v)) for c, v in top.items()),
            )
        )
    return rankings


def build_model_card(
    con: duckdb.DuckDBPyConnection, layer: SemanticLayer, config: TelcoChurnMcpConfig
) -> ModelCard:
    clean = train(con, layer, config)
    return ModelCard(
        model=type(clean.pipeline.named_steps["model"]).__name__,
        auc=clean.auc,
        features=clean.feature_names,
        cv_folds=config.ml.cv_folds,
        rows=len(clean.oof_proba),
        protected_excluded=config.ml.exclude_protected,
    )


def write_model_card(card: ModelCard, config: TelcoChurnMcpConfig) -> None:
    path = config.data.artifacts_dir / "model_card.json"
    path.write_text(json.dumps(asdict(card), indent=2))


def load_model_card(config: TelcoChurnMcpConfig) -> ModelCard:
    path = config.data.artifacts_dir / "model_card.json"
    if not path.exists():
        raise ModelCardMissing()
    raw = json.loads(path.read_text())
    return ModelCard(**{**raw, "features": tuple(raw["features"])})
