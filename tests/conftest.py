import pytest

from churn_mcp import data, semantic
from churn_mcp.config import get_config


@pytest.fixture(scope="session")
def config():
    return get_config()


@pytest.fixture(scope="session")
def con(config):
    if not (config.data.artifacts_dir / "churn.duckdb").exists():
        data.build(config)
    return data.connect(config)


@pytest.fixture(scope="session")
def layer(config):
    return semantic.load(config)


@pytest.fixture
def sample_card():
    from churn_mcp.models import ModelCard

    return ModelCard(
        model="LogisticRegression",
        auc=0.8936,
        leaky_features=("satisfaction_score", "churn_score"),
        auc_with_leaky_features=0.9981,
        features=("cat__contract_One Year", "num__tenure_in_months", "num__monthly_charge"),
        cv_folds=5,
        rows=7043,
        protected_excluded=True,
    )


@pytest.fixture
def artifacts_config(config, tmp_path):
    return config.model_copy(
        update={"data": config.data.model_copy(update={"artifacts_dir": tmp_path})}
    )
