from dataclasses import asdict

import numpy as np
import pytest

from churn_mcp import model
from churn_mcp.exceptions import LeakyFeature, ModelCardMissing


@pytest.mark.parametrize("column", ["churn", "customer_status", "churn_category", "churn_reason"])
def test_firewall_blocks_outcome_columns(layer, config, column):
    with pytest.raises(LeakyFeature):
        model.check_firewall([column], layer, config)


@pytest.mark.parametrize("column", ["satisfaction_score", "churn_score"])
def test_firewall_blocks_leaky_columns(layer, config, column):
    with pytest.raises(LeakyFeature):
        model.check_firewall([column], layer, config)


def test_firewall_blocks_protected_columns_by_default(layer, config):
    with pytest.raises(LeakyFeature):
        model.check_firewall(["gender"], layer, config)


def test_firewall_allows_protected_when_config_permits(layer, config):
    permissive = config.model_copy(
        update={"ml": config.ml.model_copy(update={"exclude_protected": False})}
    )
    model.check_firewall(["gender"], layer, permissive)  # does not raise


def test_firewall_blocks_is_active_despite_derived_role(layer, config):
    assert layer.columns["is_active"].role == "derived"
    with pytest.raises(LeakyFeature):
        model.check_firewall(["is_active"], layer, config)


def test_firewall_allows_ordinary_features(layer, config):
    model.check_firewall(["contract", "tenure_bucket", "monthly_charge"], layer, config)


def test_feature_columns_excludes_arr_and_is_active(layer, config):
    columns = model.feature_columns(layer, config)
    assert "arr" not in columns
    assert "is_active" not in columns
    assert "churn" not in columns
    assert "contract" in columns


def test_feature_columns_excludes_protected_by_default(layer, config):
    assert "gender" not in model.feature_columns(layer, config)


def test_trained_auc_beats_a_reasonable_floor(con, layer, config):
    trained = model.train(con, layer, config)
    assert 0.80 <= trained.auc <= 0.97


def test_out_of_fold_scores_are_not_perfectly_separated(con, layer, config):
    """A model scored on data it was trained on would show near-perfect separation
    on this dataset; OOF scoring must not let that happen."""
    trained = model.train(con, layer, config)
    assert trained.oof_proba.min() > 0.0
    assert trained.oof_proba.max() < 1.0
    assert 0.0 < np.mean(trained.oof_proba) < 1.0


def test_oof_scores_differ_from_in_sample_scores(con, layer, config):
    trained = model.train(con, layer, config)
    columns = model.feature_columns(layer, config)
    frame = model._frame(con, columns)
    in_sample = trained.pipeline.predict_proba(frame[columns])[:, 1]
    assert not np.allclose(trained.oof_proba, in_sample)


def test_leaky_columns_would_inflate_auc(con, layer, config):
    """Backs the one-off ablation in the README; the model card no longer reports it."""
    clean = model.train(con, layer, config)
    leaky = model._fit(con, [*model.feature_columns(layer, config), *layer.leaky()], config)
    assert leaky.auc > clean.auc + 0.05


def test_model_card_trains_only_the_firewalled_model(con, layer, config, monkeypatch):
    real_fit = model._fit
    leaky = set(layer.leaky())

    def fit_without_leaks(con, columns, config):
        assert not leaky & set(columns)
        return real_fit(con, columns, config)

    monkeypatch.setattr(model, "_fit", fit_without_leaks)
    card = model.build_model_card(con, layer, config)
    assert not any("leaky" in field for field in asdict(card))


def test_model_card_describes_the_trained_model_not_hardcoded_facts(con, layer, config):
    card = model.build_model_card(con, layer, config)
    assert card.model == "LogisticRegression"
    assert card.rows == config.data.expected_rows
    assert card.cv_folds == config.ml.cv_folds


def test_train_always_applies_the_firewall(con, layer, config):
    with pytest.raises(LeakyFeature):
        model.train(con, layer, config, columns=["satisfaction_score"])


def test_at_risk_customers_only_includes_active_customers(con, layer, config):
    ranked = model.at_risk_customers(con, layer, config, top_n=50)
    ids = [r.customer_id for r in ranked]
    active_ids = {
        row[0]
        for row in con.execute("SELECT customer_id FROM customers WHERE is_active = 1").fetchall()
    }
    assert set(ids) <= active_ids


def test_at_risk_customers_sorted_by_expected_loss_descending(con, layer, config):
    ranked = model.at_risk_customers(con, layer, config, top_n=20)
    losses = [r.expected_loss for r in ranked]
    assert losses == sorted(losses, reverse=True)


def test_at_risk_customers_probability_is_a_valid_probability(con, layer, config):
    for r in model.at_risk_customers(con, layer, config, top_n=10):
        assert 0.0 <= r.probability <= 1.0


def test_at_risk_customers_expected_loss_equals_probability_times_arr(con, layer, config):
    for r in model.at_risk_customers(con, layer, config, top_n=10):
        assert r.expected_loss == pytest.approx(r.probability * r.arr, rel=1e-6)


def test_at_risk_customers_returns_three_distinct_reasons(con, layer, config):
    for r in model.at_risk_customers(con, layer, config, top_n=10):
        names = [name for name, _ in r.reasons]
        assert len(names) == len(set(names))


def test_model_card_round_trips_through_disk(sample_card, artifacts_config):
    model.write_model_card(sample_card, artifacts_config)
    assert model.load_model_card(artifacts_config) == sample_card


def test_missing_model_card_raises(artifacts_config):
    with pytest.raises(ModelCardMissing):
        model.load_model_card(artifacts_config)


@pytest.mark.parametrize(
    "column",
    [
        "is_active",
        "arr",
        "total_revenue",
        "total_charges",
        "num_addons",
        "internet_service",
        "tenure_in_months",
        "number_of_referrals",
        "referred_a_friend",
    ],
)
def test_each_always_excluded_column_reports_its_own_reason(layer, config, column):
    with pytest.raises(LeakyFeature, match=model.ALWAYS_EXCLUDED[column].split(" ")[0]):
        model.check_firewall([column], layer, config)


def test_design_matrix_has_full_rank(con, layer, config):
    """Exact linear dependencies make per-feature contributions meaningless."""
    trained = model.train(con, layer, config)
    columns = model.feature_columns(layer, config)
    transformed = trained.pipeline.named_steps["preprocess"].transform(
        model._frame(con, columns)[columns]
    )
    matrix = np.column_stack([np.ones(transformed.shape[0]), np.asarray(transformed)])
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]


def test_at_risk_reasons_only_raise_risk(con, layer, config):
    for r in model.at_risk_customers(con, layer, config, top_n=20):
        assert all(value > 0 for _, value in r.reasons)


def test_at_risk_reasons_name_the_source_column_and_its_value(con, layer, config):
    columns = set(model.feature_columns(layer, config))
    for r in model.at_risk_customers(con, layer, config, top_n=20):
        for name, _ in r.reasons:
            column, _, value = name.partition("=")
            assert column in columns
            assert value


def test_at_risk_customers_reuses_a_trained_model(con, layer, config, monkeypatch):
    trained = model.train(con, layer, config)
    monkeypatch.setattr(model, "train", lambda *a, **k: pytest.fail("retrained"))
    assert model.at_risk_customers(con, layer, config, top_n=3, trained=trained)


def test_two_or_more_referrals_is_never_a_risk_reason(con, layer, config):
    for r in model.at_risk_customers(con, layer, config, top_n=100):
        names = [name for name, _ in r.reasons]
        assert "referral_bucket=2-3" not in names
        assert "referral_bucket=4+" not in names
