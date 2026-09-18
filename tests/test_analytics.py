import pytest

from churn_mcp import analytics
from churn_mcp.exceptions import UnknownColumn, UnknownValue


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        (26, 100, (0.184047, 0.353710)),
        (1, 30, (0.005909, 0.166704)),
        (1869, 7043, (0.255188, 0.275807)),
        (5, 10, (0.236593, 0.763407)),
    ],
)
def test_wilson_ci_matches_statsmodels(k, n, expected):
    low, high = analytics.wilson_ci(k, n)
    assert low == pytest.approx(expected[0], abs=1e-5)
    assert high == pytest.approx(expected[1], abs=1e-5)


def test_wilson_ci_empty_segment():
    assert analytics.wilson_ci(0, 0) == (0.0, 0.0)


def test_two_prop_z_matches_statsmodels():
    p = analytics.two_prop_z(50, 100, 1869, 7043)
    assert p == pytest.approx(1.471398e-07, rel=1e-4)


def test_two_prop_z_identical_proportions_is_one():
    assert analytics.two_prop_z(1869, 7043, 1869, 7043) == pytest.approx(1.0)


def test_two_prop_z_empty_group_is_one():
    assert analytics.two_prop_z(0, 0, 1869, 7043) == 1.0


def test_bh_qvalues_matches_statsmodels():
    pvalues = [0.001, 0.01, 0.02, 0.03, 0.04, 0.5, 0.6, 0.9]
    expected = [0.008, 0.04, 0.053333, 0.06, 0.064, 0.666667, 0.685714, 0.9]
    qvalues = analytics.bh_qvalues(pvalues)
    for actual, want in zip(qvalues, expected, strict=True):
        assert actual == pytest.approx(want, abs=1e-5)


def test_bh_qvalues_is_monotonic_under_sorted_order():
    pvalues = [0.9, 0.001, 0.5, 0.02]
    qvalues = analytics.bh_qvalues(pvalues)
    order = sorted(range(4), key=lambda i: pvalues[i])
    sorted_q = [qvalues[i] for i in order]
    assert sorted_q == sorted(sorted_q)


def test_segment_churn_matches_independent_sql(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["contract"])
    by_group = {s.group: s for s in segments}

    for contract in ("Month-to-Month", "One Year", "Two Year"):
        expected_n, expected_churned = con.execute(
            "SELECT COUNT(*), SUM(churn) FROM customers WHERE contract = ?", [contract]
        ).fetchone()
        segment = by_group[(contract,)]
        assert segment.n == expected_n
        assert segment.churned == expected_churned
        assert segment.churn_rate == pytest.approx(expected_churned / expected_n)


def test_segment_churn_mrr_lost_matches_independent_sql(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["internet_type"])
    for segment in segments:
        expected = con.execute(
            "SELECT SUM(monthly_charge) FILTER (WHERE churn = 1) FROM customers "
            "WHERE internet_type = ?",
            [segment.group[0]],
        ).fetchone()[0]
        assert segment.mrr_lost == pytest.approx(expected or 0.0)


def test_segment_churn_respects_filters(con, layer, config):
    segments = analytics.segment_churn(
        con, layer, config, group_by=["contract"], filters={"internet_type": "Fiber Optic"}
    )
    total_filtered = sum(s.n for s in segments)
    expected = con.execute(
        "SELECT COUNT(*) FROM customers WHERE internet_type = 'Fiber Optic'"
    ).fetchone()[0]
    assert total_filtered == expected


def test_segment_churn_filter_accepts_a_list(con, layer, config):
    segments = analytics.segment_churn(
        con, layer, config, group_by=["contract"], filters={"contract": ["One Year", "Two Year"]}
    )
    assert {s.group[0] for s in segments} == {"One Year", "Two Year"}


def test_segment_churn_rejects_unknown_column(con, layer, config):
    with pytest.raises(UnknownColumn):
        analytics.segment_churn(con, layer, config, group_by=["not_a_column"])


def test_segment_churn_rejects_unknown_filter_value(con, layer, config):
    with pytest.raises(UnknownValue):
        analytics.segment_churn(
            con, layer, config, group_by=["contract"], filters={"internet_type": "fiber"}
        )


def test_segment_churn_flags_small_segments(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["city"], top_n=200)
    small = [s for s in segments if s.n < config.analytics.min_segment_size]
    assert small and all(s.suppressed for s in small)
    large = [s for s in segments if s.n >= config.analytics.min_segment_size]
    assert all(not s.suppressed for s in large)


def test_segment_churn_sorted_by_rate_descending(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["contract"])
    rates = [s.churn_rate for s in segments]
    assert rates == sorted(rates, reverse=True)


def test_segment_churn_top_n_limits_results(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["city"], top_n=5)
    assert len(segments) == 5


def test_segment_churn_qvalues_never_below_pvalues(con, layer, config):
    segments = analytics.segment_churn(con, layer, config, group_by=["city"], top_n=200)
    assert all(0.0 <= s.q_value <= 1.0 for s in segments)


def test_segment_churn_quotes_filter_values_safely(con, layer, config):
    segments = analytics.segment_churn(
        con, layer, config, group_by=["contract"], filters={"city": "O'Brien"}
    )
    assert segments == []


def test_segment_churn_filter_cannot_inject_sql(con, layer, config):
    segments = analytics.segment_churn(
        con, layer, config, group_by=["contract"], filters={"city": "x' OR 1=1 --"}
    )
    assert segments == []


@pytest.mark.parametrize("group_by", [[], ["contract", "internet_type", "offer", "city"]])
def test_segment_churn_rejects_wrong_dimension_count(con, layer, config, group_by):
    with pytest.raises(ValueError, match="group_by"):
        analytics.segment_churn(con, layer, config, group_by=group_by)
