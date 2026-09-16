import pytest
from hypothesis import given
from hypothesis import strategies as st

from churn_mcp import sql_guard


def check(sql, layer, config):
    return sql_guard.validate(sql, layer, config)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE customers",
        "DELETE FROM customers",
        "UPDATE customers SET churn = 0",
        "INSERT INTO customers VALUES (1)",
        "CREATE TABLE evil AS SELECT 1",
        "ATTACH '/tmp/evil.db' AS evil",
        "PRAGMA database_list",
    ],
)
def test_non_select_statements_are_blocked(sql, layer, config):
    assert not check(sql, layer, config).ok


def test_multiple_statements_are_blocked(layer, config):
    verdict = check("SELECT 1; DROP TABLE customers", layer, config)
    assert not verdict.ok
    assert "one statement" in verdict.error


def test_unparseable_sql_is_blocked(layer, config):
    assert not check("SELECT FROM WHERE ***", layer, config).ok


def test_file_reading_functions_are_blocked(layer, config):
    verdict = check("SELECT * FROM read_csv_auto('/etc/passwd')", layer, config)
    assert not verdict.ok
    assert "read_csv_auto" in verdict.error


def test_unknown_table_is_blocked(layer, config):
    verdict = check("SELECT * FROM secrets", layer, config)
    assert not verdict.ok
    assert "customers" in verdict.error


def test_unknown_column_is_blocked_with_a_suggestion(layer, config):
    verdict = check("SELECT monthly_charges FROM customers", layer, config)
    assert not verdict.ok
    assert "monthly_charge" in verdict.error


def test_wrong_literal_is_blocked_with_the_exact_value(layer, config):
    verdict = check("SELECT * FROM customers WHERE internet_type = 'fiber'", layer, config)
    assert not verdict.ok
    assert "Fiber Optic" in verdict.error


def test_wrong_literal_inside_in_list_is_blocked(layer, config):
    verdict = check("SELECT * FROM customers WHERE contract IN ('Monthly')", layer, config)
    assert not verdict.ok
    assert "Month-to-Month" in verdict.error


def test_correct_literal_passes(layer, config):
    assert check("SELECT * FROM customers WHERE contract = 'Two Year'", layer, config).ok


def test_missing_limit_is_added(layer, config):
    verdict = check("SELECT customer_id FROM customers", layer, config)
    assert verdict.ok
    assert f"LIMIT {config.security.max_rows}" in verdict.sql


def test_oversized_limit_is_lowered(layer, config):
    verdict = check("SELECT customer_id FROM customers LIMIT 99999", layer, config)
    assert f"LIMIT {config.security.max_rows}" in verdict.sql


def test_small_limit_is_kept(layer, config):
    assert "LIMIT 5" in check("SELECT customer_id FROM customers LIMIT 5", layer, config).sql


def test_grouped_rate_gains_a_row_count(layer, config):
    verdict = check("SELECT contract, AVG(churn) FROM customers GROUP BY contract", layer, config)
    assert "COUNT(*) AS n" in verdict.sql
    assert "added COUNT(*) AS n" in verdict.rewrites


def test_grouped_rate_that_already_counts_is_untouched(layer, config):
    sql = "SELECT contract, COUNT(*) AS n, AVG(churn) FROM customers GROUP BY contract"
    assert "added COUNT(*) AS n" not in check(sql, layer, config).rewrites


def test_leaky_column_passes_but_warns(layer, config):
    verdict = check(
        "SELECT satisfaction_score, AVG(churn) FROM customers GROUP BY satisfaction_score",
        layer,
        config,
    )
    assert verdict.ok
    assert verdict.warnings and "satisfaction_score" in verdict.warnings[0]


def test_clean_query_has_no_warnings(layer, config):
    verdict = check("SELECT AVG(churn) AS churn_rate FROM customers", layer, config)
    assert verdict.ok and not verdict.warnings


def test_cte_names_and_aliases_are_accepted(layer, config):
    sql = (
        "WITH per_contract AS ("
        "  SELECT contract, AVG(churn) AS rate FROM customers GROUP BY contract"
        ") SELECT contract, rate FROM per_contract ORDER BY rate DESC"
    )
    assert check(sql, layer, config).ok


def test_verified_examples_all_pass_the_guard(layer, config):
    for example in layer.examples:
        assert check(example["sql"], layer, config).ok, example["q"]


@given(
    verb=st.sampled_from(
        ["DROP", "DELETE FROM", "UPDATE", "INSERT INTO", "TRUNCATE", "ALTER TABLE"]
    ),
    target=st.sampled_from(["customers", "x", "information_schema.tables"]),
)
def test_no_mutation_ever_passes(layer, config, verb, target):
    assert not check(f"{verb} {target}", layer, config).ok
