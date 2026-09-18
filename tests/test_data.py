import duckdb
import pytest

from churn_mcp import data


def test_row_count_matches_expected(con, config):
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == config.data.expected_rows


def test_dropped_columns_are_absent(con):
    cols = {r[0] for r in con.execute("DESCRIBE customers").fetchall()}
    assert cols.isdisjoint(data.DROPPED)


@pytest.mark.parametrize("column", ["offer", "internet_type", "churn_category", "churn_reason"])
def test_nullable_categoricals_are_filled(con, column):
    assert con.execute(f"SELECT COUNT(*) FROM customers WHERE {column} IS NULL").fetchone()[0] == 0


def test_is_active_is_the_complement_of_churn(con):
    mismatched = con.execute(
        "SELECT COUNT(*) FROM customers WHERE is_active = (customer_status = 'Churned')::INT"
    ).fetchone()[0]
    assert mismatched == 0


def test_arr_is_twelve_monthly_charges(con):
    assert (
        con.execute(
            "SELECT COUNT(*) FROM customers WHERE ABS(arr - monthly_charge * 12) > 0.01"
        ).fetchone()[0]
        == 0
    )


def test_num_addons_within_bounds(con):
    low, high = con.execute("SELECT MIN(num_addons), MAX(num_addons) FROM customers").fetchone()
    assert (low, high) == (0, len(data.ADDONS))


def test_tenure_buckets_cover_every_row(con):
    labels = {r[0] for r in con.execute("SELECT DISTINCT tenure_bucket FROM customers").fetchall()}
    assert labels == set(data.TENURE_LABELS)


def test_referral_buckets_match_the_referral_count(con):
    rows = con.execute(
        "SELECT referral_bucket, MIN(number_of_referrals), MAX(number_of_referrals) "
        "FROM customers GROUP BY referral_bucket ORDER BY referral_bucket"
    ).fetchall()
    assert [r[0] for r in rows] == list(data.REFERRAL_LABELS)
    assert rows[0][1:] == (0, 0)
    assert rows[1][1:] == (1, 1)
    assert rows[2][1:] == (2, 3)
    assert rows[3][1] == 4


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE evil AS SELECT 1",
        "DELETE FROM customers",
        "UPDATE customers SET churn = 0",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SET enable_external_access = true",
        "ATTACH '/tmp/evil.db' AS evil",
    ],
)
def test_connection_rejects_writes_and_file_access(con, sql):
    with pytest.raises(duckdb.Error):
        con.execute(sql)


def test_connect_twice_in_one_process(con, config):
    second = data.connect(config)
    assert (
        second.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == config.data.expected_rows
    )
    with pytest.raises(duckdb.Error):
        second.execute("SET enable_external_access = true")


def test_cursor_inherits_the_sandbox(con):
    with con.cursor() as cur, pytest.raises(duckdb.Error):
        cur.execute("SELECT * FROM read_csv_auto('/etc/passwd')")
