import pytest


def test_every_declared_column_exists_in_the_view(con, layer):
    actual = {r[0] for r in con.execute("DESCRIBE customers").fetchall()}
    assert layer.names() - actual == set()


def test_every_view_column_is_declared(con, layer):
    actual = {r[0] for r in con.execute("DESCRIBE customers").fetchall()}
    assert actual - layer.names() == set()


def test_declared_value_domains_match_the_data(con, layer):
    for column, declared in layer.categorical().items():
        actual = {r[0] for r in con.execute(f"SELECT DISTINCT {column} FROM customers").fetchall()}
        unexpected = actual - set(declared)
        assert not unexpected, f"{column} has undeclared values {unexpected}"


def test_leaky_and_protected_roles_are_populated(layer):
    assert set(layer.leaky()) == {"satisfaction_score", "churn_score"}
    assert "gender" in layer.protected()


@pytest.mark.parametrize("index", range(15))
def test_verified_examples_execute(con, layer, index):
    example = layer.examples[index]
    assert con.execute(example["sql"]).fetchall()


def test_context_block_honours_ablation_toggles(config, layer):
    full = layer.context_block(config)
    assert "Fiber Optic" in full and "TRAPS" in full and "VERIFIED EXAMPLES" in full

    stripped = config.model_copy(
        update={
            "t2sql": config.t2sql.model_copy(
                update={
                    "include_value_domains": False,
                    "include_traps": False,
                    "include_examples": False,
                }
            )
        }
    )
    minimal = layer.context_block(stripped)
    assert "TRAPS" not in minimal and "VERIFIED EXAMPLES" not in minimal
    assert "exact values" not in minimal
    assert len(minimal) < len(full)
