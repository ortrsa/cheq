import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "evals"))

from run_evals import rows_match  # noqa: E402


def test_extra_count_column_is_ignored():
    assert rows_match([("One Year", 0.11)], (("One Year", 1550, 0.11),))


def test_row_order_is_ignored():
    assert rows_match([(0, 0.33), (1, 0.20)], ((1, 0.20), (0, 0.33)))


def test_descriptive_label_matches_boolean_group():
    gold = [(False, 0.2936), (True, 0.2020)]
    predicted = (("At least one add-on", 4870, 0.2936), ("Zero add-ons", 2173, 0.2020))
    assert rows_match(gold, predicted)


def test_fuller_breakdown_than_asked_still_matches():
    assert rows_match([("Fiber Optic", 91.53)], (("Fiber Optic", 91.53), ("DSL", 58.18)))


def test_float_tolerance_is_four_decimals():
    assert rows_match([(0.26536987,)], ((0.26537,),))


def test_wrong_number_fails():
    assert not rows_match([(0, 0.33), (1, 0.20)], ((0, 0.37), (1, 0.20)))


def test_missing_row_fails():
    assert not rows_match([(0, 0.33), (1, 0.20)], ((0, 0.33),))
