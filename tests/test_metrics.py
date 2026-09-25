import pytest

from medseek.metrics import average_precision


def test_average_precision_treats_tied_scores_as_one_threshold() -> None:
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_average_precision_without_ties_is_unchanged() -> None:
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(5 / 6)
