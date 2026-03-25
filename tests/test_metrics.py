from metrics.inequality import gini_coefficient, lorenz_curve
from metrics.descriptive import mean, median, variance


def test_gini_equal_distribution():
    values = [10.0, 10.0, 10.0, 10.0]
    gini = gini_coefficient(values)
    assert abs(gini - 0.0) < 1e-9


def test_gini_unequal_distribution():
    values = [0.0, 0.0, 0.0, 100.0]
    gini = gini_coefficient(values)
    assert gini > 0.7


def test_lorenz_curve_starts_at_zero():
    curve = lorenz_curve([10.0, 20.0, 30.0])
    assert curve["population_share"][0] == 0.0
    assert curve["value_share"][0] == 0.0


def test_descriptive_stats():
    values = [1.0, 2.0, 3.0, 4.0]
    assert mean(values) == 2.5
    assert median(values) == 2.5
    assert variance(values) == 1.25