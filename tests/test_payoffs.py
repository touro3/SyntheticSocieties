"""Tests for canonical payoff constants."""
import pytest
from environment.payoffs import GamePayoffs, DEFAULT_PAYOFFS


def test_default_payoffs_match_documented_values():
    assert DEFAULT_PAYOFFS.work_income == 10.0
    assert DEFAULT_PAYOFFS.work_stress_increase == 1.0
    assert DEFAULT_PAYOFFS.save_stress_relief == -0.2
    assert DEFAULT_PAYOFFS.cooperate_stress_relief == -0.1


def test_payoffs_are_frozen():
    with pytest.raises(AttributeError):
        DEFAULT_PAYOFFS.work_income = 999.0


def test_custom_payoffs():
    custom = GamePayoffs(work_income=5.0)
    assert custom.work_income == 5.0
    assert custom.save_stress_relief == -0.2  # other defaults unchanged
