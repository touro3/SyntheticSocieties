"""Tests for canonical Gini implementation — single source of truth."""
import pytest
from metrics.inequality import gini_coefficient


class TestGiniCanonical:
    def test_perfectly_equal(self):
        assert gini_coefficient([100, 100, 100]) == 0.0

    def test_perfectly_unequal(self):
        g = gini_coefficient([0, 0, 100])
        assert abs(g - 2 / 3) < 0.01

    def test_single_value(self):
        assert gini_coefficient([42]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            gini_coefficient([])

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            gini_coefficient([-5, 10, 20])

    def test_all_zeros(self):
        assert gini_coefficient([0, 0, 0]) == 0.0

    def test_macro_metrics_uses_canonical(self):
        """Verify SocietyMacroMetrics.calculate_gini delegates to canonical impl."""
        from metrics.macro_metrics import SocietyMacroMetrics
        import numpy as np

        values = [10, 20, 30, 40, 50]
        canonical = gini_coefficient(values)
        macro = SocietyMacroMetrics.calculate_gini(np.array(values))
        assert abs(canonical - macro) < 0.001
