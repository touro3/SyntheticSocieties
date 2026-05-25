"""Property-based tests for BGF's headline metrics.

These tests use `hypothesis` to generate adversarial inputs and assert
the algebraic / range invariants that the formal-framework section of
the paper (§3.2) relies on. They complement the example-based tests
in `tests/test_distribution_metrics.py` and friends, which fix specific
input values and check specific outputs.

Each invariant maps to a statement in `docs/paper.md`:

- BRM ∈ [0, 1]                              (§3.2.2)
- B_RLHF ∈ [0, 2/3] for |A| = 3             (§3.2.1, Proposition 1)
- B_RLHF(π_uniform) = 0                     (§3.2.1)
- B_RLHF reaches 2/3 only at a one-hot π    (§3.2.1)
- Gini ∈ [0, 1]                             (standard)
- Gini(constant) = 0                        (standard)
- Gini(one-rich) → 1 as N → ∞              (standard)
- compute_brm_jsd is symmetric in its args  (audit A2.1 fix consequence)
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from metrics.behavioral_realism import (
    _ACTIONS,
    compute_brm_jsd,
    compute_rlhf_bias_index,
)
from metrics.inequality import gini_coefficient

# ── B_RLHF ────────────────────────────────────────────────────────────────────


@st.composite
def action_distributions(draw, allow_extra: bool = False) -> dict[str, float]:
    """A valid probability distribution over the canonical action space.

    With ``allow_extra``, also draws a probability mass on `steal` to
    confirm the helper handles non-canonical actions per its docstring.
    """
    masses = draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3))
    if allow_extra:
        masses.append(draw(st.floats(min_value=0.0, max_value=0.5)))
    total = sum(masses)
    if total == 0:
        masses = [1.0, 0.0] + [0.0] * (len(masses) - 2)
        total = 1.0
    keys = list(_ACTIONS) + (["steal"] if allow_extra else [])
    return dict(zip(keys, [m / total for m in masses]))


@given(action_distributions())
@settings(deadline=None, max_examples=200)
def test_b_rlhf_is_bounded_in_zero_two_thirds(dist: dict[str, float]) -> None:
    """Proposition 1, §3.2.1: B_RLHF ∈ [0, 2/3] for the 3-action space."""
    b = compute_rlhf_bias_index(dist)
    assert 0.0 <= b <= 2.0 / 3.0 + 1e-12, (dist, b)


def test_b_rlhf_zero_at_uniform() -> None:
    """B_RLHF(π_uniform) = 0 by definition of TV distance."""
    uniform = dict.fromkeys(_ACTIONS, 1.0 / len(_ACTIONS))
    assert compute_rlhf_bias_index(uniform) == pytest.approx(0.0, abs=1e-12)


@given(st.sampled_from(list(_ACTIONS)))
def test_b_rlhf_max_at_one_hot(action: str) -> None:
    """B_RLHF reaches 2/3 only when probability is on a single action."""
    one_hot = {a: (1.0 if a == action else 0.0) for a in _ACTIONS}
    assert compute_rlhf_bias_index(one_hot) == pytest.approx(2.0 / 3.0, abs=1e-12)


@given(action_distributions(allow_extra=True))
@settings(deadline=None, max_examples=100)
def test_b_rlhf_handles_steal_without_error(dist: dict[str, float]) -> None:
    """Non-canonical actions (`steal`) are documented to be ignored with a
    warning; computation must not raise. See module docstring."""
    b = compute_rlhf_bias_index(dist)
    assert 0.0 <= b <= 2.0 / 3.0 + 1e-12


# ── BRM-JSD ───────────────────────────────────────────────────────────────────


@st.composite
def wealth_vectors(draw, *, min_size: int = 5, max_size: int = 200) -> list[float]:
    """Positive wealth values with non-trivial variance."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )


@given(wealth_vectors(), wealth_vectors())
@settings(deadline=None, max_examples=100)
def test_brm_in_unit_interval(sim: list[float], emp: list[float]) -> None:
    """BRM ∈ [0, 1] (§3.2.2). Pre-A2.1 bug compressed this to [0.307, 1]."""
    brm = compute_brm_jsd(sim, emp, bins=30)
    assert 0.0 <= brm <= 1.0


@given(wealth_vectors())
@settings(deadline=None, max_examples=50)
def test_brm_self_comparison_is_one(values: list[float]) -> None:
    """BRM(x, x) = 1 — a distribution is maximally realistic relative to itself."""
    brm = compute_brm_jsd(values, values, bins=30)
    assert brm == pytest.approx(1.0, abs=1e-9)


@given(wealth_vectors(), wealth_vectors())
@settings(deadline=None, max_examples=50)
def test_brm_is_symmetric(a: list[float], b: list[float]) -> None:
    """JSD is symmetric; therefore so is BRM_JSD = 1 - JSD."""
    assert compute_brm_jsd(a, b, bins=30) == pytest.approx(
        compute_brm_jsd(b, a, bins=30), abs=1e-9
    )


# ── Gini ──────────────────────────────────────────────────────────────────────


@given(wealth_vectors())
@settings(deadline=None, max_examples=200)
def test_gini_in_unit_interval(values: list[float]) -> None:
    g = gini_coefficient(values)
    assert 0.0 <= g <= 1.0 + 1e-9


@given(st.floats(min_value=1.0, max_value=1e6), st.integers(min_value=2, max_value=500))
def test_gini_constant_is_zero(v: float, n: int) -> None:
    """A perfectly equal population has Gini = 0."""
    g = gini_coefficient([v] * n)
    assert g == pytest.approx(0.0, abs=1e-12)


@given(st.integers(min_value=50, max_value=1000))
def test_gini_one_rich_approaches_one(n: int) -> None:
    """One agent holding all wealth approaches Gini = 1 as N grows.

    The exact bound is `(N-1)/N` for a population with N-1 zeros and a
    single non-zero value (standard textbook result)."""
    values = [0.0] * (n - 1) + [1.0]
    g = gini_coefficient(values)
    expected_upper = (n - 1) / n
    assert g == pytest.approx(expected_upper, abs=1e-9)


# ── Histogram-edge cases ──────────────────────────────────────────────────────


def test_brm_handles_zero_variance_emp() -> None:
    """Empirical reference is a constant array — histogram has one filled
    bin. The metric should not crash and should remain bounded."""
    brm = compute_brm_jsd([1, 2, 3, 4, 5], [3, 3, 3, 3, 3], bins=10)
    assert 0.0 <= brm <= 1.0
    assert math.isfinite(brm)


def test_brm_handles_single_sample() -> None:
    """Degenerate single-element input — bounded, finite."""
    brm = compute_brm_jsd([1.0], [2.0], bins=2)
    assert 0.0 <= brm <= 1.0
    assert math.isfinite(brm)
