"""Tests for AgentProfile field validation."""
import pytest
from agents.profile import AgentProfile


def _defaults(**overrides):
    base = dict(
        agent_id="a0", age=35, income=1000.0, education="college",
        occupation="worker", location="urban", political_preference="center",
        risk_tolerance=0.5, social_class="middle",
    )
    base.update(overrides)
    return base


class TestNormalizedFieldValidation:
    @pytest.mark.parametrize("field_name", [
        "trust_people", "trust_institutions", "political_orientation",
        "life_satisfaction", "happiness", "immigration_attitude",
        "social_activity", "competitiveness", "leadership_preference",
        "health_status", "religiosity",
    ])
    def test_rejects_value_above_one(self, field_name):
        with pytest.raises(ValueError):
            AgentProfile(**_defaults(**{field_name: 1.5}))

    @pytest.mark.parametrize("field_name", [
        "trust_people", "trust_institutions", "political_orientation",
        "life_satisfaction", "happiness", "immigration_attitude",
        "social_activity", "competitiveness", "leadership_preference",
        "health_status", "religiosity",
    ])
    def test_rejects_value_below_zero(self, field_name):
        with pytest.raises(ValueError):
            AgentProfile(**_defaults(**{field_name: -0.1}))

    @pytest.mark.parametrize("field_name", [
        "trust_people", "trust_institutions",
    ])
    def test_accepts_valid_normalized(self, field_name):
        p = AgentProfile(**_defaults(**{field_name: 0.7}))
        assert getattr(p, field_name) == 0.7

    def test_none_values_accepted(self):
        """All ESS fields default to None and that's valid."""
        p = AgentProfile(**_defaults())
        assert p.trust_people is None


class TestGenderValidation:
    def test_rejects_invalid_gender(self):
        with pytest.raises(ValueError):
            AgentProfile(**_defaults(gender=3))

    def test_accepts_male(self):
        p = AgentProfile(**_defaults(gender=1))
        assert p.gender == 1

    def test_accepts_female(self):
        p = AgentProfile(**_defaults(gender=2))
        assert p.gender == 2

    def test_accepts_none(self):
        p = AgentProfile(**_defaults())
        assert p.gender is None


class TestEducationLevelValidation:
    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            AgentProfile(**_defaults(education_level=10))

    def test_accepts_valid(self):
        p = AgentProfile(**_defaults(education_level=5))
        assert p.education_level == 5
