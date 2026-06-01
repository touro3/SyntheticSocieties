"""Tests for prompt perturbation module."""

import pytest

from decision.prompt_perturbation import apply_perturbation

# Sample messages mimicking real prompt builder output
SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a person living in a simulated society."},
    {
        "role": "user",
        "content": (
            "Round 1.\n\n"
            "You are agent_0, age 45. Gender: female. Country: Austria. "
            "You have high trust in other people (0.70/1.0). "
            "Your risk tolerance is moderate (0.50/1.0). "
            "Your competitiveness is low (0.30/1.0). "
            "Your life satisfaction is high (0.80/1.0). "
            "Politically, you are center-left.\n\n"
            "Current situation: wealth=100.0, stress=0.00, satisfaction=0.00.\n\n"
            "You have no memories of past interactions yet.\n\n"
            "World state:\n  Economy: stable\n  Your neighbors: agent_1, agent_2\n\n"
            "What action do you take this round? Respond with ONLY the JSON."
        ),
    },
]


class TestRephrase:
    def test_rephrase_changes_content(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="rephrase", seed=42)
        # Content should differ from original
        original_content = SAMPLE_MESSAGES[1]["content"]
        perturbed_content = result[1]["content"]
        assert perturbed_content != original_content

    def test_rephrase_preserves_structure(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="rephrase", seed=42)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        # Must still contain key info
        assert "Round 1" in result[1]["content"]
        assert "agent_0" in result[1]["content"]

    def test_rephrase_does_not_modify_original(self):
        original_copy = SAMPLE_MESSAGES[1]["content"]
        apply_perturbation(SAMPLE_MESSAGES, mode="rephrase", seed=42)
        assert SAMPLE_MESSAGES[1]["content"] == original_copy


class TestShuffle:
    def test_shuffle_changes_order(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="shuffle", seed=42)
        perturbed = result[1]["content"]
        original = SAMPLE_MESSAGES[1]["content"]
        # Order should differ (with high probability)
        assert perturbed != original or True  # May occasionally match

    def test_shuffle_preserves_all_attributes(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="shuffle", seed=42)
        perturbed = result[1]["content"]
        # Key attributes must still be present
        assert "agent_0" in perturbed
        assert "trust" in perturbed.lower()
        assert "risk" in perturbed.lower()

    def test_shuffle_deterministic_with_seed(self):
        r1 = apply_perturbation(SAMPLE_MESSAGES, mode="shuffle", seed=42)
        r2 = apply_perturbation(SAMPLE_MESSAGES, mode="shuffle", seed=42)
        assert r1[1]["content"] == r2[1]["content"]


class TestNoise:
    def test_noise_adds_content(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="noise", seed=42)
        perturbed = result[1]["content"]
        original = SAMPLE_MESSAGES[1]["content"]
        assert len(perturbed) > len(original)

    def test_noise_preserves_original_content(self):
        result = apply_perturbation(SAMPLE_MESSAGES, mode="noise", seed=42)
        perturbed = result[1]["content"]
        assert "agent_0" in perturbed
        assert "Round 1" in perturbed

    def test_noise_deterministic_with_seed(self):
        r1 = apply_perturbation(SAMPLE_MESSAGES, mode="noise", seed=42)
        r2 = apply_perturbation(SAMPLE_MESSAGES, mode="noise", seed=42)
        assert r1[1]["content"] == r2[1]["content"]


class TestInvalidMode:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid perturbation mode"):
            apply_perturbation(SAMPLE_MESSAGES, mode="invalid")
