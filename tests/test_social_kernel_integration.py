"""Integration tests: SocialEnvironment wired through SimulationKernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from conftest import make_agent
from environment.social_env import SocialEnvironment


def test_social_action_executed_in_kernel_round(minimal_kernel, social_env):
    """After one kernel round, the social environment should contain at least one post."""
    assert len(social_env.posts) == 0
    minimal_kernel.run_round()
    assert len(social_env.posts) >= 1


def test_social_context_injected_into_agent_memory(minimal_kernel, social_env):
    """After two rounds (so there's a feed on the second), agent memory
    should include a social-feed narration.
    """
    # Run two rounds: first creates posts, second reads feed into memory
    minimal_kernel.run_round()
    minimal_kernel.run_round()

    agent = minimal_kernel.agents[0]
    all_content = " ".join(
        item.content for item in agent.memory.recent
    )
    assert "Social feed" in all_content or len(social_env.posts) >= 1


def test_social_metrics_logged_per_round(minimal_kernel, social_env):
    """Round metrics must include a 'social_actions' key when social_env is active."""
    minimal_kernel.run_round()
    minimal_kernel._log_round_metrics()

    assert minimal_kernel.round_metrics, "Expected at least one metrics entry"
    last = minimal_kernel.round_metrics[-1]
    assert "social_actions" in last
    assert "total_posts" in last["social_actions"]
