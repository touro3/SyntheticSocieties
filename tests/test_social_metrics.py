"""Tests for metrics/social_metrics.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from environment.social_env import SocialEnvironment
from environment.world_state import WorldState
from metrics.social_metrics import engagement_rate, network_amplification, post_diversity


def _env_with_posts() -> SocialEnvironment:
    """Return a SocialEnvironment with a few posts and reactions."""
    env = SocialEnvironment(state=WorldState())
    p1 = env.submit_post("agent_0", "Cooperation worked well this round.")
    p2 = env.submit_post("agent_1", "I chose to save my resources.")
    env.react("agent_2", p1.post_id, "like")
    env.react("agent_3", p1.post_id, "upvote")
    env.submit_post("agent_2", "reply to agent_0", parent_id=p1.post_id)
    return env


def test_social_metrics_computes_engagement_rate():
    env = _env_with_posts()

    rate = engagement_rate(env)

    # p1 has 2 reactions → engaged; p2 has none → not engaged
    # 2 top-level posts, 1 engaged → 0.5
    assert rate == pytest.approx(0.5)


def test_post_diversity_is_positive_when_varied():
    env = _env_with_posts()

    diversity = post_diversity(env)

    # We have posts, a comment, and reactions — entropy should be > 0
    assert diversity > 0.0


def test_network_amplification_equals_average_reactions_per_post():
    env = _env_with_posts()

    amp = network_amplification(env)

    # 2 top-level posts: p1 has 2 reactions, p2 has 0 → mean = 1.0
    assert amp == pytest.approx(1.0)
