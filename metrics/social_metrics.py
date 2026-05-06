from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.social_env import SocialEnvironment


def engagement_rate(social_env: "SocialEnvironment") -> float:
    """Fraction of posts that received at least one reaction."""
    posts = [p for p in social_env.posts.values() if p.parent_id is None]
    if not posts:
        return 0.0
    engaged = sum(1 for p in posts if sum(p.reactions.values()) > 0)
    return engaged / len(posts)


def post_diversity(social_env: "SocialEnvironment") -> float:
    """Shannon entropy over action types (post / comment / react).

    Higher values indicate more varied social behaviour.
    """
    counts: dict[str, int] = {"post": 0, "comment": 0, "react": 0}
    for post in social_env.posts.values():
        if post.parent_id is None:
            counts["post"] += 1
        else:
            counts["comment"] += 1
    counts["react"] = len(social_env._reaction_ledger)

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def network_amplification(social_env: "SocialEnvironment") -> float:
    """Average reactions per post (measures how content spreads through the network)."""
    posts = [p for p in social_env.posts.values() if p.parent_id is None]
    if not posts:
        return 0.0
    total_reactions = sum(sum(p.reactions.values()) for p in posts)
    return round(total_reactions / len(posts), 4)
