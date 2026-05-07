from __future__ import annotations

import types
from dataclasses import asdict, dataclass, field
from typing import Literal

from environment.institutions import ValidationResult
from environment.world_state import WorldState


@dataclass
class Post:
    post_id: str
    author_id: str
    content: str
    platform: Literal["short_form", "long_form"]
    parent_id: str | None
    round_id: int
    reactions: dict[str, int] = field(default_factory=dict)

    def model_dump(self) -> dict:
        return asdict(self)


class SocialEnvironment:
    """Drop-in companion to World for social simulation rounds."""

    VALID_REACTIONS = {"like", "upvote", "disagree"}
    MAX_SHORT_FORM_CHARS = 280
    MAX_LONG_FORM_CHARS = 5000

    def __init__(
        self,
        state: WorldState | None = None,
        network_manager=None,
        platform: Literal["short_form", "long_form"] = "short_form",
    ) -> None:
        self.state = state or WorldState()
        self.network_manager = network_manager
        self.platform = platform
        self.posts: dict[str, Post] = {}
        self._timeline: list[str] = []
        self._reaction_ledger: set[tuple[str, str, str]] = set()
        # Monotonic counter — never reuses IDs even after apply_round_decay.
        # Using len(self.posts)+1 caused ID collisions after deletions, leaving
        # stale IDs in _timeline that produced KeyError in get_trending().
        self._post_counter: int = 0

    def submit_post(self, agent_id: str, content: str, parent_id: str | None = None) -> Post:
        if parent_id is not None and parent_id not in self.posts:
            raise ValueError(f"Unknown parent post: {parent_id}")
        self._post_counter += 1
        post_id = f"post_{self._post_counter}"
        post = Post(
            post_id=post_id,
            author_id=agent_id,
            content=content.strip(),
            platform=self.platform,
            parent_id=parent_id,
            round_id=self.state.round_id,
            reactions={},
        )
        self.posts[post_id] = post
        self._timeline.append(post_id)
        return post

    def react(self, agent_id: str, post_id: str, reaction: str) -> None:
        if post_id not in self.posts:
            raise ValueError(f"Unknown post: {post_id}")
        if reaction not in self.VALID_REACTIONS:
            raise ValueError(f"Invalid reaction: {reaction}")
        key = (agent_id, post_id, reaction)
        if key in self._reaction_ledger:
            return
        self._reaction_ledger.add(key)
        post = self.posts[post_id]
        post.reactions[reaction] = post.reactions.get(reaction, 0) + 1

    def get_feed(self, agent_id: str, n: int = 10) -> list[Post]:
        neighbors = set(self._neighbors(agent_id))
        indexed = [
            (idx, self.posts[pid])
            for idx, pid in enumerate(self._timeline)
            if pid in self.posts  # guard against stale _timeline entries
        ]

        def _score(item: tuple[int, Post]) -> tuple[float, int]:
            idx, post = item
            neighbor_bonus = 1000.0 if post.author_id in neighbors else 0.0
            reaction_bonus = 0.25 * sum(post.reactions.values())
            reply_bonus = 0.5 if post.parent_id else 0.0
            return (idx + neighbor_bonus + reaction_bonus + reply_bonus, idx)

        ranked = sorted(indexed, key=_score, reverse=True)
        return [post for _, post in ranked[: max(0, n)]]

    def get_trending(self, top_n: int = 5) -> list[Post]:
        ranked = sorted(
            self.posts.values(),
            key=lambda post: (sum(post.reactions.values()), post.round_id, post.post_id),
            reverse=True,
        )
        return ranked[: max(0, top_n)]

    def get_agent_context(self, agent_id: str) -> dict:
        return {
            "platform": self.platform,
            "round_id": self.state.round_id,
            "feed": [post.model_dump() for post in self.get_feed(agent_id)],
            "trending": [post.model_dump() for post in self.get_trending()],
            "neighbors": self._neighbors(agent_id),
            "public_signal": self.state.public_signal,
        }

    def validate_action(self, action, agent, agent_lookup) -> ValidationResult:
        action_type = getattr(action, "action_type", None)
        content = (getattr(action, "content", None) or "").strip()
        target_id = getattr(action, "target_id", getattr(action, "target_agent_id", None))
        reaction = getattr(action, "reaction", None)

        if action_type not in {"post", "comment", "react"}:
            return ValidationResult(valid=False, reason="invalid_social_action")

        if agent.profile.agent_id not in agent_lookup:
            return ValidationResult(valid=False, reason="unknown_actor")

        if action_type == "post":
            return self._validate_content(content)

        if action_type == "comment":
            if not target_id or target_id not in self.posts:
                return ValidationResult(valid=False, reason="unknown_target_post")
            content_result = self._validate_content(content)
            if not content_result.valid:
                return content_result
            return ValidationResult(valid=True)

        if not target_id or target_id not in self.posts:
            return ValidationResult(valid=False, reason="unknown_target_post")
        if reaction not in self.VALID_REACTIONS:
            return ValidationResult(valid=False, reason="invalid_reaction")
        return ValidationResult(valid=True)

    def execute_action(self, action, agent, agent_lookup) -> dict:  # noqa: ARG002
        agent_id = agent.profile.agent_id
        action_type = getattr(action, "action_type")
        content = (getattr(action, "content", None) or "").strip()
        target_id = getattr(action, "target_id", getattr(action, "target_agent_id", None))
        reaction = getattr(action, "reaction", None)

        event = {
            "agent_id": agent_id,
            "action_type": action_type,
            "target_agent_id": None,
            "target_id": target_id,
            "wealth_delta": 0.0,
            "stress_delta": 0.0,
            "satisfaction_delta": 0.03,
            "target_wealth_delta": 0.0,
            "interaction_type": "social",
            "round_id": self.state.round_id,
        }

        if action_type == "post":
            post = self.submit_post(agent_id, content)
            event["post_id"] = post.post_id
            event["content"] = post.content
            event["interaction_type"] = "social_post"
        elif action_type == "comment":
            post = self.submit_post(agent_id, content, parent_id=target_id)
            event["post_id"] = post.post_id
            event["parent_id"] = target_id
            event["content"] = post.content
            event["interaction_type"] = "social_comment"
            event["satisfaction_delta"] = 0.04
        elif action_type == "react":
            self.react(agent_id, target_id, reaction)
            event["post_id"] = target_id
            event["reaction"] = reaction
            event["interaction_type"] = "social_reaction"
            event["satisfaction_delta"] = 0.02

        return event

    def apply_exogenous_updates(self) -> list[dict]:
        self.state.round_id += 1
        applied = list(getattr(self.state, "pending_injections", []))
        self.state.pending_injections.clear()
        for event in applied:
            if event.get("event_type") == "narrative":
                payload = event.get("payload", {})
                content = str(payload.get("content", payload.get("message", ""))).strip()
                if content:
                    self.state.public_signal["narrative"] = content
            elif event.get("event_type") == "signal_update":
                payload = event.get("payload", {})
                signal = payload.get("signal", payload)
                if isinstance(signal, dict):
                    self.state.public_signal.update({str(k): str(v) for k, v in signal.items()})
        return applied

    def apply_round_decay(self, current_round: int, max_post_age: int = 10) -> None:
        """Remove posts older than max_post_age rounds to keep the feed fresh."""
        cutoff = current_round - max_post_age
        stale = [pid for pid, post in self.posts.items() if post.round_id < cutoff]
        for pid in stale:
            self.posts.pop(pid, None)
            # Remove ALL occurrences: old len()-based ID generation could insert
            # duplicates; a single remove() would leave a stale entry that causes
            # KeyError in get_feed(). The monotonic counter prevents new duplicates,
            # but purging all is the safe contract regardless.
            self._timeline = [t for t in self._timeline if t != pid]

    def step(self, agents: list, round_id: int) -> list[dict]:
        """Generate and execute rule-based social actions for all agents.

        Cooperating agents post; others react to trending posts (or post if
        nothing is trending yet). Returns a list of executed event dicts.
        """
        events = []
        agent_lookup = {a.profile.agent_id: a for a in agents}
        trending = self.get_trending(top_n=3)

        for agent in agents:
            last_action = agent.state.last_action or "work"

            if last_action == "cooperate" or not trending:
                content = f"Round {round_id}: community action — {last_action}."
                content = content[: self.MAX_SHORT_FORM_CHARS]
                action = types.SimpleNamespace(action_type="post", content=content, target_id=None, reaction=None)
            else:
                target_post = trending[0]
                reaction = "like" if last_action in ("work", "save") else "disagree"
                action = types.SimpleNamespace(
                    action_type="react",
                    content="",
                    target_id=target_post.post_id,
                    reaction=reaction,
                )

            validation = self.validate_action(action, agent, agent_lookup)
            if validation.valid:
                event = self.execute_action(action, agent, agent_lookup)
                events.append(event)

        return events

    def get_stats(self) -> dict:
        """Return aggregate counts suitable for per-round metrics."""
        posts = [p for p in self.posts.values() if p.parent_id is None]
        comments = [p for p in self.posts.values() if p.parent_id is not None]
        total_reactions = sum(sum(p.reactions.values()) for p in self.posts.values())
        reaction_types = {r: 0 for r in self.VALID_REACTIONS}
        for _, _, r in self._reaction_ledger:
            if r in reaction_types:
                reaction_types[r] += 1
        return {
            "total_posts": len(posts),
            "total_comments": len(comments),
            "total_reactions": total_reactions,
            "reaction_types": reaction_types,
        }

    def _neighbors(self, agent_id: str) -> list[str]:
        if self.network_manager is None:
            return []
        return list(self.network_manager.get_neighbors(agent_id))

    def _validate_content(self, content: str) -> ValidationResult:
        if not content:
            return ValidationResult(valid=False, reason="empty_content")
        max_chars = self.MAX_SHORT_FORM_CHARS if self.platform == "short_form" else self.MAX_LONG_FORM_CHARS
        if len(content) > max_chars:
            return ValidationResult(valid=False, reason="content_too_long")
        return ValidationResult(valid=True)
