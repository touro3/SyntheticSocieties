from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Literal

from decision.prompt_builder import build_memory_block, build_persona_block, build_state_block


@dataclass
class SocialAction:
    action_type: Literal["post", "comment", "react"]
    content: str | None = None
    target_id: str | None = None
    reaction: str | None = None
    reasoning_summary: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.reasoning_summary is None:
            if self.action_type == "react":
                self.reasoning_summary = f"Reacted with {self.reaction or 'like'}."
            elif self.content:
                self.reasoning_summary = self.content[:240]
            else:
                self.reasoning_summary = f"Chose social action: {self.action_type}."

    @property
    def target_agent_id(self) -> str | None:
        return self.target_id

    @property
    def amount(self) -> None:
        return None

    def model_dump(self) -> dict:
        data = asdict(self)
        data["target_agent_id"] = self.target_id
        return data


class SocialPolicy:
    """LLM-backed policy for social posting, commenting, and reactions."""

    _JSON_PATTERNS = (
        r'\{[^{}]*"action_type"[^{}]*\}',
        r'\{.*?"action_type".*?\}',
    )
    _VALID_ACTIONS = {"post", "comment", "react"}
    _VALID_REACTIONS = {"like", "upvote", "disagree"}

    def __init__(
        self,
        backend=None,
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
    ) -> None:
        self.backend = backend
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self._fallback_count = 0
        self._total_proposals = 0

    def propose_action(self, profile, state, memory, context: dict, round_id: int) -> SocialAction:
        self._total_proposals += 1
        messages = self._build_messages(profile, state, memory, context, round_id)
        raw_text = ""
        action = None

        if self.backend is not None:
            for _ in range(self.max_retries + 1):
                try:
                    raw_text = self._call_backend(messages)
                    action = self._parse_social_output(raw_text, context)
                    if action is not None:
                        break
                except Exception:
                    continue

        if action is None:
            action = self._fallback_action(profile, state, context)
            self._fallback_count += 1

        self._log_prompt(round_id, profile.agent_id, messages, raw_text, action)
        return action

    def get_fallback_rate(self) -> float:
        if self._total_proposals == 0:
            return 0.0
        return self._fallback_count / self._total_proposals

    def _build_messages(self, profile, state, memory, context: dict, round_id: int) -> list[dict]:
        social_context = self._build_social_context_block(context)
        user_parts = [
            f"Round {round_id}.",
            build_persona_block(profile),
            build_state_block(state),
            build_memory_block(memory, window=self.memory_window, profile=profile),
            social_context,
            (
                "Choose one social action. Respond with ONLY JSON matching: "
                '{"action_type":"post|comment|react","content":string|null,'
                '"target_id":string|null,"reaction":"like|upvote|disagree"|null,'
                '"reasoning_summary":string,"confidence":number}.'
            ),
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You are participating in a social simulation platform. "
                    "You may post, comment on a visible post, or react to a visible post."
                ),
            },
            {"role": "user", "content": "\n\n".join(part for part in user_parts if part)},
        ]

    def _call_backend(self, messages: list[dict]) -> str:
        try:
            result = self.backend.generate(messages=messages, temperature=self.temperature)
        except TypeError:
            result = self.backend.generate(messages, self.temperature)
        if isinstance(result, tuple):
            return str(result[0])
        return str(result)

    def _parse_social_output(self, raw_text: str, context: dict) -> SocialAction | None:
        if not raw_text or not raw_text.strip():
            return None

        candidates = [raw_text.strip()]
        for pattern in self._JSON_PATTERNS:
            candidates.extend(re.findall(pattern, raw_text, flags=re.DOTALL))

        for candidate in candidates:
            try:
                data = json.loads(self._repair_json(candidate))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            action = self._action_from_dict(data, context)
            if action is not None:
                return action

        return self._keyword_fallback(raw_text, context)

    def _action_from_dict(self, data: dict, context: dict) -> SocialAction | None:
        if not isinstance(data, dict):
            return None
        action_type = data.get("action_type")
        if action_type not in self._VALID_ACTIONS:
            return None

        target_id = data.get("target_id") or data.get("post_id")
        content = data.get("content")
        reaction = data.get("reaction")
        confidence = data.get("confidence")

        if action_type == "post":
            if not content:
                content = "Sharing an update from my current situation."
            return SocialAction(
                action_type="post",
                content=str(content)[:5000],
                reasoning_summary=str(data.get("reasoning_summary") or content)[:500],
                confidence=self._clamp_confidence(confidence),
            )

        if action_type == "comment":
            target_id = target_id or self._first_visible_post_id(context)
            if not target_id:
                return SocialAction(action_type="post", content=str(content or "Starting a new discussion."))
            return SocialAction(
                action_type="comment",
                content=str(content or "I have a response to this.")[:5000],
                target_id=str(target_id),
                reasoning_summary=str(data.get("reasoning_summary") or content or "Commented on a post.")[:500],
                confidence=self._clamp_confidence(confidence),
            )

        target_id = target_id or self._first_visible_post_id(context)
        if not target_id:
            return SocialAction(action_type="post", content="Starting a new discussion.")
        reaction = reaction if reaction in self._VALID_REACTIONS else "like"
        return SocialAction(
            action_type="react",
            target_id=str(target_id),
            reaction=str(reaction),
            reasoning_summary=str(data.get("reasoning_summary") or f"Reacted with {reaction}.")[:500],
            confidence=self._clamp_confidence(confidence),
        )

    def _keyword_fallback(self, text: str, context: dict) -> SocialAction | None:
        lower = text.lower()
        target_id = self._first_visible_post_id(context)
        if target_id and re.search(r"\b(disagree|oppose|skeptical)\b", lower):
            return SocialAction(action_type="react", target_id=target_id, reaction="disagree")
        if target_id and re.search(r"\b(comment|reply|respond)\b", lower):
            return SocialAction(action_type="comment", target_id=target_id, content=text.strip()[:280])
        if target_id and re.search(r"\b(like|upvote|agree)\b", lower):
            return SocialAction(action_type="react", target_id=target_id, reaction="like")
        if text.strip():
            return SocialAction(action_type="post", content=text.strip()[:280])
        return None

    def _fallback_action(self, profile, state, context: dict) -> SocialAction:
        feed_post_id = self._first_visible_post_id(context)
        social_activity = getattr(profile, "social_activity", None)
        if feed_post_id and (social_activity is None or social_activity < 0.65):
            return SocialAction(action_type="react", target_id=feed_post_id, reaction="like", confidence=0.5)
        if feed_post_id:
            return SocialAction(
                action_type="comment",
                target_id=feed_post_id,
                content="This seems important for the community.",
                confidence=0.5,
            )
        stress = getattr(state, "stress", 0.0)
        content = (
            "I am feeling pressure and watching how others respond."
            if stress >= 0.7
            else "Sharing a brief update with the community."
        )
        return SocialAction(action_type="post", content=content, confidence=0.5)

    def _log_prompt(
        self,
        round_id: int,
        agent_id: str,
        messages: list[dict],
        raw_text: str,
        action: SocialAction,
    ) -> None:
        if self.prompt_logger is None:
            return
        prompt_text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
        self.prompt_logger.log(
            round_id=round_id,
            agent_id=agent_id,
            prompt=prompt_text,
            raw_output=raw_text,
            parsed_action=action.model_dump(),
            latency_ms=0.0,
            parse_metadata={"policy": "social"},
        )

    @staticmethod
    def _build_social_context_block(context: dict) -> str:
        world = context.get("world", context)
        network = context.get("network", {})
        feed = world.get("feed", [])
        trending = world.get("trending", [])
        neighbors = network.get("neighbors", world.get("neighbors", []))

        lines = [f"Social platform: {world.get('platform', 'short_form')}."]
        if neighbors:
            lines.append(f"Network neighbors: {', '.join(neighbors)}.")
        if feed:
            lines.append("Visible feed:")
            for post in feed[:5]:
                lines.append(
                    f"  {post.get('post_id')} by {post.get('author_id')}: {str(post.get('content', ''))[:180]}"
                )
        else:
            lines.append("Visible feed is empty.")
        if trending:
            lines.append("Trending posts: " + ", ".join(str(post.get("post_id")) for post in trending[:5]) + ".")
        return "\n".join(lines)

    @staticmethod
    def _first_visible_post_id(context: dict) -> str | None:
        world = context.get("world", context)
        feed = world.get("feed", [])
        if not feed:
            return None
        return str(feed[0].get("post_id"))

    @staticmethod
    def _repair_json(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        open_braces = text.count("{") - text.count("}")
        if open_braces > 0:
            text += "}" * open_braces
        return text

    @staticmethod
    def _clamp_confidence(value) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None
