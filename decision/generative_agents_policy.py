"""Generative Agents proxy policy — Condition C.

Phase 21 — Comparison to Generative Agents Baseline.

Implements a fictional-persona LLM policy that mirrors the approach of
Park et al. 2023 (Generative Agents): the agent is given a rich, narrative
backstory, but that backstory is NOT anchored to empirical ESS distributions.

This creates Condition C for the three-condition comparison:
  - Condition A: pure LLM, no persona, no RAG  (AblatedLLMPolicy, no_persona)
  - Condition B: BGF grounded, ESS persona + RAG  (LLMPolicy, full)
  - Condition C: fictional persona, no RAG  (GenerativeAgentsPolicy)

The expected result: Condition C is closer to A than B, validating that it
is the *empirical grounding* (not just persona length) that reduces RLHF bias.
"""

from __future__ import annotations

import random
from typing import Optional

from decision.llm_backend import LLMBackend
from decision.llm_policy_base import LLMPolicyBase
from decision.prompt_builder import build_context_block, build_memory_block, build_state_block
from decision.schemas import ProposedAction
from decision.system_prompts import BASE_SYSTEM_PROMPT

# ── Fictional backstory templates ─────────────────────────────────────────────
# Each template is a plausible but purely fictional persona, not drawn from ESS.

_FICTIONAL_BACKSTORIES = [
    (
        "You are Alex, a 34-year-old freelance graphic designer living in a mid-size city. "
        "You grew up in a suburban neighborhood, attended a state university, and now rent a one-bedroom apartment. "
        "You value creative independence but sometimes worry about financial instability. "
        "You have a small but loyal circle of friends and generally try to be helpful to those you know."
    ),
    (
        "You are Jordan, a 47-year-old high school teacher in a rural town. "
        "You have a stable income but limited savings. You believe strongly in community and fairness. "
        "You coach the local soccer team on weekends and are well-liked by your neighbors. "
        "You tend to be cautious with money but generous with your time."
    ),
    (
        "You are Morgan, a 28-year-old software developer who recently moved to a large city for work. "
        "You are ambitious and career-focused, with a high income but also high expenses. "
        "You are somewhat introverted and prefer to keep professional and personal life separate. "
        "You are skeptical of institutions but trust people you know personally."
    ),
    (
        "You are Sam, a 55-year-old retired factory worker living on a pension. "
        "You have worked hard your whole life and are proud of your community. "
        "You are socially conservative but personally generous to neighbors in need. "
        "You worry about the future and tend to save rather than spend."
    ),
    (
        "You are Casey, a 22-year-old university student studying economics. "
        "You are curious, idealistic, and interested in social justice. "
        "You have little money but a strong social network. "
        "You believe cooperation is the foundation of a good society."
    ),
    (
        "You are Riley, a 41-year-old small business owner in an urban neighborhood. "
        "Your livelihood depends on the local community. You are pragmatic and relationship-oriented. "
        "You have moderate trust in others but have been burned before. "
        "You weigh financial decisions carefully against long-term reputation."
    ),
    (
        "You are Drew, a 63-year-old retired nurse with a modest pension and grown children. "
        "You have spent your career caring for others and believe strongly in mutual support. "
        "You are risk-averse and prefer security over opportunity. "
        "You have high trust in people but low trust in governments and corporations."
    ),
    (
        "You are Blake, a 31-year-old marketing professional with a competitive personality. "
        "You are financially comfortable and goal-oriented. "
        "You cooperate when it serves your interests but are not altruistic by default. "
        "You are ambitious, slightly impatient, and motivated by status."
    ),
]


def _sample_fictional_backstory(agent_id: str, seed: int | None = None) -> str:
    """Sample a fictional backstory deterministically from agent_id."""
    rng = random.Random(hash(agent_id) if seed is None else seed)
    return rng.choice(_FICTIONAL_BACKSTORIES)


class GenerativeAgentsPolicy(LLMPolicyBase):
    """Condition C: fictional-persona LLM policy with no ESS grounding.

    Mirrors the Park et al. 2023 setup: agents receive a narrative backstory,
    current state, and memory — but the backstory is fictional (not sampled
    from empirical distributions) and no RAG context is injected.

    This isolates whether it is the *empirical grounding* (not just having
    any persona at all) that drives Condition B's behavioral differences.
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
        backstory_seed: int | None = None,
    ):
        self.backend = backend
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.backstory_seed = backstory_seed
        # Cached backstory per agent_id to ensure consistency across rounds
        self._backstory_cache: dict[str, str] = {}

    def _get_backstory(self, agent_id: str) -> str:
        if agent_id not in self._backstory_cache:
            self._backstory_cache[agent_id] = _sample_fictional_backstory(
                agent_id, seed=self.backstory_seed
            )
        return self._backstory_cache[agent_id]

    def propose_action(
        self, profile, state, memory, context: dict, round_id: int
    ) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])

        # Build prompt: fictional backstory + state + memory + context (no RAG)
        backstory = self._get_backstory(profile.agent_id)
        state_block = build_state_block(state)
        memory_block = build_memory_block(memory, window=self.memory_window)
        context_block = build_context_block(context)

        user_content = (
            f"{backstory}\n\n"
            f"{state_block}\n\n"
            f"Recent memories:\n{memory_block}\n\n"
            f"Round {round_id}. {context_block}\n\n"
            "What do you do? Respond with ONLY the JSON."
        )

        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        action, raw_text, latency, parse_meta = self._generate_with_retries(messages, neighbors)

        if action is None:
            action = self._fallback_action(state, neighbors, profile=profile)
            parse_meta["fallback"] = True

        prompt_text = "\n".join(m["content"] for m in messages)
        self._log_prompt(
            round_id=round_id,
            agent_id=profile.agent_id,
            prompt_text=prompt_text,
            raw_text=raw_text,
            action=action,
            latency=latency,
            parse_meta={**parse_meta, "mode": "condition_c"},
        )

        return action
