from __future__ import annotations
import time 

from typing import Optional

from decision.llm_backend import LLMBackend
from decision.llm_policy_base import LLMPolicyBase
from decision.prompt_builder import build_prompt, build_prompt_text
from decision.schemas import ProposedAction


class LLMPolicy(LLMPolicyBase):
    """Policy that uses an LLM to decide agent actions."""

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
        perturbation_mode: Optional[str] = None,
        graph_rag=None,
        sql_rag=None,
        ablation_level: int = 5,
    ):
        self.backend = backend
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.perturbation_mode = perturbation_mode
        self.graph_rag = graph_rag
        self.sql_rag = sql_rag
        self.ablation_level = ablation_level

    def propose_action(
        self, profile, state, memory, context: dict, round_id: int,
    ) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])

        # Fetch RAG contexts
        social_context = None
        if self.graph_rag:
            social_context = self.graph_rag.get_social_context(profile.agent_id)

        pop_context = None
        if self.sql_rag:
            pop_context = self.sql_rag.get_peer_group_context(
                age=profile.age, gender=profile.gender, country=profile.country
            )

        # Build prompt
        messages = build_prompt(
            profile=profile, state=state, memory=memory, context=context,
            round_id=round_id, memory_window=self.memory_window,
            social_context=social_context, population_context=pop_context,
            ablation_level=self.ablation_level,
        )

        # Apply perturbation if configured
        if self.perturbation_mode:
            from decision.prompt_perturbation import apply_perturbation
            seed = hash((round_id, profile.agent_id)) % (2**31)
            messages = apply_perturbation(messages, mode=self.perturbation_mode, seed=seed)

        # Generate with retries (shared logic from LLMPolicyBase)
        action, raw_text, latency, parse_meta = self._generate_with_retries(messages, neighbors)

        if action is None:
            action = self._fallback_action(state, neighbors)
            parse_meta["fallback"] = True

        # Log
        prompt_text = build_prompt_text(
            profile, state, memory, context, round_id, self.memory_window,
            social_context=social_context, population_context=pop_context,
            ablation_level=self.ablation_level,
        )
        self._log_prompt(
            round_id=round_id, agent_id=profile.agent_id,
            prompt_text=prompt_text, raw_text=raw_text,
            action=action, latency=latency, parse_meta=parse_meta,
        )

        return action
