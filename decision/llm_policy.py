"""
LLM-based decision policy for agent action selection.

Ties together prompt building, LLM inference, and output parsing
into the standard policy interface.
"""

from __future__ import annotations

import warnings
from typing import Optional

from decision.llm_backend import LLMBackend
from decision.output_parser import parse_llm_output
from decision.prompt_builder import build_prompt, build_prompt_text
from decision.schemas import ProposedAction


class LLMPolicy:
    """
    Policy that uses an LLM to decide agent actions.

    Implements the same propose_action() interface as MockPolicy,
    RandomPolicy, RuleBasedPolicy, and DataDrivenPolicy.
    """

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
    ):
        self.backend = backend
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger

    def propose_action(
        self,
        profile,
        state,
        memory,
        context: dict,
        round_id: int,
    ) -> ProposedAction:
        """
        Use the LLM to propose an action for the agent.

        Flow:
          1. Build persona-conditioned prompt
          2. Send to LLM for generation
          3. Parse structured output
          4. Fallback to rule-based if parsing fails
          5. Log prompt + output
        """
        neighbors = context.get("network", {}).get("neighbors", [])

        # Build prompt
        messages = build_prompt(
            profile=profile,
            state=state,
            memory=memory,
            context=context,
            round_id=round_id,
            memory_window=self.memory_window,
        )

        # Try generation with retries
        action = None
        raw_text = ""
        latency = 0.0
        parse_meta = {}

        for attempt in range(self.max_retries + 1):
            try:
                raw_text, latency = self.backend.generate(
                    messages=messages,
                    temperature=self.temperature,
                )

                action, parse_meta = parse_llm_output(raw_text, neighbors)

                if action is not None:
                    break

            except Exception as e:
                warnings.warn(f"LLM generation failed (attempt {attempt+1}): {e}")
                parse_meta = {"parse_error": str(e), "parse_success": False}

        # Fallback if all attempts failed
        if action is None:
            action = self._fallback_action(state, neighbors)
            parse_meta["fallback"] = True

        # Log prompt + output
        if self.prompt_logger is not None:
            prompt_text = build_prompt_text(
                profile, state, memory, context, round_id, self.memory_window
            )
            self.prompt_logger.log(
                round_id=round_id,
                agent_id=profile.agent_id,
                prompt=prompt_text,
                raw_output=raw_text,
                parsed_action=action.model_dump() if action else None,
                latency_ms=latency * 1000,
                parse_metadata=parse_meta,
            )

        return action

    def _fallback_action(self, state, neighbors: list[str]) -> ProposedAction:
        """Rule-based fallback when LLM fails."""
        if state.wealth < 70:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary="[LLM fallback: working due to low wealth]",
                confidence=0.5,
            )

        if neighbors and state.wealth >= 100:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=5.0,
                reasoning_summary="[LLM fallback: cooperating with high wealth]",
                confidence=0.5,
            )

        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="[LLM fallback: saving as default]",
            confidence=0.5,
        )
