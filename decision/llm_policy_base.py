"""Shared retry, fallback, and logging logic for all LLM-based policies.

Eliminates the 3x code duplication across LLMPolicy, AblatedLLMPolicy,
and ConditionedLLMPolicy. Subclasses override only prompt construction.
"""

from __future__ import annotations

import warnings
from typing import Optional

from decision.output_parser import parse_llm_output
from decision.schemas import ProposedAction


class LLMPolicyBase:
    """Base class for LLM policies with shared infrastructure."""

    # Subclasses must set these in __init__:
    backend: object
    temperature: float
    max_retries: int
    prompt_logger: object | None

    def _generate_with_retries(
        self,
        messages: list[dict],
        neighbors: list[str],
    ) -> tuple[ProposedAction | None, str, float, dict]:
        """Generate LLM output with retries. Returns (action, raw_text, latency, parse_meta)."""
        action = None
        raw_text = ""
        latency = 0.0
        parse_meta: dict = {}

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
                warnings.warn(f"LLM generation failed (attempt {attempt + 1}): {e}")
                parse_meta = {"parse_error": str(e), "parse_success": False}

        return action, raw_text, latency, parse_meta

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

    def _log_prompt(
        self,
        round_id: int,
        agent_id: str,
        prompt_text: str,
        raw_text: str,
        action: ProposedAction | None,
        latency: float,
        parse_meta: dict,
        extra_meta: dict | None = None,
    ) -> None:
        """Log prompt + output if a prompt_logger is configured."""
        if self.prompt_logger is None:
            return
        meta = {**parse_meta}
        rag_context = None
        if extra_meta:
            # rag_context is a structured audit field — keep it separate from parse_metadata
            rag_context = extra_meta.pop("rag_context", None)
            meta.update(extra_meta)
        self.prompt_logger.log(
            round_id=round_id,
            agent_id=agent_id,
            prompt=prompt_text,
            raw_output=raw_text,
            parsed_action=action.model_dump() if action else None,
            latency_ms=latency * 1000,
            parse_metadata=meta,
            rag_context=rag_context,
        )
