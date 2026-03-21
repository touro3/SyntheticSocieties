"""
Ablated LLM policy — LLM decision-making with intentionally degraded prompts.

Uses the same LLMBackend but strips components from the prompt to serve
as an ablation control. This measures the marginal contribution of
persona conditioning, memory, and network context.

Ablation modes:
  - "no_persona"       → omit persona block entirely
  - "minimal_persona"  → only age + gender
  - "no_memory"        → omit memory block
  - "no_network"       → omit neighbor info
  - "no_institutions"  → remove action constraints from system prompt
  - "rich_persona"     → full persona (same as default LLM, serves as control)
"""

from __future__ import annotations

import warnings
from typing import Optional

from decision.llm_backend import LLMBackend
from decision.output_parser import parse_llm_output
from decision.prompt_builder import (
    BASE_SYSTEM_PROMPT as SYSTEM_PROMPT,
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_state_block,
)

from decision.schemas import ProposedAction


# System prompt without action constraints (for no_institutions ablation)
SYSTEM_PROMPT_NO_INSTITUTIONS = """You are a person living in a simulated society. You must decide what action to take this round based on your situation.

You MUST respond with ONLY a JSON object in the following format:
{
  "action_type": "<your choice>",
  "target_agent_id": "<neighbor_id or null>",
  "amount": <number>,
  "reasoning_summary": "<brief explanation>",
  "confidence": <0.0 to 1.0>
}

You can do anything you want. Respond with ONLY the JSON."""


class AblatedLLMPolicy:
    """
    LLM policy with ablated prompt components for controlled experiments.
    """

    VALID_ABLATIONS = {
        "no_persona",
        "minimal_persona",
        "rich_persona",
        "no_memory",
        "no_network",
        "no_institutions",
    }

    def __init__(
        self,
        backend: LLMBackend,
        ablation: str = "no_persona",
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
        graph_rag=None,
        sql_rag=None,
    ):
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"Invalid ablation: {ablation}. "
                f"Valid: {sorted(self.VALID_ABLATIONS)}"
            )
        self.backend = backend
        self.ablation = ablation
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.graph_rag = graph_rag
        self.sql_rag = sql_rag

    def propose_action(
        self,
        profile,
        state,
        memory,
        context: dict,
        round_id: int,
    ) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])
        messages = self._build_ablated_prompt(
            profile, state, memory, context, round_id
        )

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
                warnings.warn(f"Ablated LLM failed (attempt {attempt+1}): {e}")
                parse_meta = {"parse_error": str(e), "parse_success": False}

        if action is None:
            action = self._fallback(state, neighbors)
            parse_meta["fallback"] = True

        # Log
        if self.prompt_logger is not None:
            prompt_text = "\n".join(
                f"[{m['role'].upper()}]\n{m['content']}" for m in messages
            )
            self.prompt_logger.log(
                round_id=round_id,
                agent_id=profile.agent_id,
                prompt=prompt_text,
                raw_output=raw_text,
                parsed_action=action.model_dump() if action else None,
                latency_ms=latency * 1000,
                parse_metadata={
                    **parse_meta,
                    "ablation": self.ablation,
                },
            )

        return action

    def _build_ablated_prompt(
        self, profile, state, memory, context, round_id
    ) -> list[dict]:
        """Build prompt with specific components removed."""

        # System prompt
        if self.ablation == "no_institutions":
            system = SYSTEM_PROMPT_NO_INSTITUTIONS
        else:
            system = SYSTEM_PROMPT

        # Persona block
        if self.ablation == "no_persona":
            persona = "You are an anonymous participant."
        elif self.ablation == "minimal_persona":
            age = getattr(profile, "age", "unknown")
            gender = getattr(profile, "gender", None)
            g_str = "male" if gender == 1 else "female" if gender == 2 else ""
            persona = f"You are a {age}-year-old {g_str} participant.".strip()
        else:
            # rich_persona, no_memory, no_network, no_institutions all get full persona
            persona = build_persona_block(profile)

        # State
        state_desc = build_state_block(state)

        # Memory block
        if self.ablation == "no_memory":
            memory_desc = ""
        else:
            memory_desc = build_memory_block(memory, window=self.memory_window)

        # Context block
        if self.ablation == "no_network":
            context_desc = build_context_block({
                "world": context.get("world", {}),
                "network": {"neighbors": []},
            })
        else:
            context_desc = build_context_block(context)

        parts = [f"Round {round_id}.", persona, state_desc]
        if memory_desc:
            parts.append(memory_desc)
        parts.append(context_desc)
        parts.append("What action do you take this round? Respond with ONLY the JSON.")

        user_content = "\n\n".join(parts)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _fallback(self, state, neighbors: list[str]) -> ProposedAction:
        if state.wealth < 70:
            return ProposedAction(
                action_type="work",
                amount=10.0,
                reasoning_summary=f"[Ablated LLM fallback ({self.ablation}): work]",
                confidence=0.5,
            )
        if neighbors and state.wealth >= 100:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=5.0,
                reasoning_summary=f"[Ablated LLM fallback ({self.ablation}): cooperate]",
                confidence=0.5,
            )
        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary=f"[Ablated LLM fallback ({self.ablation}): save]",
            confidence=0.5,
        )
