"""PaddedAblationPolicy — length-controlled control for causal ablation.

Phase 28 / TOP_TIER_RESEARCH Section 1.

Condition P: prompts are padded to the same token count as a fully
grounded Condition B prompt, but ESS content is replaced with
semantically empty filler.

Causal logic:
  BRM(P) ≈ BRM(A)  →  length alone does not explain the grounding effect
  BRM(B) >> BRM(P) →  content (ESS grounding) is the causal driver
  BRM(P) > BRM(A)  →  length contributes partially; report decomposition

Either result is publishable — the contribution is having the controlled
comparison.
"""

from __future__ import annotations

from typing import Optional

from decision.llm_policy_base import LLMPolicyBase
from decision.padded_prompt_builder import build_padded_prompt, measure_grounded_token_count
from decision.prompt_builder import get_neighbors
from decision.schemas import ProposedAction


class PaddedAblationPolicy(LLMPolicyBase):
    """LLM policy whose prompts match Condition B token count but carry no ESS content.

    The policy inherits retry logic, fallback, and logging from LLMPolicyBase.
    The only difference from a plain LLMPolicy is the prompt construction:
    instead of injecting ESS persona + RAG context, it pads to the same
    target token count with semantically neutral filler sentences.

    Args:
        backend: LLM backend (LLMBackend or compatible).
        temperature: Sampling temperature.
        max_retries: Number of generation retry attempts.
        prompt_logger: Optional PromptLogger for prompt auditing.
        memory_window: Number of recent memory items to include.
        target_token_count: Fixed target token count for the user message.
            If None (default), the target is measured dynamically per agent
            using ``measure_grounded_token_count`` — i.e. each agent's padded
            prompt is matched to what *their* fully grounded prompt would be.
    """

    def __init__(
        self,
        backend,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
        memory_window: int = 5,
        target_token_count: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.memory_window = memory_window
        self.target_token_count = target_token_count

    def propose_action(
        self,
        profile,
        state,
        memory,
        context: dict,
        round_id: int,
    ) -> ProposedAction:
        neighbors = get_neighbors(context)

        # Determine target token count
        target = self.target_token_count
        if target is None:
            # Match what Condition B would produce for this specific agent
            target = measure_grounded_token_count(
                profile, state, memory, context, round_id,
                memory_window=self.memory_window,
            )

        # Derive a deterministic seed from (round_id, agent_id) so runs are
        # reproducible but padding differs across rounds and agents.
        seed = hash((round_id, profile.agent_id)) % (2**31)

        messages = build_padded_prompt(
            profile, state, memory, context, round_id,
            target_token_count=target,
            seed=seed,
            memory_window=self.memory_window,
        )

        action, raw_text, latency, parse_meta = self._generate_with_retries(messages, neighbors)

        if action is None:
            action = self._fallback_action(state, neighbors, profile=profile)
            parse_meta["fallback"] = True

        self._log_prompt(
            round_id=round_id,
            agent_id=profile.agent_id,
            prompt_text=messages[1]["content"] if len(messages) > 1 else "",
            raw_text=raw_text,
            action=action,
            latency=latency,
            parse_meta=parse_meta,
            extra_meta={"condition": "P", "target_token_count": target},
        )

        return action
