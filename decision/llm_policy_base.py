"""Shared retry, fallback, and logging logic for all LLM-based policies.

Eliminates the 3x code duplication across LLMPolicy, AblatedLLMPolicy,
and ConditionedLLMPolicy. Subclasses override only prompt construction.
"""

from __future__ import annotations

import logging
import math

from decision.constants import (
    COOPERATE_WEALTH_THRESHOLD,
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_FALLBACK_CONFIDENCE,
    DEFAULT_WORK_AMOUNT,
    RISK_HIGH,
    TRUST_HIGH,
    TRUST_LOW,
    WORK_WEALTH_THRESHOLD,
)
from decision.output_parser import parse_llm_output
from decision.schemas import ProposedAction

logger = logging.getLogger(__name__)


class LLMPolicyBase:
    """Base class for LLM policies with shared infrastructure."""

    # Subclasses must set these in __init__:
    backend: object
    temperature: float
    max_retries: int
    prompt_logger: object | None

    # Instance-level counters for tracking fallback usage and total proposals.
    # Lazily initialized via _ensure_counters() to avoid __init__ coupling.

    def _ensure_counters(self) -> None:
        """Lazily initialize proposal counters on first use."""
        if not hasattr(self, "_fallback_counter"):
            self._fallback_counter = 0
        if not hasattr(self, "_total_proposals"):
            self._total_proposals = 0

    # ── RAG context accessors ─────────────────────────────────────────────────

    def graph_rag_context(self, agent_id: str) -> str | None:
        """Return graph-RAG social context string for *agent_id*, or None."""
        rag = getattr(self, "graph_rag", None)
        if rag is None:
            return None
        return rag.get_social_context(agent_id)

    def sql_rag_context(self, age: int, gender: str, country: str) -> str | None:
        """Return SQL-RAG peer-group context string, or None."""
        rag = getattr(self, "sql_rag", None)
        if rag is None:
            return None
        return rag.get_peer_group_context(age=age, gender=gender, country=country)

    def _generate_with_retries(
        self,
        messages: list[dict],
        neighbors: list[str],
    ) -> tuple[ProposedAction | None, str, float, dict]:
        """Generate LLM output with retries. Returns (action, raw_text, latency, parse_meta).

        When the backend supports logprobs (LLMBackend with return_logprobs=True),
        the parsed action's confidence is replaced with exp(first_token_logprob) —
        a calibrated model-derived value — instead of the LLM's self-reported
        confidence field, which is known to be poorly calibrated.
        """
        self._ensure_counters()
        self._total_proposals += 1
        action = None
        raw_text = ""
        latency = 0.0
        parse_meta: dict = {}

        for attempt in range(self.max_retries + 1):
            try:
                # Request logprobs for calibrated confidence. Falls back to the
                # two-element return if the backend doesn't support this parameter.
                try:
                    result = self.backend.generate(
                        messages=messages,
                        temperature=self.temperature,
                        return_logprobs=True,
                    )
                    raw_text, latency = result[0], result[1]
                    logprob: float | None = result[2] if len(result) > 2 else None
                except TypeError:
                    # Backend doesn't accept return_logprobs (e.g. OpenAIBackend).
                    raw_text, latency = self.backend.generate(
                        messages=messages,
                        temperature=self.temperature,
                    )
                    logprob = None

                action, parse_meta = parse_llm_output(raw_text, neighbors)

                if action is not None:
                    # Replace self-reported confidence with the model's actual
                    # first-token log-probability when available. exp(logprob)
                    # gives a calibrated probability in [0, 1]; the LLM's
                    # self-reported "confidence" field is not calibrated.
                    if logprob is not None and logprob != float("-inf"):
                        calibrated_conf = max(0.0, min(1.0, math.exp(logprob)))
                        action = action.model_copy(update={"confidence": calibrated_conf})
                        parse_meta["confidence_source"] = "logprob"
                    else:
                        parse_meta["confidence_source"] = "self_reported"
                    break

            except Exception as e:
                logger.warning("LLM generation failed (attempt %d): %s", attempt + 1, e)
                parse_meta = {"parse_error": str(e), "parse_success": False}

        return action, raw_text, latency, parse_meta

    def _fallback_action(self, state, neighbors: list[str], profile=None) -> ProposedAction:
        """Rule-based fallback when LLM fails.

        When a profile is provided, uses persona traits (trust_people,
        risk_tolerance) to produce persona-consistent fallback behavior
        instead of purely wealth-based rules.
        """
        self._ensure_counters()
        self._fallback_counter += 1

        trust = getattr(profile, "trust_people", None) if profile else None
        risk = getattr(profile, "risk_tolerance", None) if profile else None

        # Persona-aware fallback when profile traits are available
        if trust is not None and risk is not None:
            # Low trust agents prefer saving even when wealthy
            if trust < TRUST_LOW:
                return ProposedAction(
                    action_type="save",
                    amount=DEFAULT_COOPERATE_AMOUNT,
                    reasoning_summary=f"[LLM fallback: saving — low trust ({trust:.2f})]",
                    confidence=DEFAULT_FALLBACK_CONFIDENCE,
                )
            # High risk tolerance agents work harder
            if risk > RISK_HIGH and state.wealth < COOPERATE_WEALTH_THRESHOLD * 1.2:
                return ProposedAction(
                    action_type="work",
                    amount=DEFAULT_WORK_AMOUNT,
                    reasoning_summary=f"[LLM fallback: working — high risk tolerance ({risk:.2f})]",
                    confidence=DEFAULT_FALLBACK_CONFIDENCE,
                )
            # High trust agents cooperate at moderate wealth
            if trust > TRUST_HIGH and neighbors and state.wealth >= WORK_WEALTH_THRESHOLD * 0.7:
                return ProposedAction(
                    action_type="cooperate",
                    target_agent_id=neighbors[0],
                    amount=DEFAULT_COOPERATE_AMOUNT,
                    reasoning_summary=f"[LLM fallback: cooperating — high trust ({trust:.2f})]",
                    confidence=DEFAULT_FALLBACK_CONFIDENCE,
                )

        # Default wealth-based fallback (backward compatible)
        if state.wealth < WORK_WEALTH_THRESHOLD:
            return ProposedAction(
                action_type="work",
                amount=DEFAULT_WORK_AMOUNT,
                reasoning_summary="[LLM fallback: working due to low wealth]",
                confidence=DEFAULT_FALLBACK_CONFIDENCE,
            )
        if neighbors and state.wealth >= COOPERATE_WEALTH_THRESHOLD:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=DEFAULT_COOPERATE_AMOUNT,
                reasoning_summary="[LLM fallback: cooperating with high wealth]",
                confidence=DEFAULT_FALLBACK_CONFIDENCE,
            )
        return ProposedAction(
            action_type="save",
            amount=DEFAULT_COOPERATE_AMOUNT,
            reasoning_summary="[LLM fallback: saving as default]",
            confidence=DEFAULT_FALLBACK_CONFIDENCE,
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

    def get_fallback_rate(self) -> float:
        """Return the fraction of proposals that used fallback.

        A high fallback rate (> 0.15) indicates the LLM is struggling to
        produce parseable output and the experiment results may be
        unreliable — dominated by rule-based behavior, not LLM behavior.
        """
        self._ensure_counters()
        if self._total_proposals == 0:
            return 0.0
        return self._fallback_counter / self._total_proposals

    def get_proposal_stats(self) -> dict:
        """Return diagnostic counters for experiment-level reporting."""
        return {
            "total_proposals": self._total_proposals,
            "fallback_count": self._fallback_counter,
            "fallback_rate": self.get_fallback_rate(),
        }
