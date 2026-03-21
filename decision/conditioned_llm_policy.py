from __future__ import annotations

import warnings
from typing import Optional

from decision.experimental_prompt_builder import (
    build_experimental_prompt,
    build_experimental_prompt_text,
)
from decision.llm_backend import LLMBackend
from decision.output_parser import parse_llm_output
from decision.schemas import ProposedAction


class ConditionedLLMPolicy:
    ACTION_BOUNDS = {
        "work": (5.0, 15.0),
        "save": (5.0, 10.0),
        "cooperate": (5.0, 10.0),
    }

    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        memory_window: int = 5,
        temperature: float = 0.7,
        max_retries: int = 2,
        prompt_logger=None,
        graph_rag=None,
        fixed_population_context: Optional[str] = None,
        use_population_context: bool = True,
        use_social_context: bool = True,
        use_memory_context: bool = True,
        system_prompt_mode: str = "balanced",
        include_balancing_hint: bool = True,
        extra_guidance: Optional[str] = None,
        condition_name: Optional[str] = None,
    ):
        self.backend = backend
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.graph_rag = graph_rag
        self.fixed_population_context = fixed_population_context
        self.use_population_context = use_population_context
        self.use_social_context = use_social_context
        self.use_memory_context = use_memory_context
        self.system_prompt_mode = system_prompt_mode
        self.include_balancing_hint = include_balancing_hint
        self.extra_guidance = extra_guidance
        self.condition_name = condition_name

    def propose_action(self, profile, state, memory, context: dict, round_id: int) -> ProposedAction:
        neighbors = context.get("network", {}).get("neighbors", [])

        social_context = None
        if self.use_social_context and self.graph_rag:
            social_context = self.graph_rag.get_social_context(profile.agent_id)

        population_context = self.fixed_population_context if self.use_population_context else None

        messages = build_experimental_prompt(
            profile=profile,
            state=state,
            memory=memory,
            context=context,
            round_id=round_id,
            memory_window=self.memory_window,
            social_context=social_context,
            population_context=population_context,
            use_memory_context=self.use_memory_context,
            use_social_context=self.use_social_context,
            use_population_context=self.use_population_context,
            system_prompt_mode=self.system_prompt_mode,
            include_balancing_hint=self.include_balancing_hint,
            extra_guidance=self.extra_guidance,
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
            except Exception as exc:
                warnings.warn(f"Conditioned LLM generation failed (attempt {attempt + 1}): {exc}")
                parse_meta = {"parse_error": str(exc), "parse_success": False}

        if action is None:
            action = self._fallback_action(state, neighbors)
            parse_meta["fallback"] = True

        action, correction_meta = self._sanitize_action(action, neighbors, state)
        parse_meta.update(correction_meta)

        if self.prompt_logger is not None:
            prompt_text = build_experimental_prompt_text(
                profile=profile,
                state=state,
                memory=memory,
                context=context,
                round_id=round_id,
                memory_window=self.memory_window,
                social_context=social_context,
                population_context=population_context,
                use_memory_context=self.use_memory_context,
                use_social_context=self.use_social_context,
                use_population_context=self.use_population_context,
                system_prompt_mode=self.system_prompt_mode,
                include_balancing_hint=self.include_balancing_hint,
                extra_guidance=self.extra_guidance,
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
                    "condition_name": self.condition_name,
                    "use_population_context": self.use_population_context,
                    "use_social_context": self.use_social_context,
                    "use_memory_context": self.use_memory_context,
                    "system_prompt_mode": self.system_prompt_mode,
                    "include_balancing_hint": self.include_balancing_hint,
                },
            )

        return action

    def _sanitize_action(self, action: ProposedAction, neighbors: list[str], state) -> tuple[ProposedAction, dict]:
        meta: dict = {}

        if action.action_type not in self.ACTION_BOUNDS:
            meta["action_type_corrected"] = {"from": action.action_type, "to": "save"}
            action = action.model_copy(update={"action_type": "save", "target_agent_id": None, "amount": 5.0})

        low, high = self.ACTION_BOUNDS[action.action_type]
        original_amount = float(action.amount)
        clamped_amount = min(high, max(low, original_amount))
        if clamped_amount != original_amount:
            meta["amount_clamped"] = {"from": original_amount, "to": clamped_amount}
            action = action.model_copy(update={"amount": clamped_amount})

        if action.action_type == "cooperate":
            if not neighbors:
                meta["cooperate_without_neighbors"] = True
                return (
                    self._fallback_action(state, neighbors),
                    meta,
                )
            if action.target_agent_id not in neighbors:
                corrected_target = neighbors[0]
                meta["target_corrected"] = {"from": action.target_agent_id, "to": corrected_target}
                action = action.model_copy(update={"target_agent_id": corrected_target})
        else:
            if action.target_agent_id is not None:
                meta["target_removed"] = action.target_agent_id
                action = action.model_copy(update={"target_agent_id": None})

        return action, meta

    def _fallback_action(self, state, neighbors: list[str]) -> ProposedAction:
        if state.stress >= 0.75:
            return ProposedAction(
                action_type="save",
                amount=7.0,
                reasoning_summary="[Conditioned LLM fallback: high stress, saving]",
                confidence=0.5,
            )

        if state.wealth < 70:
            return ProposedAction(
                action_type="work",
                amount=12.0,
                reasoning_summary="[Conditioned LLM fallback: low wealth, working]",
                confidence=0.5,
            )

        if neighbors and state.wealth >= 100:
            return ProposedAction(
                action_type="cooperate",
                target_agent_id=neighbors[0],
                amount=5.0,
                reasoning_summary="[Conditioned LLM fallback: stable wealth, cooperating]",
                confidence=0.5,
            )

        return ProposedAction(
            action_type="save",
            amount=5.0,
            reasoning_summary="[Conditioned LLM fallback: default saving]",
            confidence=0.5,
        )
