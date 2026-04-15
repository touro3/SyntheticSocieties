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

from typing import Optional

from decision.llm_backend import LLMBackend
from decision.llm_policy_base import LLMPolicyBase
from decision.prompt_builder import (
    build_context_block,
    build_memory_block,
    build_persona_block,
    build_state_block,
    get_neighbors,
)
from decision.schemas import ProposedAction
from decision.system_prompts import (
    BASE_SYSTEM_PROMPT as SYSTEM_PROMPT,
)
from decision.system_prompts import (
    SYSTEM_PROMPT_NO_INSTITUTIONS,
)


class AblatedLLMPolicy(LLMPolicyBase):
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
        perturbation_mode: Optional[str] = None,
    ):
        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(f"Invalid ablation: {ablation}. Valid: {sorted(self.VALID_ABLATIONS)}")
        self.backend = backend
        self.ablation = ablation
        self.memory_window = memory_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.prompt_logger = prompt_logger
        self.graph_rag = graph_rag
        self.sql_rag = sql_rag
        self.perturbation_mode = perturbation_mode
        # Used by kernel.run_round_batched via getattr(policy, "ablation_level", 5).
        # no_network must use level 2 to suppress trust_network from the state block.
        self.ablation_level = 2 if ablation == "no_network" else 5

    def propose_action(
        self,
        profile,
        state,
        memory,
        context: dict,
        round_id: int,
    ) -> ProposedAction:
        neighbors = get_neighbors(context)
        messages = self._build_ablated_prompt(profile, state, memory, context, round_id)

        if self.perturbation_mode:
            from decision.prompt_perturbation import apply_perturbation

            seed = hash((round_id, profile.agent_id)) % (2**31)
            messages = apply_perturbation(messages, mode=self.perturbation_mode, seed=seed)

        action, raw_text, latency, parse_meta = self._generate_with_retries(messages, neighbors)

        if action is None:
            action = self._fallback_action(state, neighbors, profile=profile)
            parse_meta["fallback"] = True

        prompt_text = "\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)
        self._log_prompt(
            round_id=round_id,
            agent_id=profile.agent_id,
            prompt_text=prompt_text,
            raw_text=raw_text,
            action=action,
            latency=latency,
            parse_meta=parse_meta,
            extra_meta={
                "ablation": self.ablation,
                "rag_context": {
                    "sql_rag_present": bool(self.sql_rag),
                    "graph_rag_present": bool(self.graph_rag) and self.ablation != "no_network",
                },
            },
        )

        return action

    def _build_ablated_prompt(self, profile, state, memory, context, round_id) -> list[dict]:
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

        # State — suppress trust_network for no_network ablation to avoid leaking
        # neighbour-relationship data into a condition that is supposed to strip it.
        state_ablation_level = 2 if self.ablation == "no_network" else 5
        state_desc = build_state_block(state, ablation_level=state_ablation_level)

        # Memory block — pass profile for persona re-anchoring when
        # the ablation mode includes a full persona.
        if self.ablation == "no_memory":
            memory_desc = ""
        else:
            anchor_profile = None
            if self.ablation in ("rich_persona", "no_network", "no_institutions"):
                anchor_profile = profile
            memory_desc = build_memory_block(
                memory,
                window=self.memory_window,
                profile=anchor_profile,
            )

        # Context block
        if self.ablation == "no_network":
            context_desc = build_context_block(
                {
                    "world": context.get("world", {}),
                    "network": {"neighbors": []},
                }
            )
        else:
            context_desc = build_context_block(context)

        # RAG context — SQL RAG is always agent-demographic (not network-related).
        # Graph RAG is social-network context; suppress it for no_network ablation.
        social_context = None
        if self.ablation != "no_network" and self.graph_rag:
            social_context = self.graph_rag.get_social_context(profile.agent_id)

        population_context = None
        if self.sql_rag:
            population_context = self.sql_rag.get_peer_group_context(
                age=profile.age, gender=profile.gender, country=profile.country
            )

        parts = [f"Round {round_id}.", persona, state_desc]

        if population_context:
            parts.append(f"General Population Trends:\n{population_context}")

        if social_context:
            parts.append(f"Social Network Context:\n{social_context}")

        if memory_desc:
            parts.append(memory_desc)
        parts.append(context_desc)
        parts.append("What action do you take this round? Respond with ONLY the JSON.")

        user_content = "\n\n".join(parts)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
