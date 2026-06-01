"""BGF decision layer — public API.

All policy classes and the action schema are importable directly from this
package, so callers never need to know which sub-module hosts each class.

Quick reference:

    Policy          Description
    ─────────────── ──────────────────────────────────────────────────────
    MockPolicy      Returns a fixed action — unit-test baseline
    RandomPolicy    Uniform random choice among valid actions
    RuleBasedPolicy Deterministic wealth-based rules — non-LLM baseline
    TemplatePolicy  Persona-driven rules without LLM — non-LLM baseline
    LLMPolicy       Full LLM inference with optional RAG (Condition B)
    AblatedLLMPolicy LLM inference with specific grounding components removed

Data classes:

    ProposedAction  Pydantic schema for every action produced by a policy

Helpers:

    AblationLevel   Named constants for the five grounding levels (0–5)
    get_neighbors   Extract neighbor IDs from a world-context dict
"""

from decision.ablated_llm_policy import AblatedLLMPolicy
from decision.constants import (
    COOPERATE_WEALTH_THRESHOLD,
    DEFAULT_COOPERATE_AMOUNT,
    DEFAULT_SAVE_AMOUNT,
    DEFAULT_WORK_AMOUNT,
    MAX_ACTION_AMOUNT,
    STRESS_CRITICAL,
    WORK_WEALTH_THRESHOLD,
)
from decision.llm_policy import LLMPolicy
from decision.mock_policy import MockPolicy
from decision.prompt_builder import AblationLevel, get_neighbors
from decision.random_policy import RandomPolicy
from decision.rule_based_policy import RuleBasedPolicy
from decision.schemas import ProposedAction
from decision.social_policy import SocialAction, SocialPolicy
from decision.template_policy import TemplatePolicy

__all__ = [
    # Action schema
    "ProposedAction",
    # Policies (non-LLM baselines)
    "MockPolicy",
    "RandomPolicy",
    "RuleBasedPolicy",
    "TemplatePolicy",
    # Policies (LLM-based)
    "LLMPolicy",
    "AblatedLLMPolicy",
    "SocialAction",
    "SocialPolicy",
    # Helpers & constants
    "AblationLevel",
    "get_neighbors",
    "WORK_WEALTH_THRESHOLD",
    "COOPERATE_WEALTH_THRESHOLD",
    "STRESS_CRITICAL",
    "DEFAULT_WORK_AMOUNT",
    "DEFAULT_SAVE_AMOUNT",
    "DEFAULT_COOPERATE_AMOUNT",
    "MAX_ACTION_AMOUNT",
]
