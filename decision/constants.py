"""Named constants for the BGF decision layer.

Single source of truth for all thresholds, default amounts, and confidence
values used by policies and the output parser.  Importing from here (rather
than scattering literals) makes the decision logic self-documenting and easy
to adjust in one place.

Usage:
    from decision.constants import WORK_WEALTH_THRESHOLD, DEFAULT_WORK_AMOUNT
"""

from __future__ import annotations

# ── Wealth thresholds ─────────────────────────────────────────────────────────
# Agents below WORK_WEALTH_THRESHOLD prioritise income generation.
# Agents at or above COOPERATE_WEALTH_THRESHOLD can afford to share surplus.
WORK_WEALTH_THRESHOLD: float = 70.0
COOPERATE_WEALTH_THRESHOLD: float = 100.0

# Minimum wealth an agent must hold to be eligible for cooperation.
MIN_COOPERATE_WEALTH: float = 5.0

# ── Persona thresholds (ESS-derived normalised [0, 1] scores) ─────────────────
# Used by the persona-aware fallback to infer likely behaviour from traits.
TRUST_LOW: float = 0.3  # Below this → agent distrusts others → save
TRUST_HIGH: float = 0.7  # Above this → agent trusts others → cooperate
RISK_HIGH: float = 0.7  # Above this → agent accepts risk → work harder
RELIGIOSITY_THRESHOLD: float = 0.5  # Used to label agents as religious/not

# ── Stress threshold ──────────────────────────────────────────────────────────
# V1 ablation: display a critical warning when stress exceeds this level.
STRESS_CRITICAL: float = 0.7

# ── Default action amounts ────────────────────────────────────────────────────
# Sent in ProposedAction when the LLM or rule produces no explicit amount.
DEFAULT_WORK_AMOUNT: float = 10.0
DEFAULT_SAVE_AMOUNT: float = 5.0
DEFAULT_COOPERATE_AMOUNT: float = 5.0

# Maximum amount any single action may transfer (enforced by output parser).
MAX_ACTION_AMOUNT: float = 20.0

# ── Default confidence values ─────────────────────────────────────────────────
DEFAULT_FALLBACK_CONFIDENCE: float = 0.5  # Rule-based and fallback actions
DEFAULT_RULE_CONFIDENCE: float = 0.9  # RuleBasedPolicy (deterministic)
DEFAULT_RANDOM_CONFIDENCE: float = 0.5  # RandomPolicy (uniform)
DEFAULT_KEYWORD_CONFIDENCE: float = 0.3  # Keyword-fallback parser (low trust)
