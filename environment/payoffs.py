"""Canonical game-theory payoff constants for the BGF economy.

Single source of truth — all modules must import from here.
Frozen dataclass ensures immutability during simulation runs.

Game-theoretic model:
  Cooperation is a public-goods interaction. The cooperator spends `amount`
  wealth, but the recipient receives `amount * cooperation_multiplier`.
  When multiplier > 1.0, cooperation creates net surplus — giving rational
  agents a reason to cooperate beyond altruism. When multiplier = 1.0,
  cooperation is a pure zero-sum transfer.

  Default multiplier = 1.5 (recipient gets 50% more than cooperator spends).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GamePayoffs:
    work_income: float = 10.0
    work_stress_increase: float = 1.0
    save_wealth_delta: float = 0.0
    save_stress_relief: float = -0.2
    cooperate_stress_relief: float = -0.1
    cooperation_multiplier: float = 1.5


# Default instance — import this, don't instantiate your own unless testing.
DEFAULT_PAYOFFS = GamePayoffs()
