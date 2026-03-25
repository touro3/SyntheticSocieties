"""Canonical game-theory payoff constants for the BGF economy.

Single source of truth — all modules must import from here.
Frozen dataclass ensures immutability during simulation runs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GamePayoffs:
    work_income: float = 10.0
    work_stress_increase: float = 1.0
    save_wealth_delta: float = 0.0
    save_stress_relief: float = -0.2
    cooperate_stress_relief: float = -0.1


# Default instance — import this, don't instantiate your own unless testing.
DEFAULT_PAYOFFS = GamePayoffs()
