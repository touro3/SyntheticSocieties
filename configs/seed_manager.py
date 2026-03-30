"""Deterministic seed hierarchy for reproducible experiments.

Derives independent sub-seeds from a master seed so that changes to
one component (e.g., network topology) don't alter the randomness of
another (e.g., population sampling).

Usage:
    from configs.seed_manager import SeedManager

    seeds = SeedManager(master_seed=42)
    population_rng = seeds.population_seed()   # always the same for seed=42
    network_rng = seeds.network_seed()         # independent from population
"""

from __future__ import annotations

from numpy.random import SeedSequence


class SeedManager:
    """Derives deterministic, independent sub-seeds from a master seed.

    Each component gets its own seed derived via numpy's SeedSequence,
    guaranteeing statistical independence between components while
    maintaining full reproducibility from the master seed.
    """

    def __init__(self, master_seed: int) -> None:
        self._master = master_seed
        self._seq = SeedSequence(master_seed)
        # Pre-spawn 5 child sequences for each component
        self._children = self._seq.spawn(5)

    @property
    def master_seed(self) -> int:
        return self._master

    def population_seed(self) -> int:
        """Seed for population sampling (ESS row selection)."""
        return int(self._children[0].generate_state(1)[0])

    def network_seed(self) -> int:
        """Seed for network topology generation."""
        return int(self._children[1].generate_state(1)[0])

    def llm_seed(self) -> int:
        """Seed for LLM sampling (where supported by backend)."""
        return int(self._children[2].generate_state(1)[0])

    def simulation_seed(self) -> int:
        """Seed for simulation-level randomness (tie-breaking, order)."""
        return int(self._children[3].generate_state(1)[0])

    def analysis_seed(self) -> int:
        """Seed for statistical analysis (bootstrap, permutation tests)."""
        return int(self._children[4].generate_state(1)[0])

    def all_seeds(self) -> dict[str, int]:
        """Return all component seeds as a dict for logging."""
        return {
            "master": self._master,
            "population": self.population_seed(),
            "network": self.network_seed(),
            "llm": self.llm_seed(),
            "simulation": self.simulation_seed(),
            "analysis": self.analysis_seed(),
        }
