"""Generational cultural-evolution loop over the iterated Donor Game.

Faithful to Vallinder & Hughes (2024, arXiv:2412.10270):

  for generation in range(N):
      run R donor-game rounds          # environment.donor_game.DonorGameEngine
      rank agents by accumulated resources
      keep the top `survivor_fraction` (default 0.5)
      spawn offspring to replace the culled, each inheriting a survivor's
          distilled *strategy text* (prompt-text transmission)
      reset resources for the next generation

The runner is policy-agnostic. It is driven by three injected callables so it
runs with mock/rule-based deciders (tests, dry runs) and the LLM donor policy
(Phase 2) alike:

  * ``decision_fn(donor, recipient, reputation, round_id) -> DonorDecision``
        how an agent decides in an encounter (LLM reads ``agent.inherited_strategy``).
  * ``strategy_fn(agent, encounters) -> str``
        distil a survivor's strategy into text for its successor.
  * ``offspring_fn(parent, new_id, inherited_strategy) -> Agent``
        build a new agent from a surviving parent, carrying the inherited text.

Defaults are provided for all three (rule-based summary, profile-cloning
offspring) so the loop is usable out of the box.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from environment.donor_game import DONATE, DonorGameEngine, EncounterRecord, reputation_threshold

StrategyFn = Callable[["object", "list[EncounterRecord]"], str]
OffspringFn = Callable[["object", str, str], "object"]


# ── Records ───────────────────────────────────────────────────────────────────


@dataclass
class LineageEntry:
    generation: int
    agent_id: str
    parent_id: str | None
    fitness_rank: int  # 0 = fittest in its generation
    final_resources: float
    survived: bool
    inherited_strategy: str = ""


@dataclass
class GenerationSummary:
    generation: int
    cooperation_rate: float  # mean donation rate across the generation's rounds
    mean_resources: float
    survivor_ids: list[str]
    culled_ids: list[str]
    agent_resources: dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    generations: list[GenerationSummary]
    lineage: list[LineageEntry]
    config: dict

    @property
    def cooperation_trajectory(self) -> list[float]:
        return [g.cooperation_rate for g in self.generations]


# ── Default strategy distillation + offspring construction ────────────────────


def default_strategy_fn(agent, encounters: list[EncounterRecord]) -> str:
    """Rule-based summary of how an agent behaved as a donor this generation.

    Used as the inheritable "advice" when no LLM articulation is supplied.
    """
    mine = [e for e in encounters if e.donor_id == agent.profile.agent_id]
    if not mine:
        return "No donor history; cooperate conditionally on recipient reputation."
    donate_rate = sum(1 for e in mine if e.action == DONATE) / len(mine)
    # Did I condition on reputation? Compare donate-rate to high- vs low-image recipients.
    hi = [e for e in mine if e.recipient_donate_rate >= 0.5]
    lo = [e for e in mine if e.recipient_donate_rate < 0.5]
    hi_rate = (sum(1 for e in hi if e.action == DONATE) / len(hi)) if hi else None
    lo_rate = (sum(1 for e in lo if e.action == DONATE) / len(lo)) if lo else None
    parts = [f"I donated {donate_rate:.0%} of the time."]
    if hi_rate is not None and lo_rate is not None:
        if hi_rate - lo_rate > 0.15:
            parts.append("I rewarded good reputations and withheld from poor ones (indirect reciprocity).")
        elif donate_rate > 0.8:
            parts.append("I cooperated almost unconditionally.")
        elif donate_rate < 0.2:
            parts.append("I rarely cooperated.")
        else:
            parts.append("My cooperation was largely unconditional.")
    return " ".join(parts)


def default_offspring_fn(parent, new_id: str, inherited_strategy: str):
    """Build an offspring by cloning the parent's profile with a fresh state/memory.

    Carries the inherited strategy text on ``agent.inherited_strategy`` so an
    LLM donor policy can fold it into the successor's prompt. Any heritable
    numeric trait used by a custom ``decision_fn`` should be copied by a custom
    offspring_fn instead — this default only transmits the strategy text.
    """
    import copy

    from agents.agent import Agent
    from agents.memory import HierarchicalMemory
    from agents.state import AgentState

    profile = copy.deepcopy(parent.profile)
    profile.agent_id = new_id
    child = Agent(
        profile=profile,
        state=AgentState(wealth=0.0),
        memory=HierarchicalMemory(max_recent=getattr(parent.memory, "max_recent", 10)),
        policy=parent.policy,  # policies are stateless / shared
    )
    child.inherited_strategy = inherited_strategy
    return child


# ── Runner ────────────────────────────────────────────────────────────────────


class EvolutionRunner:
    """Runs the generational donor-game cultural-evolution loop."""

    def __init__(
        self,
        agents: list,
        decision_fn=None,
        *,
        n_generations: int = 10,
        rounds_per_generation: int = 12,
        survivor_fraction: float = 0.5,
        endowment: float = 10.0,
        donation: float = 2.0,
        multiplier: float = 2.0,
        reputation_window: int = 10,
        strategy_fn: StrategyFn = default_strategy_fn,
        offspring_fn: OffspringFn = default_offspring_fn,
        seed: int = 0,
    ) -> None:
        if not 0.0 < survivor_fraction < 1.0:
            raise ValueError("survivor_fraction must be in (0, 1)")
        if len(agents) < 2:
            raise ValueError("need at least 2 agents")
        self.agents = list(agents)
        self.decision_fn = decision_fn if decision_fn is not None else reputation_threshold(0.5)
        self.n_generations = n_generations
        self.rounds_per_generation = rounds_per_generation
        self.survivor_fraction = survivor_fraction
        self.endowment = endowment
        self.donation = donation
        self.multiplier = multiplier
        self.reputation_window = reputation_window
        self.strategy_fn = strategy_fn
        self.offspring_fn = offspring_fn
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Seed agents carry an (empty) inherited strategy unless one is preset.
        for a in self.agents:
            if not hasattr(a, "inherited_strategy"):
                a.inherited_strategy = ""

        self.generations: list[GenerationSummary] = []
        self.lineage: list[LineageEntry] = []
        self._offspring_counter = 0

    def _config(self) -> dict:
        return {
            "n_generations": self.n_generations,
            "rounds_per_generation": self.rounds_per_generation,
            "survivor_fraction": self.survivor_fraction,
            "endowment": self.endowment,
            "donation": self.donation,
            "multiplier": self.multiplier,
            "reputation_window": self.reputation_window,
            "n_agents": len(self.agents),
            "seed": self.seed,
        }

    def _run_generation(self, gen: int) -> tuple[DonorGameEngine, GenerationSummary]:
        # Per-generation engine seed derived from the runner rng for reproducibility.
        engine_seed = int(self.rng.integers(0, 2**31 - 1))
        engine = DonorGameEngine(
            self.agents,
            self.decision_fn,
            endowment=self.endowment,
            donation=self.donation,
            multiplier=self.multiplier,
            reputation_window=self.reputation_window,
            seed=engine_seed,
        )
        rounds = engine.run(self.rounds_per_generation)
        coop = float(np.mean([r.cooperation_rate for r in rounds])) if rounds else 0.0
        resources = dict(engine.resources)
        summary = GenerationSummary(
            generation=gen,
            cooperation_rate=coop,
            mean_resources=float(np.mean(list(resources.values()))),
            survivor_ids=[],  # filled after selection
            culled_ids=[],
            agent_resources=resources,
        )
        return engine, summary

    def _select_and_reproduce(self, gen: int, engine: DonorGameEngine, summary: GenerationSummary) -> None:
        ranked = engine.ranked_agents()  # fittest first
        n_survivors = max(1, int(round(len(ranked) * self.survivor_fraction)))
        survivor_ids = ranked[:n_survivors]
        culled_ids = ranked[n_survivors:]
        survivor_set = set(survivor_ids)
        by_id = {a.profile.agent_id: a for a in self.agents}
        all_encounters = [e for r in engine.history for e in r.encounters]

        # Distil each survivor's strategy once for inheritance.
        survivor_strategy = {sid: self.strategy_fn(by_id[sid], all_encounters) for sid in survivor_ids}

        # Lineage for this generation (resources are final accumulated wealth).
        for rank, aid in enumerate(ranked):
            self.lineage.append(
                LineageEntry(
                    generation=gen,
                    agent_id=aid,
                    parent_id=getattr(by_id[aid], "parent_id", None),
                    fitness_rank=rank,
                    final_resources=engine.resources[aid],
                    survived=aid in survivor_set,
                    inherited_strategy=getattr(by_id[aid], "inherited_strategy", ""),
                )
            )

        summary.survivor_ids = survivor_ids
        summary.culled_ids = culled_ids

        # Build next generation: survivors persist (reset for next gen), culled
        # slots are replaced by offspring of randomly-sampled survivors.
        next_gen: list = [by_id[sid] for sid in survivor_ids]
        n_offspring = len(self.agents) - len(next_gen)
        for _ in range(n_offspring):
            parent_id = survivor_ids[int(self.rng.integers(0, len(survivor_ids)))]
            parent = by_id[parent_id]
            self._offspring_counter += 1
            new_id = f"g{gen + 1}_o{self._offspring_counter}"
            child = self.offspring_fn(parent, new_id, survivor_strategy[parent_id])
            child.parent_id = parent_id
            if not hasattr(child, "inherited_strategy"):
                child.inherited_strategy = survivor_strategy[parent_id]
            next_gen.append(child)

        self.agents = next_gen

    def run(self) -> EvolutionResult:
        for gen in range(self.n_generations):
            engine, summary = self._run_generation(gen)
            self.generations.append(summary)
            # Last generation: record lineage but skip reproduction.
            if gen < self.n_generations - 1:
                self._select_and_reproduce(gen, engine, summary)
            else:
                self._record_terminal_lineage(gen, engine, summary)
        return EvolutionResult(generations=self.generations, lineage=self.lineage, config=self._config())

    def _record_terminal_lineage(self, gen: int, engine: DonorGameEngine, summary: GenerationSummary) -> None:
        ranked = engine.ranked_agents()
        n_survivors = max(1, int(round(len(ranked) * self.survivor_fraction)))
        survivor_set = set(ranked[:n_survivors])
        by_id = {a.profile.agent_id: a for a in self.agents}
        for rank, aid in enumerate(ranked):
            self.lineage.append(
                LineageEntry(
                    generation=gen,
                    agent_id=aid,
                    parent_id=getattr(by_id[aid], "parent_id", None),
                    fitness_rank=rank,
                    final_resources=engine.resources[aid],
                    survived=aid in survivor_set,
                    inherited_strategy=getattr(by_id[aid], "inherited_strategy", ""),
                )
            )
        summary.survivor_ids = list(ranked[:n_survivors])
        summary.culled_ids = list(ranked[n_survivors:])

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_results(self, path: str | Path) -> None:
        """Write generation summaries + lineage as JSON (checkpoint of evolved state)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self._config(),
            "cooperation_trajectory": [g.cooperation_rate for g in self.generations],
            "generations": [asdict(g) for g in self.generations],
            "lineage": [asdict(e) for e in self.lineage],
        }
        path.write_text(json.dumps(payload, indent=2))
