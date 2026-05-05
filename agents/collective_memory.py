from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field


@dataclass
class WorldFact:
    round_id: int
    fact_type: str
    content: str
    importance: float
    last_decayed_round: int | None = field(default=None, repr=False)


class CollectiveMemory:
    """Thread-safe shared memory of population-level world facts."""

    def __init__(
        self,
        half_life_rounds: float = 10.0,
        prune_below: float = 0.05,
        max_facts: int = 100,
    ) -> None:
        self.half_life_rounds = max(1.0, float(half_life_rounds))
        self.prune_below = max(0.0, float(prune_below))
        self.max_facts = max(1, int(max_facts))
        self._facts: list[WorldFact] = []
        self._lock = threading.RLock()

    def record(self, round_id: int, fact_type: str, content: str, importance: float = 0.5) -> None:
        content = str(content).strip()
        if not content:
            return
        fact = WorldFact(
            round_id=int(round_id),
            fact_type=str(fact_type),
            content=content,
            importance=self._clamp01(importance),
            last_decayed_round=int(round_id),
        )
        with self._lock:
            self._facts.append(fact)
            if len(self._facts) > self.max_facts:
                self._facts.sort(key=lambda item: (item.importance, item.round_id), reverse=True)
                del self._facts[self.max_facts :]

    def advance_round(self, current_round: int) -> None:
        current_round = int(current_round)
        decay = math.log(2) / self.half_life_rounds
        with self._lock:
            live: list[WorldFact] = []
            for fact in self._facts:
                last = fact.last_decayed_round if fact.last_decayed_round is not None else fact.round_id
                delta = max(0, current_round - last)
                if delta:
                    fact.importance *= math.exp(-decay * delta)
                    fact.last_decayed_round = current_round
                if fact.importance >= self.prune_below:
                    live.append(fact)
            self._facts = live

    def get_context(self, max_items: int = 3) -> list[str]:
        with self._lock:
            ranked = sorted(self._facts, key=lambda item: (item.importance, item.round_id), reverse=True)
            selected = ranked[: max(0, max_items)]
            return [f"Round {fact.round_id} {fact.fact_type}: {fact.content}" for fact in selected]

    def snapshot(self) -> list[WorldFact]:
        with self._lock:
            return list(self._facts)

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5
