"""Trajectory + pattern bank — a Python adaptation of ruflo's ReasoningBank.

ruflo records agent trajectories (state → action → outcome → verdict) and
distills the recurring successful ones into reusable patterns.  Here we keep
the *recording* and *read-only aggregation* halves only: a structured
per-agent trajectory log plus ``top_patterns()`` analysis.

Critically, this is **observational**.  Patterns are NOT fed back into agent
decisions by default — doing so would contaminate the controlled A/B design.
The bank is an analysis artifact (like the existing metrics modules); an
opt-in learning policy can consume it in a future, clearly-labelled study.

Storage reuses the project's JSONL convention
(``experiments/<exp_id>/trajectory.jsonl``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _verdict(outcome: dict) -> str:
    """Coarse outcome label used for pattern mining.

    Positive wealth change → 'good'; negative → 'bad'; otherwise 'neutral'.
    Mirrors ruflo's per-step success/failure verdicts.
    """
    delta = 0.0
    if isinstance(outcome, dict):
        delta = outcome.get("wealth_delta", 0) or 0
    if delta > 0:
        return "good"
    if delta < 0:
        return "bad"
    return "neutral"


def _state_digest(agent) -> str:
    """Compact, low-cardinality state key so patterns generalize.

    Buckets wealth into coarse bands; pairs with the agent's social class.
    Defensive against partial agent stubs (used in tests).
    """
    try:
        wealth = float(getattr(agent.state, "wealth", 0.0))
    except Exception:
        wealth = 0.0
    band = "low" if wealth < 40 else "mid" if wealth < 80 else "high"
    sclass = getattr(getattr(agent, "profile", None), "social_class", "?")
    return f"wealth={band};class={sclass}"


class TrajectoryBank:
    """Append-only trajectory recorder with read-only pattern aggregation."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def record(self, agent, action_type: str, outcome: dict, round_id: int) -> None:
        row = {
            "round_id": round_id,
            "agent_id": getattr(getattr(agent, "profile", None), "agent_id", "?"),
            "state": _state_digest(agent),
            "action": action_type,
            "verdict": _verdict(outcome),
            "wealth_delta": (outcome or {}).get("wealth_delta", 0) if isinstance(outcome, dict) else 0,
        }
        self._fh.write(json.dumps(row) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass

    # ── Analysis ───────────────────────────────────────────────────────────

    @staticmethod
    def top_patterns(jsonl_path: str | Path, limit: int = 10) -> list[dict]:
        """Aggregate recurring (state → action → good-outcome) tuples.

        Returns the most frequent state+action combinations together with
        their empirical success rate (fraction of 'good' verdicts), sorted
        by success rate then frequency.  Pure read-only analysis.
        """
        path = Path(jsonl_path)
        if not path.is_file():
            return []

        totals: Counter[tuple[str, str]] = Counter()
        good: Counter[tuple[str, str]] = Counter()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["state"], r["action"])
                totals[key] += 1
                if r["verdict"] == "good":
                    good[key] += 1

        patterns = []
        for key, n in totals.items():
            patterns.append(
                {
                    "state": key[0],
                    "action": key[1],
                    "count": n,
                    "success_rate": round(good[key] / n, 4),
                }
            )
        patterns.sort(key=lambda p: (p["success_rate"], p["count"]), reverse=True)
        return patterns[:limit]

    def __enter__(self) -> TrajectoryBank:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
