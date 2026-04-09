"""Cross-model comparison metrics.

Phase 16 — Multi-Model Generalizability Study.

Aggregates RLHF bias index, cooperation rate, and Gini coefficient across
model families and conditions (A=ungrounded, B=grounded) to produce the
cross-model comparison table (Table 2 in the paper).

Central claim being tested:
    The RLHF cooperative bias is a general phenomenon across instruction-tuned
    LLM families, not a Mistral-7B artifact.

Expected result:
    bias_index(Condition A) > bias_index(Condition B)
    for ALL tested model families.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from metrics.behavioral_realism import rlhf_bias_index_from_counts
from metrics.inequality import gini_coefficient

# ── Result container ─────────────────────────────────────────────────────────


class CrossModelResult:
    """Stores per-model, per-condition comparison metrics.

    Attributes:
        model_id: Short model name (e.g. "mistral-7b").
        condition: "A" (ungrounded) or "B" (grounded).
        cooperation_rate: Fraction of actions that were "cooperate".
        gini: Gini coefficient of final wealth distribution.
        rlhf_bias_index: TV distance from uniform action distribution.
        n_agents: Number of agents in the run.
        n_rounds: Number of simulation rounds.
    """

    def __init__(
        self,
        model_id: str,
        condition: str,
        cooperation_rate: float,
        gini: float,
        rlhf_bias_index: float,
        n_agents: int = 0,
        n_rounds: int = 0,
    ):
        self.model_id = model_id
        self.condition = condition
        self.cooperation_rate = cooperation_rate
        self.gini = gini
        self.rlhf_bias_index = rlhf_bias_index
        self.n_agents = n_agents
        self.n_rounds = n_rounds

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "condition": self.condition,
            "cooperation_rate": round(self.cooperation_rate, 4),
            "gini": round(self.gini, 4),
            "rlhf_bias_index": round(self.rlhf_bias_index, 4),
            "n_agents": self.n_agents,
            "n_rounds": self.n_rounds,
        }


# ── Event parsing helpers ────────────────────────────────────────────────────


def _iter_events(events_path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL events file."""
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_action_counts(events_path: Path) -> dict[str, int]:
    """Count each action type in an experiment's events.jsonl.

    Handles two formats:
    - Flat:   {"action": "work", ...}  (test/synthetic data)
    - Nested: {"action": {"action_type": "work", ...}, ...}  (real kernel output)

    Args:
        events_path: Path to the events.jsonl file.

    Returns:
        Dict mapping action_type → count.
    """
    counts: dict[str, int] = {"work": 0, "save": 0, "cooperate": 0}
    for event in _iter_events(events_path):
        raw = event.get("action") or event.get("action_type")
        if isinstance(raw, dict):
            action = raw.get("action_type", "")
        else:
            action = raw or ""
        if action in counts:
            counts[action] += 1
    return counts


def extract_final_wealth(events_path: Path) -> list[float]:
    """Extract the final wealth of each agent from an events.jsonl file.

    Reads all events and returns the last recorded wealth per agent.
    Handles two formats:
    - Flat:   {"wealth": 120.0, ...}
    - Nested: {"state_after": {"wealth": 120.0}, ...}  (real kernel output)

    Args:
        events_path: Path to the events.jsonl file.

    Returns:
        List of final wealth values (one per agent that appeared).
    """
    last_wealth: dict[str, float] = {}
    for event in _iter_events(events_path):
        agent_id = event.get("agent_id")
        # Try flat format first, then state_after (real kernel), then state
        wealth = (
            event.get("wealth")
            or (event.get("state_after") or {}).get("wealth")
            or (event.get("state") or {}).get("wealth")
        )
        if agent_id and wealth is not None:
            try:
                last_wealth[agent_id] = float(wealth)
            except (TypeError, ValueError):
                pass
    return list(last_wealth.values())


# ── Main aggregation function ────────────────────────────────────────────────


def compute_cross_model_result(
    events_path: Path,
    model_id: str,
    condition: str,
) -> CrossModelResult:
    """Compute cross-model comparison metrics from a simulation run.

    Args:
        events_path: Path to events.jsonl for the completed run.
        model_id: Short model identifier (e.g. "mistral-7b").
        condition: "A" (ungrounded) or "B" (grounded).

    Returns:
        CrossModelResult with all metrics populated.

    Raises:
        ValueError: If events file has no valid action events.
    """
    action_counts = extract_action_counts(events_path)
    total = sum(action_counts.values())
    if total == 0:
        raise ValueError(f"No valid action events found in {events_path}")

    cooperation_rate = action_counts.get("cooperate", 0) / total
    rlhf_bias = rlhf_bias_index_from_counts(action_counts)

    final_wealth = extract_final_wealth(events_path)
    gini = gini_coefficient(final_wealth) if len(final_wealth) > 1 else 0.0

    # Infer n_agents / n_rounds from event structure
    agents_seen = set()
    rounds_seen = set()
    for event in _iter_events(events_path):
        if event.get("agent_id"):
            agents_seen.add(event["agent_id"])
        if event.get("round") is not None:
            rounds_seen.add(event["round"])

    return CrossModelResult(
        model_id=model_id,
        condition=condition,
        cooperation_rate=cooperation_rate,
        gini=gini,
        rlhf_bias_index=rlhf_bias,
        n_agents=len(agents_seen),
        n_rounds=len(rounds_seen),
    )


# ── Comparison table ─────────────────────────────────────────────────────────


def build_comparison_table(results: list[CrossModelResult]) -> list[dict]:
    """Build the cross-model comparison table (Table 2 in paper).

    Pairs Condition A and B results for each model and computes the
    bias reduction: (bias_A - bias_B) / bias_A.

    Args:
        results: List of CrossModelResult instances.

    Returns:
        List of row dicts, one per model, with A/B metrics and delta.
    """
    # Index by (model_id, condition)
    indexed: dict[tuple[str, str], CrossModelResult] = {
        (r.model_id, r.condition): r for r in results
    }

    models = sorted({r.model_id for r in results})
    rows = []

    for model in models:
        a = indexed.get((model, "A"))
        b = indexed.get((model, "B"))

        row: dict = {"model": model}

        if a:
            row["coop_rate_A"] = round(a.cooperation_rate, 3)
            row["bias_A"] = round(a.rlhf_bias_index, 3)
            row["gini_A"] = round(a.gini, 3)

        if b:
            row["coop_rate_B"] = round(b.cooperation_rate, 3)
            row["bias_B"] = round(b.rlhf_bias_index, 3)
            row["gini_B"] = round(b.gini, 3)

        if a and b and a.rlhf_bias_index > 0:
            bias_reduction = (a.rlhf_bias_index - b.rlhf_bias_index) / a.rlhf_bias_index
            row["bias_reduction_pct"] = round(bias_reduction * 100, 1)
            row["grounding_effective"] = bias_reduction > 0
        else:
            row["bias_reduction_pct"] = None
            row["grounding_effective"] = None

        rows.append(row)

    return rows
