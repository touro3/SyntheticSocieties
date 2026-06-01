from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_events(events_path: str | Path) -> list[dict]:
    """Load events from a JSONL file, transparently concatenating rotated shards.

    Accepts either the active ``events.jsonl`` path or its parent directory.
    When the active file rotates (see ``bgf_logging.event_logger``), shards
    are written as ``events.0001.jsonl``, ``events.0002.jsonl``, … with the
    tail in ``events.jsonl``. This function reads all shards in shard order,
    then the active file, so downstream metrics see the full run regardless
    of how many rotations occurred.
    """
    p = Path(events_path)
    if p.is_dir():
        base_dir = p
        active = base_dir / "events.jsonl"
    else:
        base_dir = p.parent
        active = p

    shards = sorted(base_dir.glob("events.[0-9]*.jsonl"))
    files: list[Path] = list(shards)
    if active.exists():
        files.append(active)

    events: list[dict] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return events


def action_counts_from_events(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for event in events:
        action_type = event.get("action", {}).get("action_type")
        if action_type is None:
            continue
        counts[action_type] = counts.get(action_type, 0) + 1

    return counts


def validation_counts(events: list[dict]) -> dict[str, int]:
    valid = 0
    invalid = 0

    for event in events:
        validation = event.get("validation", {})
        if validation.get("valid") is True:
            valid += 1
        else:
            invalid += 1

    return {
        "valid_actions": valid,
        "invalid_actions": invalid,
    }


def action_rate(events: list[dict], action_name: str) -> float:
    counts = action_counts_from_events(events)
    total = sum(counts.values())

    if total == 0:
        return 0.0

    return counts.get(action_name, 0) / total


def behavior_summary_from_events(events: list[dict]) -> dict:
    counts = action_counts_from_events(events)
    validation = validation_counts(events)
    total = sum(counts.values())

    if total == 0:
        total = 1

    return {
        "event_action_counts": counts,
        "event_behavior": {
            "work_rate": counts.get("work", 0) / total,
            "save_rate": counts.get("save", 0) / total,
            "cooperation_rate": counts.get("cooperate", 0) / total,
            "rejected_action_rate": validation["invalid_actions"] / total,
        },
        "validation_summary": validation,
    }


def temporal_stability(events: list[dict]) -> dict:
    """
    Compute temporal stability of agent behavior across rounds.

    Groups events by round, computes the action distribution per round,
    then measures Jensen–Shannon divergence between consecutive rounds.

    Returns:
        Dict with per-round distributions, round-to-round JSD values,
        and an overall mean stability score (lower JSD = more stable).
    """
    from scipy.stats import entropy

    # Group events by round
    round_actions: dict[int, list[str]] = defaultdict(list)
    for event in events:
        round_id = event.get("round_id")
        action_type = event.get("action", {}).get("action_type")
        if round_id is not None and action_type is not None:
            round_actions[round_id].append(action_type)

    if len(round_actions) < 2:
        return {
            "n_rounds": len(round_actions),
            "round_distributions": {},
            "round_jsd": [],
            "mean_jsd": 0.0,
            "stable": True,
        }

    # All possible actions
    all_actions = sorted(set(a for actions in round_actions.values() for a in actions))

    # Compute per-round distributions
    round_dists: dict[int, dict[str, float]] = {}
    for round_id in sorted(round_actions.keys()):
        actions = round_actions[round_id]
        total = len(actions)
        dist = {a: actions.count(a) / total for a in all_actions}
        round_dists[round_id] = dist

    # Compute JSD between consecutive rounds
    sorted_rounds = sorted(round_dists.keys())
    jsd_values = []
    eps = 1e-10

    for i in range(1, len(sorted_rounds)):
        p = np.array([round_dists[sorted_rounds[i - 1]].get(a, 0) + eps for a in all_actions])
        q = np.array([round_dists[sorted_rounds[i]].get(a, 0) + eps for a in all_actions])
        p = p / p.sum()
        q = q / q.sum()
        m = (p + q) / 2
        # base=2 → JSD in [0,1] bits, matches metrics.distribution.
        jsd = float(entropy(m, base=2) - (entropy(p, base=2) + entropy(q, base=2)) / 2)
        jsd_values.append(jsd)

    mean_jsd = float(np.mean(jsd_values)) if jsd_values else 0.0

    return {
        "n_rounds": len(round_actions),
        "round_distributions": {str(k): v for k, v in round_dists.items()},
        "round_jsd": jsd_values,
        "mean_jsd": mean_jsd,
        "stable": mean_jsd < 0.1,  # Threshold for "stable" behavior
    }
