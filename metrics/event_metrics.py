from __future__ import annotations

import json
from pathlib import Path


def load_events(events_path: str | Path) -> list[dict]:
    events = []
    with Path(events_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
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
