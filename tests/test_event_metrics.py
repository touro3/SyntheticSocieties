from metrics.event_metrics import (
    action_counts_from_events,
    action_rate,
    behavior_summary_from_events,
)


def test_action_counts_from_events():
    events = [
        {"action": {"action_type": "work"}, "validation": {"valid": True}},
        {"action": {"action_type": "cooperate"}, "validation": {"valid": True}},
        {"action": {"action_type": "work"}, "validation": {"valid": True}},
    ]

    counts = action_counts_from_events(events)

    assert counts["work"] == 2
    assert counts["cooperate"] == 1


def test_action_rate():
    events = [
        {"action": {"action_type": "work"}, "validation": {"valid": True}},
        {"action": {"action_type": "cooperate"}, "validation": {"valid": True}},
        {"action": {"action_type": "work"}, "validation": {"valid": True}},
        {"action": {"action_type": "save"}, "validation": {"valid": True}},
    ]

    assert action_rate(events, "work") == 0.5
    assert action_rate(events, "cooperate") == 0.25


def test_behavior_summary_from_events():
    events = [
        {"action": {"action_type": "work"}, "validation": {"valid": True}},
        {"action": {"action_type": "cooperate"}, "validation": {"valid": True}},
        {"action": {"action_type": "save"}, "validation": {"valid": True}},
        {"action": {"action_type": "work"}, "validation": {"valid": False}},
    ]

    summary = behavior_summary_from_events(events)

    assert summary["event_action_counts"]["work"] == 2
    assert summary["event_behavior"]["cooperation_rate"] == 0.25
    assert summary["validation_summary"]["valid_actions"] == 3
    assert summary["validation_summary"]["invalid_actions"] == 1
