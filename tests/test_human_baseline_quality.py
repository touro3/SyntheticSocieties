from __future__ import annotations

import pandas as pd

from scripts.analyze_human_baseline import _quality_checks


def test_quality_checks_detect_demo_like_synthetic_pattern():
    rows = []
    for pid in range(1, 6):
        for rnd in range(1, 6):
            rows.append(
                {
                    "participant_id": f"P_{pid:03d}",
                    "round_id": rnd,
                    "action": "work",
                    "wealth_after": 100 + rnd,
                }
            )
    df = pd.DataFrame(rows)
    qc = _quality_checks(df, min_participants=30, min_rounds_per_participant=10)
    assert qc["synthetic_pattern_detected"] is True
    assert qc["passes_min_participants"] is False
    assert qc["passes_min_rounds_per_participant"] is False


def test_quality_checks_accept_publication_like_shape():
    rows = []
    for pid in range(30):
        pid_str = f"5f9a0e2d2c4b{pid:04d}"
        for rnd in range(1, 11):
            rows.append(
                {
                    "participant_id": pid_str,
                    "round_id": rnd,
                    "action": "cooperate" if rnd % 3 == 0 else "work",
                    "wealth_after": 100 + rnd + pid,
                }
            )
    df = pd.DataFrame(rows)
    qc = _quality_checks(df, min_participants=30, min_rounds_per_participant=10)
    assert qc["synthetic_pattern_detected"] is False
    assert qc["passes_min_participants"] is True
    assert qc["passes_min_rounds_per_participant"] is True
    assert qc["duplicate_participant_round_rows"] == 0
