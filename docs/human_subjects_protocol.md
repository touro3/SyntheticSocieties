# Human Baseline Protocol (Prolific)

This protocol defines a minimal human-subject baseline for BGF realism claims.
Goal: compare real participant behavior in the same `{work, save, cooperate}` game
against Condition A (ungrounded) and Condition B (grounded).

## 1) Study Design

- Platform: Prolific
- Sample size: `n = 30–50` participants
- Session length: 8–12 minutes
- Rounds: 10
- Action space per round: `work`, `save`, `cooperate`
- Suggested compensation: `$3.00–$4.00` base (+ optional performance bonus)

## 2) Inclusion / Quality

- Adults (18+), English proficient
- Completion-time filter: exclude < 1/3 median time
- Attention checks: at least 1 instructional manipulation check
- Exclude duplicate Prolific IDs

## 3) Data Schema

Save one row per participant per round as CSV with these required columns:

- `participant_id`
- `round_id` (1-indexed integer)
- `action` (`work|save|cooperate`)
- `wealth_after` (numeric wealth after round resolution)

Optional columns:

- `condition` (if running multiple experimental variants)
- `reaction_time_ms`
- `country`, `age_band`, `gender`

Important:
- Do not use synthetic/demo rows for publication claims.
- `scripts/analyze_human_baseline.py` now rejects likely synthetic/demo-like data
  unless `--allow-synthetic` is explicitly passed for local smoke tests.

## 4) Primary Endpoints

- Human cooperation rate
- Human final-round wealth Gini
- Human RLHF-bias proxy `B_RLHF = TV(pi_actions, uniform)`

Use the same definitions as simulation metrics to keep direct comparability.

## 5) Analysis Command

```bash
source venv/bin/activate
python scripts/analyze_human_baseline.py \
  --input-csv data/human/prolific_round_data.csv \
  --output-json analysis/tables/human_baseline_metrics.json \
  --comparison-json analysis/tables/human_vs_simulation_reference.json \
  --output-markdown analysis/reports/human_baseline_comparison.md
```

Publication defaults enforced by the script:
- `min_participants >= 30`
- `min_rounds_per_participant >= 10`
- no duplicate `participant_id + round_id` rows

For pilot testing only:

```bash
python scripts/analyze_human_baseline.py \
  --input-csv data/human/prolific_round_data.csv \
  --allow-noncompliant --allow-synthetic
```

## 6) Comparison JSON Format

`analysis/tables/human_vs_simulation_reference.json` should be:

```json
{
  "Condition A (ungrounded)": {
    "gini": 0.08,
    "cooperation_rate": 0.96,
    "b_rlhf": 0.52
  },
  "Condition B (BGF grounded)": {
    "gini": 0.31,
    "cooperation_rate": 0.58,
    "b_rlhf": 0.21
  }
}
```

## 7) Reporting Template

Target table in paper:

| Condition | Gini | Coop Rate | B_RLHF |
|---|---:|---:|---:|
| Real humans | `...` | `...` | `...` |
| Condition B (BGF) | `...` | `...` | `...` |
| Condition A (ungrounded) | `...` | `...` | `...` |
