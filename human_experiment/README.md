# Human Baseline Experiment

Flask + plain HTML/JS implementation of the BGF economic decision game
for human participants. Enables direct human-vs-simulation comparison
.

## Setup

```bash
pip install flask flask-cors
```

## Local run

```bash
# From repo root:
cd human_experiment && python server/server.py
# Open http://localhost:5100 in browser
```

## File layout

```
human_experiment/
  server/server.py      Flask backend (payoffs, session management, CSV logging)
  app/index.html        Single-page game UI
  app/static/game.js    AJAX game client
```

## Data schema

`data/human/responses.csv` — one row per participant-round:

| Column | Description |
|---|---|
| participant_id | UUID assigned at session start |
| round_id | 1-10 |
| action | work / save / cooperate |
| target | neighbor ID (for cooperate only) |
| wealth_after | wealth after action |
| stress_after | stress after action |
| pre_trust | normalised trust score (0-1) |
| pre_risk | normalised risk tolerance (0-1) |
| cooperation_count | cumulative cooperations so far |
| total_rounds | always 10 |

## Analysis

```bash
# With real Prolific data:
python scripts/analyze_human_baseline.py \
  --input-csv data/human/responses.csv \
  --comparison-json analysis/cross_model_results.json

# Dry-run (synthetic):
python scripts/analyze_human_baseline.py --synthetic
```

## Prolific deployment

1. Host the server on a public URL (e.g. ngrok or university VPS).
2. Update `API` in `app/static/game.js` to point to the public URL.
3. Create a Prolific study; set the study URL to `http://<your-host>/`.
4. Participants submit the completion code shown at the end of the game.
5. Download `data/human/responses.csv` after the study closes.

Target: N ≥ 30 participants, 10 rounds each (publication threshold).
