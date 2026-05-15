# B_RLHF Benchmark Leaderboard

Auto-generated from `benchmark/submissions/*.json`. To regenerate run `python benchmark/build_leaderboard.py`. The protocol that submissions must follow is `benchmark/SPECIFICATION.md` (v1.0).

Sort order: by tier (confirmatory → pilot), then by `delta_B_RLHF_relative` ascending (more negative = stronger bias reduction under grounding).


## Tier: pilot

| Submission | Team | Model | Memory | RAG | B_RLHF (A→B) | BRM (A→B) | ΔB_RLHF | ΔBRM | Schema | Date |
|---|---|---|---|---|---|---|---:|---:|---|---|
| `bgf_paper_pilot` | BGF authors (reference) | mistralai/Mistral-7B-Instruct-v0.3 | M3 | sql,graph | 0.324 → 0.290 | 0.803 → 0.875 | -0.104 | +0.072 | 1.0 | 2026-05-15T02:01:20Z |
