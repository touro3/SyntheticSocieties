# Paper.md vs Experiments Audit — 2026-05-27

Generated alongside `scripts/bootstrap_ci_report.py` re-run.
**Verdict: paper is clean on figures/tables/numbers; two consistency tweaks recommended; one open critical-path experiment is the N=500 LLM A/B cells.**

## ✅ Clean

- **All 11 referenced figures exist on disk** (`analysis/figures/*.png`).
- **All referenced tables exist** (`analysis/tables/*.{json,csv}`).
- **Headline §8.1 numbers reproduced exactly** by `scripts/bootstrap_ci_report.py` over the same 10 mx_A_s* / mx_B_s* runs:
  - A: BRM 0.832 [0.818, 0.845], Gini 0.715, coop 0.455, wealth 174.6
  - B: BRM 0.848 [0.838, 0.859], Gini 0.718, coop 0.461, wealth 177.3
  - A vs B: best p = 0.14 (BRM), Cohen's d = −0.81 (large but not significant)
- Pre-registered falsifications (H2) and IRB-pending status are already disclosed in §8.1 / §4.0 / §8.4.

## ⚠️ Two internal inconsistencies to fix before submission

### 1. Abstract line 11 vs §8.1 line 1061 — "in-flight" vs "stalled"

- **Line 11 (abstract):** "Whether the dissociation closes at higher capacity or larger population scale is the question the *in-flight N=500 LLM-scale run* will settle."
- **Line 1061 (§8.1):** "The Condition-A and Condition-B N=500 cells … were launched on 2026-05-24 but **stalled at `completed_rounds: 0`** and no on-disk progress … the runs must be considered non-completed."

The abstract implies an actively running experiment; §8.1 says it failed. Recommend the abstract say "the N=500 LLM-scale run, currently being re-attempted following the 2026-05-24 stall (§8.1.2)" or similar.

### 2. Mediation §6.X — cells missing

`analysis/mediation_summary.py` emits `_status: "cells_missing"` (persona_only & rag_only seeds 43, 44; rag_only seed 42). The paper text reports "prior preliminary estimates." Reviewers will ask why a §6 result is preliminary at submission.

## 🔬 Open critical-path experiments (paper acknowledges, but reviewers will probe)

| Experiment | Status | Time | Blocking? |
|---|---|---|---|
| **N=500 LLM A/B cells (10 seeds)** | 2 seeds only (s1, s6); §8.1 says stalled | ~5-10 GPU days | The dissociation disambiguator — most important |
| Memory ablation LLM rerun (H8) | mock-policy only on disk | ~6-8 GPU-h | Disambiguates H8 prediction vs measurement |
| Cross-model panel re-execution | Mistral only post-patch | ~2-3 GPU days each | H7 cross-family universality claim |
| Mediation cells s42/43/44 missing factorial | Some empty | ~1 GPU day | §6 mediation table |
| Human eval n=30-50 (Prolific) | Infra ready, IRB pending | ~$120 + 1 week | H6 / B_RLHF* human-calibrated index |

## What you can do without GPU/IRB

1. **Tighten the abstract↔§8.1 stalled-run phrasing** (5 min).
2. **Add a "limitations" subsection** explicitly listing the 5 open experiments — already implicit, but reviewers reward explicit honesty.
3. **Add the bootstrap_ci_report.md results table to §8.1 as a verification block** — duplicates evidence the headline cells reproduce.

## What requires GPU (now running)

- 10-seed sweep launched 2026-05-27 (PID 714999) at N=100 — this is a *re-run* of the existing mx_A/B/D 10-seed cells, not the N=500 LLM disambiguator. Useful as a reproducibility witness; does not advance any of the 5 open experiments above.

To advance the most important blocker (N=500 LLM A/B), the command should be:

```bash
# Stop current sweep, then:
python scripts/run_full_pipeline.py --include-llm \
    --seeds 1,2,3,4,5,6,7,8,9,10 --rounds 30 --agents 500 \
    --skip-existing
```

(~5-10 GPU days on a single P100; the current 4×P100 farm could parallelise this.)
