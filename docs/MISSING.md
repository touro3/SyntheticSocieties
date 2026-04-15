# What Is Missing

> Items needed for **conference/journal acceptance** (AAMAS 2026 primary, NeurIPS 2026 fallback).
> The capstone itself is defensible now — everything below is post-capstone research.

---

## CRITICAL — blocks acceptance

### 1. 10-seed A/B LLM comparison (GPU, ~2 weeks)
Current results use 3 seeds. Reviewers will flag this immediately as insufficient statistical power.

**Command:**
```bash
source venv/bin/activate
python scripts/run_experiment_matrix.py --include-llm \
  --conditions A B --seeds 1..10 --rounds 30 --agents 100
```

**Required config fix already in place:** `population.source=empirical` is now set for both A and B.
**What it unlocks:** Bootstrap 95% CIs on all primary metrics (BRM, B_RLHF, Gini, cooperation rate).

---

### 2. Condition D at full scale (GPU, ~3 days)
`decision/rule_based_ess_policy.py` is implemented but never run at paper scale.
Needed to answer the core question: "why use LLMs at all over a pure rule-based ESS policy?"

**Command:**
```bash
python scripts/run_full_pipeline.py --policy rule_based_ess \
  --seeds 42,123,7 --rounds 30 --agents 500
```

**What it unlocks:** Table 3 row (Condition D) with real BRM/B_RLHF numbers.

---

### 3. Padded control run — Phase 29.1 (GPU, ~1 day)
Infrastructure is complete (`scripts/run_padded_control.py`, `PaddedAblationPolicy`).
Needed to rule out that grounding effects are driven by prompt length, not content.

**Command:**
```bash
python scripts/run_padded_control.py --seeds 42,123,7 --agents 500 --rounds 30
```

**What it unlocks:** Confound control for prompt-length artifact — critical for rigorous Condition A vs B comparison.

---

## HIGH VALUE — unlocks venue competitiveness

### 4. Human evaluation (1 week, ~$100–200 budget)
n = 30–50 Prolific participants rating realism of sampled Condition A vs. B agent behaviors.
Protocol is already designed: `docs/human_subjects_protocol.md`.

**What it unlocks:** "Validated by human judges" — the strongest possible evidence for the central claim; no metric can substitute.

---

### 5. Venue-specific LaTeX formatting (2 days writing)
Paper is in Markdown (`docs/paper.md`). Needs to be converted to LaTeX and cut to venue page limits.

| Venue | Page limit | Framing |
|-------|------------|---------|
| AAMAS 2026 | 8 pages | Multi-agent systems |
| NeurIPS 2026 | 9 pages | LLM behavior + benchmark |
| ACL 2026 | 8 pages | LLM behavioral analysis |

**Primary venue: AAMAS 2026.**

---

### 6. External reproducibility validation
Recruit 2+ external collaborators to clone the repo and run `reproduce_paper.sh` on fresh hardware.
Remaining blockers to fix: HuggingFace cache path hardcoded to `/mnt/raid/workspace/...`, GPU memory documentation.

---

## SECONDARY — nice-to-have

### 7. 70B model validation
Run grounding experiment with Llama-3.1-70B or Mistral-Large (if access available).
Shows that grounding works at frontier model scale, not just 7B.

---

## Status snapshot (2026-04-15)

| Item | Status |
|------|--------|
| BGF framework + 921+ tests | ✅ Done |
| Conditions A/B (3 seeds, rule-based) | ✅ Done |
| Cross-cultural LLM validation (10 seeds/cluster) | ✅ Done — Spearman ρ = 1.000 |
| Feature importance (9,000 obs.) | ✅ Done |
| Long-horizon T=100 analysis | ✅ Done — 1.8× decay gap |
| Policy intervention sweep | ✅ Done — δ=20% → +4.5pp |
| Memory ablation M0–M3 | ✅ Done |
| Phase transitions (bad apple, shock, topology) | ✅ Done |
| Cross-model (Mistral/Qwen/GPT-4o-mini) | ✅ Done |
| Empirical cooperation model (logistic, AUC=0.640) | ✅ Done |
| 10-seed A/B LLM comparison | ❌ GPU pending |
| Condition D full-scale run | ❌ GPU pending |
| Padded control (Phase 29.1) | ❌ GPU pending |
| Human evaluation (Prolific) | ❌ Budget + recruitment |
| LaTeX venue formatting | ❌ Writing work |
| External reproducibility | ❌ Collaborator recruitment |
