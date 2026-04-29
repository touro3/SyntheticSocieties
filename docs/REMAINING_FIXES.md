# Remaining Fixes Before Capstone Defense

This file lists every outstanding item from the examining-board review that **you** need to execute. Items already landed by Claude in the last pass are noted at the bottom.

Order is **importance × cost**: highest-impact, cheapest-to-run first.

---

## 0. Before anything — sanity check

```bash
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
git status --short | head -40
pytest tests/ -q 2>&1 | tail -20          # confirms the 1,203 tests still pass
```

If anything is broken, fix before running long jobs.

---

## 1. 🔥 Run the 10-seed N=500 LLM A/B extension  (~2 weeks on 2× P100)

This is the only way to promote the paper's headline claims from "pilot-level, directionally consistent" to "confirmatory with tight CIs and formal p-values." Until this run completes, the paper is honest-but-weak; once it completes, the headline sections can be tightened.

Optional but recommended: first pre-download the Mistral weights so the run doesn't wait on HF.

```bash
export BGF_MODEL_CACHE_DIR=/mnt/raid/workspace/lucastourinho/models    # or wherever you have space
python scripts/download_model.py --model mistralai/Mistral-7B-Instruct-v0.3
```

Launch in a detached tmux so it survives SSH disconnect:

```bash
mkdir -p logs
tmux new-session -d -s gpu_ab "bash scripts/launch_gpu_ab.sh 2>&1 | tee logs/gpu_ab_$(date +%Y%m%d_%H%M%S).log"
```

Attach to watch progress:

```bash
tmux attach -t gpu_ab
```

Detach again without killing: `Ctrl-b` then `d`.

Check progress from anywhere:

```bash
tmux capture-pane -t gpu_ab -p | tail -40
ls experiments/ | grep -E "mx_[AB]_s" | wc -l      # should climb to 20 (10 seeds × 2 conditions)
```

When complete, regenerate aggregate tables and figures:

```bash
python scripts/aggregate_seeds.py --experiment-glob "mx_[AB]_s*" --out analysis/tables/ab_10seed.csv
python scripts/plot_trajectories_full.py --seeds 10
python scripts/run_full_pipeline.py --plots-only
```

Then edit `docs/paper.md` §6.1 and the abstract to swap the pilot numbers for the 10-seed confirmatory numbers and add bootstrap 95% CIs. Remove the "pending" language from §8.1.

---

## 2. 🔥 Run the length-matched padded control (closes causal confound)

Still at pilot scale first (N=50, T=30, seed=42) to match `phase_c_comparison`:

```bash
python scripts/run_padded_control.py --agents 50 --rounds 30 --seed 42 \
    --out experiments/padded_control_s42
```

If the padded-A result matches ungrounded-A (same high cooperation, high Gini), then the grounding effect survives length-matching and Limitation 8 can be closed. If padded-A converges toward grounded-B, prompt length is doing real work and the paper's causal framing needs further softening.

Once the 10-seed extension (item 1) is running, also queue the padded control at N=500:

```bash
tmux new-session -d -s padded "python scripts/run_padded_control.py --agents 500 --rounds 30 --seeds 42,123,7 --out experiments/padded_control_primary 2>&1 | tee logs/padded_$(date +%Y%m%d_%H%M%S).log"
```

---

## 3. ✅ DONE — Regenerate `cross_model_bias_comparison.png` to match Table 3

`analysis/tables/cross_model_comparison.csv` does not exist (Qwen/GPT results are in the JSON but only Mistral has multi-seed data). Figure 10 (`cross_model_bias_comparison.png`) has been regenerated directly from Table 3's hardcoded values (Mistral 0.567→0.467, Qwen 0.333→0.233, GPT 0.223→0.313) using a grouped bar chart. PNG at `analysis/figures/cross_model_bias_comparison.png` now shows the correct numbers. Figure 10 caption in §6.6 already notes the source and the previous error.

---

## 4. Regenerate `llm_grounding_comparison.png` (Figure 2) or cite its source

The caption now says the figure is from an ancillary ablation run (`analysis/tables/llm_vs_baselines.csv`) rather than `phase_c_comparison`. Either:

**(a) Regenerate Figure 2 from `phase_c_comparison` so it matches §6.1:**

```bash
python scripts/plot_grounding_comparison.py \
    --cond-a experiments/phase_c_comparison/condition_a_events.parquet \
    --cond-b experiments/phase_c_comparison/condition_b_events.parquet \
    --out analysis/figures/llm_grounding_comparison.png
```

Then update the Figure 2 caption in `docs/paper.md` to drop the source-experiment disclaimer.

**(b) Or leave it as-is**, since the caption already cites its real source. Less work, acceptable.

---

## 5. ✅ DONE — Expand cross-cultural sweep from 3 to 6+ clusters

`analysis/tables/cross_cultural_expanded_correlation.csv` exists with 6 clusters (Eastern, Southern, Western, Anglo, Northern, Nordic). Actual statistics: Pearson r = +0.9828 (p = 0.0004), Spearman ρ = +1.000 (exact two-sided p ≈ 0.003); WVS Wave 7 replication r = +0.977.

**Changes landed in `docs/paper.md`:**
- Abstract: updated cross-cultural sentence to 6-cluster result with formal p-values
- Contribution 6: updated to reflect 6-cluster significance
- §8.3 primary table: expanded from 3 to 6 clusters with CIs from CSV
- §8.3 narrative: upgraded from "directional evidence" to "formally significant" with exact permutation p
- §10 Conclusion item 6: updated to 6-cluster result
- The previous ρ = +0.943 claim is replaced with the actual ρ = +1.000 from the CSV

---

## 6. Stratify §6.5 trust-gradient by social engagement

Current §3.2 finding: trust is *not* a significant predictor in the ESS cooperation model — risk and social engagement are. But §6.5 stratifies by trust anyway and the paper's rescue note is post-hoc. Re-run the gradient stratified by social-engagement quantiles:

```bash
python scripts/run_trust_gradient.py --stratify-by social_meeting_freq --seeds 5 --out analysis/tables/social_engagement_gradient.json
```

If the gradient holds when stratifying by the actual empirically-significant driver, that is the correct experiment to put in §6.5 and the trust stratification becomes ancillary.

---

## 7. ✅ DONE — Reconcile Figure 10 PNG with the updated caption

Covered by Item 3 above. PNG regenerated with the Table 3 values (all 3 models shown side-by-side). Previous item:

```bash
python -c "
import matplotlib.pyplot as plt
import pandas as pd
df = pd.DataFrame([
    {'model':'Mistral-7B','A':0.567,'B':0.467},
    {'model':'Qwen2.5-7B','A':0.333,'B':0.233},
    {'model':'GPT-4o-mini','A':0.223,'B':0.313},
])
fig,ax = plt.subplots(figsize=(8,4))
x = range(len(df))
ax.bar([i-0.2 for i in x], df['A'], width=0.4, label='Condition A')
ax.bar([i+0.2 for i in x], df['B'], width=0.4, label='Condition B')
ax.set_xticks(list(x)); ax.set_xticklabels(df['model'])
ax.set_ylabel('B_RLHF'); ax.legend(); ax.set_title('Cross-model B_RLHF (N=20, T=10)')
plt.tight_layout()
plt.savefig('analysis/figures/cross_model_bias_comparison.png', dpi=150)
print('wrote', 'analysis/figures/cross_model_bias_comparison.png')
"
```

---

## 8. ✅ DONE — Final pass: global consistency check

Run 2026-04-28. All clean:

- **1,272**: zero hits in `docs/paper.md` or `README.md`. Test count is uniformly 1,203 across 91 files.
- **N=500**: all hits are either Condition D (which genuinely ran at N=500 rule-based, 3 seeds) or explicit pre-registration language for the pending LLM extension — no false primary-scale claims.
- **p < 0.05**: two legitimate hits: (a) OLS decay rate across seeds (§6.9, rule-based), (b) the explanation that n=3 cannot reach two-sided p<0.05 — both correct as written.
- **test count**: no stale 1,272+ references remain.

---

## 9. Commit and tag

Once §§1–8 are done:

```bash
git add -p                                           # review each hunk before staging
git commit -m "paper: reconcile scale/statistical claims with actual pilot data

- replace non-existent N=500 LLM primary-scale claims with pilot scale
  (N=50/T=30 single-seed + N=20/T=5 3-seed replication)
- remove mathematically impossible p<0.05 MWU (n=3) and p<0.001 Spearman (n=3) claims
- reconcile Fig 10 caption with Table 3
- reconcile Fig 8 caption with \u00a76.1
- test count 1,272+ -> 1,203 (measured)
- add Limitations 9 and 10 (figure/table inconsistency; statistical power)"

git tag -a capstone-defense-draft -m "State of paper at capstone defense submission"
```

Push only when you are confident:

```bash
git push origin main --tags
```

---

## Already landed by Claude — Pass 1

- Abstract: replaced non-existent "N=500" LLM primary claim with honest pilot numbers (N=50/T=30 single-seed; N=20/T=5 3-seed replication)
- Abstract: removed `p < 0.05` claim tied to 3-seed MWU (mathematically impossible)
- Abstract: reframed cross-cultural Spearman `p < 0.001` to directional evidence (n=3 makes formal testing meaningless)
- Test count: 1,272+ → 1,203 across `docs/paper.md` and `README.md`
- §4 experimental setup table: expanded to show all three pilot scales separately and flagged the stat-test entry
- §5.1 Condition A framing: now honest about two distinct regimes (short-horizon uniform-low vs. long-horizon runaway inequality) rather than single "utopian" narrative
- §6.1: rewritten against actual measured numbers from `phase_c_comparison` and `grounding_comparison_seed_metrics.csv`; statistical claims softened to "consistent direction, formal testing deferred"
- §6.6 GPT-4o-mini discussion: removed comparison to non-existent "primary N=500, T=30" cross-model setup
- §7.1 Discussion opening: causal language downgraded to pilot-level pending padded control and 10-seed extension
- §8.1: reinforced pilot framing
- §10 Conclusion: headline claims rewritten to match actual data
- §2 Contributions list: Contribution 3 (BRM) statistical claim softened; Contribution 6 (cross-cultural) softened
- Fig 2 caption: added source-experiment disclaimer
- Fig 8 caption: corrected from self-inconsistent Cond A narrative
- Fig 10 caption: numbers corrected to match Table 3; note added that cached PNG may need regeneration
- §9 Limitations: added items 9 (figure/table inconsistency) and 10 (statistical power)
- Stale "174+ runs" → "185 runs" to match actual tracker state
- `decision/padded_prompt_builder.py`: added iteration cap guard to prevent infinite loop when `estimate_tokens` is miscalibrated
- `tests/test_llm_backend.py`: added `teardown_method` to `TestLoadLocalFilesOnly` to prevent cross-test singleton pollution

## Already landed by Claude — Pass 2 (2026-04-28)

- **Item 3/7 DONE**: `analysis/figures/cross_model_bias_comparison.png` regenerated with correct Table 3 values (Mistral 0.567→0.467, Qwen 0.333→0.233, GPT 0.223→0.313); grouped bar chart with all 3 models
- **Item 5 DONE**: §8.3 promoted to 6-cluster primary result: table expanded to all 6 clusters with CIs from `cross_cultural_expanded_correlation.csv`; stats corrected to Pearson r=+0.983 (p=0.0004), Spearman ρ=+1.000 (exact p≈0.003); previous spurious ρ=0.943 removed
- **Item 5 DONE**: Abstract, Contribution 6, and §10 Conclusion item 6 all updated to reflect 6-cluster significance with formal p-values
- **Item 8 DONE**: Global consistency check run — zero stale test counts, N=500 refs all legitimate, p<0.05 claims all correct
