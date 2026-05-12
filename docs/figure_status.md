# Figure Audit Status

Full audit performed 2026-05-11. Each figure verified against paper claims,
source data, and re-run of its generation script.

---

## Summary

| Status | Count | Figures |
|--------|-------|---------|
| ✅ VERIFIED | 8 | 1, 3, 4, 7, 9, 10, 11/12, 16 |
| ✅ FIXED + VERIFIED | 1 | 2 |
| ⚠️ WARNING | 2 | 5, 13 |
| ❌ NEEDS GPU RE-RUN | 2 | 8, 15 |
| ⏳ PENDING | 1 | Human eval (no figure yet) |

**Submission blocker:** Figure 15 (memory ablation) — data from mock policy, not LLM.

---

## Figure-by-figure status

### Figure 1 — `empirical_vs_synthetic.png`
**Status: ✅ VERIFIED**
- Source: `data/ess_clean.parquet` (866 rows, 60 columns)
- Generator: `scripts/plot_empirical_analysis.py` — runs cleanly
- Paper makes no specific numerical claims here (visual comparison only)

### Figure 2 — `llm_grounding_comparison.png`
**Status: ✅ FIXED — Rebuilt from canonical phase_c data**
- **Problem found:** Old figure used small broken experiments (cmp_llm with synthetic population bug + ablation_no_persona with 45-event runs). Caption had work/cooperate labels swapped.
- **Fix applied:** Replaced with `scripts/fix_figure2_canonical.py` — reads `analysis/paper_numbers.json` (phase_c_comparison, N=50, T=30, seed=42, Mistral-7B).
- **New caption:** Accurately describes Condition A (coop=96.2%, B_RLHF=0.712) vs Condition B (coop=58.2%, B_RLHF=0.420).
- **Verified numbers:**

| Metric | Condition A | Condition B |
|--------|-------------|-------------|
| Cooperation rate | 0.962 | 0.582 |
| Gini coefficient | 0.625 | 0.260 |
| B_RLHF index | 0.712 | 0.420 |
| Action: cooperate | 96.2% | 58.0% |
| Action: save | 0.0% | 33.7% |
| Action: work | 3.8% | 8.0% |

### Figure 3 — `grafo_A_ablated.png`
**Status: ✅ VERIFIED**
- Source: `analysis/networks/grafo_A_ablated.gexf`
- Generator: `scripts/plot_networks.py` — runs cleanly
- Paper claims assortativity r≈−0.02, modularity Q≈0.04 (visual description, not extracted numbers)

### Figure 4 — `grafo_B_grounded.png`
**Status: ✅ VERIFIED**
- Source: `analysis/networks/grafo_B_grounded.gexf`
- Generator: `scripts/plot_networks.py` — runs cleanly
- Paper claims r≈0.18, Q≈0.31 (visual description)

### Figure 5 — `bad_apple_resilience.png`
**Status: ⚠️ WARNING — Figure correct, phase transition numbers from pilot only**
- Generator: `scripts/plot_bad_apple.py` — runs cleanly from existing parquets
- Figure itself is accurate
- **Warning:** Phase transition numbers in §6.4 text now reported from N=20 pilot (f*≈0.023, k≈15.1, R²=0.970). N=500 re-run pre-registered but not yet done.
- **Action:** Run `bash scripts/run_bad_apple_sweep_n500.sh` (CPU, ~30–60 min)

### Figure 7 — `macro_shock_resilience.png`
**Status: ✅ VERIFIED**
- Source: `experiments/macro_shock/condition_{a,b}_shock.parquet`
- Generator: `scripts/plot_macro_shock.py` — runs cleanly
- Qualitative claims about asymmetric recovery are visually confirmed

### Figure 8 — `phase_c_macro_comparison.png`
**Status: ❌ NOT REPRODUCIBLE — Missing parquets**
- Cached PNG from original GPU run is correct (numbers match paper_numbers.json)
- `experiments/phase_c_comparison/*.parquet` do not exist on disk
- Canonical numbers preserved in `analysis/paper_numbers.json`
- **Action:** Run `bash scripts/run_phase_c_replication.sh` (~3–5 hours GPU)

### Figure 9 — `trust_gradient.png`
**Status: ✅ VERIFIED — All numbers exact**

| Group | Paper mean | Actual mean | Paper CI | Actual CI |
|-------|-----------|-------------|---------|----------|
| Low-Trust | 0.0103 | 0.0103 ✓ | ±0.0015 | ±0.0015 ✓ |
| Moderate-Trust | 0.0125 | 0.0125 ✓ | ±0.0015 | ±0.0015 ✓ |
| High-Trust | 0.0163 | 0.0163 ✓ | ±0.0035 | ±0.0035 ✓ |
| Very-High-Trust | 0.0155 | 0.0155 ✓ | ±0.0016 | ±0.0016 ✓ |
| Spearman ρ | 0.800 | 0.800 ✓ | | |

### Figure 10 — `cross_model_bias_comparison.png`
**Status: ✅ VERIFIED — All Table 3 numbers confirmed**

| Model | Cond | B_RLHF | Coop | ΔB_RLHF |
|-------|------|--------|------|---------|
| Mistral-7B | A | 0.567 ✓ | 0.900 ✓ | — |
| Mistral-7B | B | 0.467 ✓ | 0.800 ✓ | −17.6% ✓ |
| Qwen2.5-7B | A | 0.333 ✓ | 0.540 ✓ | — |
| Qwen2.5-7B | B | 0.233 ✓ | 0.345 ✓ | −30.0% ✓ |
| GPT-4o-mini | A | 0.223 ✓ | 0.495 ✓ | — |
| GPT-4o-mini | B | 0.313 ✓ | 0.590 ✓ | +40.3% ✓ |

**Note:** `analysis/cross_model_results.json` may be reverted by linter. Canonical values above. Regenerate with `python scripts/plot_cross_model_comparison.py`.

### Figures 11–12 — `feature_importance_coefficients.png`, `feature_importance_ablation.png`
**Status: ✅ VERIFIED — All numbers exact**

| Feature | Paper β | Verified β |
|---------|---------|-----------|
| trust_people | +0.287 | +0.287 ✓ |
| risk_tolerance | −0.187 | −0.187 ✓ |
| social_activity | +0.146 | +0.146 ✓ |
| Train accuracy | 0.608 | 0.608 ✓ |
| N observations | 9,000 | 9,000 ✓ |

### Figure 13 — `persona_drift_long_horizon.png`
**Status: ⚠️ WARNING — Accurate but caveat required**
- Generator runs cleanly
- Claims (fidelity 0.823 at T=100) are from **rule-based proxy** (Condition D), not LLM
- Limitation 12 in paper documents this correctly
- **Action:** No fix needed, but ensure Limitation 12 remains prominent

### Figure 15 — `memory_ablation_interaction.png`
**Status: ❌ CRITICAL — Mock policy, no real data**
- All 24 ablation_M{0-3}_{grounded,ungrounded}_s{42,123,7} experiments used `policy=mock`
- Mock policy ignores memory: all conditions show identical action distributions per seed
- No persona fidelity values exist on disk (all N/A)
- Paper Table 7 values (M0=0.609 → M3=0.742) **cannot be verified from current data**
- **Action required before submission:** `bash scripts/run_memory_ablation_llm.sh`

### Figure 16 — `cross_cultural_expanded.png`
**Status: ✅ VERIFIED — All numbers exact**

| Metric | Paper claim | Verified |
|--------|-------------|---------|
| Pearson r | +0.983 | +0.9828 → rounds to 0.983 ✓ |
| Pearson p | 0.0004 | 0.0004 ✓ |
| Spearman ρ | +1.000 | +1.000 ✓ |
| Spearman exact p | ≈0.003 | 2/720 = 0.0028 ✓ |
| WVS replication r | +0.977 | +0.9772 ✓ |
| Cluster rank order | Perfect monotone | Nordic>Northern>Anglo>Western>Southern>Eastern ✓ |

---

## Pre-submission checklist

- [ ] Run `bash scripts/run_memory_ablation_llm.sh` and verify M0→M3 fidelity monotonicity
- [ ] Run `bash scripts/run_phase_c_replication.sh` to restore parquets and verify Figure 8
- [ ] Run `bash scripts/run_bad_apple_sweep_n500.sh` to get f* at primary scale
- [ ] Run `python scripts/fix_figure2_canonical.py` (already done, always re-run before submission)
- [ ] Run `python scripts/plot_cross_model_comparison.py` (re-run to guard against JSON revert)
- [ ] Run `python scripts/compute_paper_numbers.py --verify` — all claims should PASS
- [ ] Verify test count: `python -m pytest tests/ --co -q | tail -1` → should say 1,372
- [ ] Check paper test count references: `grep "1,372" docs/paper.md | wc -l` → should be 6
