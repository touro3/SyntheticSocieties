# BGF Experimental Hypotheses (H1–H9)

Pre-registered hypotheses for the Behavioral Grounding Framework paper.
Authoritative record: `docs/hypothesis_preregistration.md` (includes deviations #1–#9).

---

## H1: ESS Grounding Improves Behavioural Realism

**Claim**: ΔBRM_composite > 0 — ESS grounding raises the composite Behavioural Realism Metric.

**Metric**: BRM_composite = weighted average of JSD, Gini gap, cooperation accuracy, temporal stability.

**Result**: Directional at N=100 (+0.016, within seed variance; Hedges' g ≈ +0.78, p = 0.089). Magnitude collapsed relative to pilot. Awaits N=500 multi-seed confirmation.

---

## H2: Grounding Reduces RLHF Cooperative Bias

**Claim**: ΔB_RLHF < 0 within Mistral-7B — grounding reduces TV-distance from uniform action prior.

**Metric**: B_RLHF = TV(π, π_uniform) = 0.5 · Σ|π(a) − 1/3|.

**Result**: **Falsified at N=100** (cooperation A=0.455 vs B=0.461, MWU p = 0.91; B_RLHF(A) ≈ B_RLHF(B) ≈ 0.195). At N=500 (T=30 complete): both arms cascade to B_RLHF=0.607/0.627 — grounding does not suppress cascade. Exploratory finding: condB leads condA at every round R1–R30 (single seed, requires multi-seed confirmation).

---

## H3: Gini Falls Within Eurostat Range Under LLM Grounding

**Claim**: Final-round Gini coefficient ∈ Eurostat European empirical range (~0.28–0.38).

**Metric**: Gini coefficient at terminal round.

**Result**: **Falsified at LLM scale** (N=100: Gini ≈ 0.715–0.718; N=500: Gini ≈ 0.965–0.970). Confirmed for rule-based ESS policy (Condition D, N=500, 10 seeds: Gini = 0.325 BCa [0.324, 0.325]).

---

## H4: Trust-Band Cooperation Rank-Orders with ESS Trust

**Claim**: BRM or cooperation rate rank-orders with ESS trust bands — higher trust → higher cooperation.

**Metric**: Rank ordering across 4 trust bands; Spearman ρ across seeds.

**Result**: Directional (ρ = +0.800, p = 0.167 at group-level, n=4); significant at seed-level (continuous n=20, ρ = +0.781, p < 0.0001, post-hoc design — see Limitation 17).

---

## H5: Cross-Cultural Trust Gradient Recovered via Grounding

**Claim**: Simulated cooperation rank-orders with ESS interpersonal trust across six cultural clusters.

**Metric**: Spearman ρ across 6 ESS cultural clusters.

**Result**: **Confirmed (rule-based proxy)** — Spearman ρ = +1.000 (exact p ≈ 0.003); Pearson r = +0.983; WVS Wave 7 replication r = +0.977. LLM-scale replication pending.

---

## H6: Bad-Apple Effect Localises at Low Adversarial Fraction

**Claim**: Adversarial injection produces a phase transition; inflection point f* < 10% (below Nowak & May 1992 prediction).

**Metric**: Sigmoid inflection f* on Gini/cooperation vs adversarial fraction; R² > 0.85.

**Result**: **Partially confirmed (rule-based scale only)**. N=20: f*=0.023, k=15.1, R²=0.97 (Gini increases). N=500 (2026-06-05): f*=0.041, k=5.2, R²=0.996 — **scale reversal**: Gini *decreases* at N=500 (cooperation suppression equalises outcomes). f* remains well below 10% at both scales. LLM-scale sweep pending.

---

## H7: RLHF Cooperative Bias Generalises Across LLM Families

**Claim**: B_RLHF(A) > 0 for all tested instruction-tuned LLMs; grounding reduces B_RLHF in the majority of families.

**Metric**: B_RLHF per model × condition pair.

**Result**: **Confirmed for existence** (all three tested models exhibit B_RLHF > 0 in Condition A). Grounding reduces B_RLHF for Mistral-7B (−17.6%) and Qwen2.5-7B (−30.0%) but increases it for GPT-4o-mini (+40.3%) — alignment methodology is a moderating variable. Full multi-seed cross-model panel requires re-execution at patched-code scale.

---

## H8: Memory Depth Monotonically Increases Cooperation Fidelity (M0→M3)

**Claim**: Under ESS grounding, cooperation increases monotonically with memory depth: M0 < M1 < M2 < M3.

**Metric**: Terminal-round cooperation rate across memory levels {M0, M1, M2, M3} × {grounded, ungrounded}.

**Result**: **FALSIFIED for both arms** (v2 re-run, 24/24 cells, N=20, T=10, Mistral-7B-Instruct-v0.3, 2026-06-05).
- **Grounded arm**: M0G(0.583) > M1G(0.367) = M2G(0.367) = M3G(0.367) — monotone decrease at M0→M1, flat thereafter.
- **Ungrounded arm**: M0U(0.417) < M1U(0.633) = M2U(0.633) > M3U(0.450) — inverted-U, non-monotone.
- M3G does NOT exceed M0G; full memory does not rescue hypothesis.
- B_RLHF global minimum at M3G (0.072±0.034). M3U converges to 0.450±0.000 across all 3 seeds (RLHF attractor stabilisation).
- Pre-registration deviation #9 logged in `docs/hypothesis_preregistration.md`.

---

## H9: Simulated Cooperation Matches Cross-Cultural Lab Benchmark

**Claim**: Simulated cluster cooperation rank-orders with Herrmann, Thöni & Gächter (2008) PGG per-city contributions — an independent behavioural benchmark never ingested by BGF.

**Metric**: Spearman ρ between simulated cooperation and PGG contributions across 6 city-cluster pairs.

**Result**: **Confirmed at per-test α=0.05** — Spearman ρ = +0.886 (exact p = 0.033), Pearson r = +0.899 (p = 0.015). Does **not** survive Holm-Bonferroni family-wise correction (α/9 ≈ 0.0056). LLM-scale replication pending.
