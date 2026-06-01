# Appendix A — Forensic Audit Trail

This appendix preserves the forensic chain that validates the patched-code results in the main paper (`docs/paper.md`). It is not required reading for the main scientific argument; it exists so an auditor can verify which prior numbers were withdrawn, why, and what replaced them. All claims in the main paper rest on the patched-code reality documented here; everything in this appendix has either been superseded by a clean re-execution or is reported as a historical artefact for traceability.

The appendix is organised into five sections that mirror the inline forensic blocks removed from the main text:

- **§A.1** — Infrastructure bugs (L-1 PromptLogger, L-2 kernel batched-path `rag_context`, C-1 env-var batch-size override) and the patched-code seed-42 re-execution.
- **§A.2** — Withdrawn pilot B_RLHF values (0.712 → 0.420), the TV-bound arithmetic that proves they are inadmissible, and the recomputed admissible value (0.2347 for both arms at seed 42).
- **§A.3** — Withdrawn cross-model rows for Qwen2.5-7B and GPT-4o-mini in Table 3; on-disk-only Mistral row reconciliation.
- **§A.4** — Pilot-vs-extension attribution gap: five axes of difference between the single-seed N=50 pilot and the §8.1 10-seed N=100 extension, plus the commit-by-commit "first/second rigor pass" changelog.
- **§A.5** — Historical Figure 17 (pre-§8.1 forest plot) and pre-§8.1 Table 3 prose, retained as historical snapshots.

Forensic JSON files: `analysis/reports/b_rlhf_pilot_forensic.json` (§A.2) and `analysis/reports/cross_model_b_rlhf_forensic.json` (§A.3).

---

## §A.1 Infrastructure Bugs (L-1, L-2, C-1) and the Patched-Code Replication

Three logging- and inference-infrastructure bugs were discovered on 2026-05-23 during the gpu3 audit re-run and are disclosed here in full. All three are now patched (commits subsequent to `c73554e`); the patches do not alter scientific code paths (population synthesis, policy decision, payoff computation, metric aggregation) — they restore auditability of, and remove a confound from, the pre-existing pipeline. The §8.1 N=100 cells (`mx_{A,B}_n100_s{1..10}`) and the in-flight N=500 extension (`mx_{A,B}_n500_s{1..10}`) all run on the patched code.

**Bug L-1 (PromptLogger off-by-one).** `bgf_logging/prompt_logger.py:133` previously read `if self._total_calls % self._sample_every != 1: return`. With the default `sample_rate=1.0` ⇒ `_sample_every=1`, the condition `N % 1 == 0 ≠ 1` holds for every record, so the logger silently discarded every prompt write. The companion `bgf_logging/memory_diagnostics.py:87` carries the correct guard `if self._sample_every > 1 and …`, demonstrating the intended semantics. **Consequence for prior data:** `prompts.jsonl` is 0 bytes in every Condition A/B run produced by this codebase up to 2026-05-23 (verified against 105 experiments on disk). This means *the `rag_context` field that would have enabled post-hoc verification of RAG presence was never written*. RAG presence in prior Condition B runs is inferable only from code-path inspection (`scripts/run_config_simulation.py:243-250` unconditionally instantiates `GraphRAG` + `SQLRAG` when `policy.type == "llm"`), not from on-disk artefacts. Patch: add the `_sample_every > 1` guard.

**Bug L-2 (kernel batched-path drops `rag_context`).** `simulation/kernel.py` `run_round_batched()` calls `policy.prompt_logger.log(...)` without the optional `rag_context=` keyword, even though the kernel does compute `social_context` / `pop_context` per agent immediately above. The single-agent path in `decision/llm_policy.py:127-133` passes the field correctly. **Consequence:** even after L-1 is fixed, the `rag_context` JSON field is `null` for every LLM run that uses batched inference (i.e. all runs from this codebase). Patch: thread the cached `social_context` / `pop_context` from `agent_data[i]` into the `log()` call.

**Bug C-1 (`BGF_MAX_BATCH_SIZE` env override silently lost).** `scripts/run_config_simulation.py:231-232` unconditionally overwrote `backend._max_batch_size` from `llm_cfg["max_batch_size"]` *after* the backend constructor read the env var. On the 16 GB P100s used in this work, the canonical configs ship `max_batch_size: 16`, which immediately OOMs and falls into the kernel's halve-retry loop (chunk 16 → 8 → 4) on every batched inference. This wastes throughput and creates non-stationary latency in long-horizon runs, but does not change the policy decisions (the halve-retry succeeds before the action is committed). Patch: let the explicit env var win over the config default.

**Patched-code replication (`cmp_llm_s42_condB`).** With L-1, L-2, C-1 patched, we re-ran the canonical T=30 / N=50 / seed=42 Condition B configuration on a single 16 GB P100 (Mistral-7B-Instruct-v0.3, 4-bit quantisation, `BGF_MAX_BATCH_SIZE=4`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `temperature=0.7`, `memory_window=5`). Wall-clock: 10 107.7 s ≈ 2.81 h. The run produces 1 500 agent-decision events (50 agents × 30 rounds). The empirical run-average action distribution (extracted from `experiments/cmp_llm_s42_condB/events.jsonl`):

| Action | Run-average frequency π | Count (agent-rounds) |
|--------|-------------------------|----------------------|
| cooperate | **0.539** | 809 |
| work      | **0.362** | 543 |
| save      | **0.099** | 148 |
| *parse failures / fallback actions* | 0.000 | 0 / 1 500 |

The corrected RLHF Bias Index is `B_RLHF = TV(π, π_uniform) = ½ · Σ_a |π(a) − ⅓| = `**`0.2347`**, well inside the theoretical interval `[0, 2/3]` and consistent with the cooperation-rate triplet. A bit-level diff of the seed-42 ablation_level=0 (`cmp_llm_s42_condA`) and ablation_level=5 (`cmp_llm_s42_condB`) re-executions shows their `events.jsonl` files are **byte-identical** (differing configs only in `experiment_id` and `llm.ablation_level`); B_RLHF(A) = B_RLHF(B) = 0.2347 at this seed, and the action triplet is identical across conditions. This is the seed-level analogue of the §8.1 N=100 null and is captured in `analysis/reports/b_rlhf_pilot_forensic.json`. The corresponding final-round Gini coefficient is `0.380` and final-round mean wealth `457.6`.

**Substantive observation — temporal phase shift in cooperation.** The run-average frequency of 0.539 cooperation masks a strongly non-stationary trajectory. Cooperation rises monotonically across rounds:

| Round band | Cooperation rate | Dominant action |
|------------|------------------|-----------------|
| R 1 – R 5  | 0.024 (mean)     | work (≥ 88 % every round) |
| R 6 – R 10 | 0.212            | mixed (work/save/cooperate oscillate) |
| R 11 – R 15 | 0.504           | cooperate emerging |
| R 16 – R 20 | 0.768           | cooperate dominant |
| R 21 – R 25 | 0.876           | cooperate ≥ 0.86 every round |
| R 26 – R 30 | 0.908           | cooperate ≥ 0.90 every round |

This is a **first-passage transition**, not a stationary distribution: with uniform initial wealth `w_0 = 50.0` every agent is forced to work, and the cooperative attractor is reached only after capital accumulates beyond the minimum cooperate-amount of 5 wealth units. Reporting *run-averaged* π conflates the work-dominated warm-up regime with the cooperate-dominated steady state. Future Condition B summaries should report **(a) steady-state π** (e.g. averaged over R 21 – R 30, here giving cooperation ≈ 0.89 and `B_RLHF_ss = 0.4`) **and (b) the run-average π** as separate observables.

---

## §A.2 Withdrawn Pilot B_RLHF Values (T=30 single seed)

The pre-patch single-seed T=30 pilot (`phase_c_comparison`) cited `B_RLHF(A) = 0.712` and `B_RLHF(B) = 0.420`. Both values are **mathematically impossible** under the TV bound proved in §3.2.1 Proposition 1: for any 3-action distribution, `B_RLHF = TV(π, π_uniform) ≤ 1 − 1/|A| = 2/3 ≈ 0.667`.

The B_RLHF(A) value (0.712) violates the absolute bound. The B_RLHF(B) value (0.420) is below the absolute bound but violates the *conditional* bound at the observed cooperation rate (≈ 0.58): the maximum achievable TV at coop = 0.582 is `0.333` (verified analytically), so 0.420 cannot be reconciled with the cited action distribution.

The original `experiments/phase_c_comparison/` directory is not available on the current host, so a direct recompute on the original event log is not possible; **both values are withdrawn**. The patched-code re-executions (§A.1, `cmp_llm_s42_condA` and `cmp_llm_s42_condB`) yield `B_RLHF = 0.2347` for *both* conditions — consistent with the TV bound and with the §8.1 N=100 null (both arms statistically indistinguishable).

Forensic JSON: `analysis/reports/b_rlhf_pilot_forensic.json` (per-row TV-bound check, replacement values, and the byte-identity diff between the two seed-42 re-executions).

This withdrawal supersedes audit rows **A.13** and **A.14** by replacement. The previously reported pilot magnitudes (0.712 → 0.420, −41 %) should not be cited in any downstream analysis.

---

## §A.3 Withdrawn Cross-Model Rows (Qwen2.5-7B, GPT-4o-mini)

A second-pass audit (2026-05-24) of §6.6 Table 3 found that the only on-disk cross-model artefact is `analysis/cross_model_results.json`, which contains **Mistral-7B only** (2 A-runs + 2 B-runs at N=20, T=10). The Qwen2.5-7B and GPT-4o-mini rows of the historical Table 3 have no on-disk source of any kind (no event log, no per-cell summary).

| Model | Cond. | Historical Coop | Historical Gini | Historical B_RLHF | Status |
|-------|-------|-----------------|------------------|-------------------|--------|
| Mistral-7B-Instruct-v0.3 | A | 0.900 | 0.253 | 0.567 | On-disk JSON shows mean coop = 0.588, B_RLHF = 0.254 across 2 runs — disagrees with the historical narrative magnitude. |
| Mistral-7B-Instruct-v0.3 | B | 0.800 | 0.153 | 0.467 (−17.6 %) | On-disk JSON shows mean coop = 0.351, B_RLHF = 0.039 — implies ≈ −85 % reduction at N=20, much larger than the −17.6 % narrative figure. |
| Qwen2.5-7B-Instruct | A | 0.540 | 0.047 | 0.333 | **Withdrawn** — no on-disk artefact. |
| Qwen2.5-7B-Instruct | B | 0.345 | 0.141 | 0.233 (−30.0 %) | **Withdrawn**. |
| GPT-4o-mini | A | 0.495 | 0.309 | 0.223 | **Withdrawn** — no on-disk artefact. |
| GPT-4o-mini | B | 0.590 | 0.204 | 0.313 (+40.3 % inversion) | **Withdrawn**. |

**Internal consistency check (audit).** Under the equal work/save split assumption, `B_RLHF = ½ · Σ |π(a) − 1/3|` simplifies. Mistral-A (historical 0.900 cooperation, equal w/s split) implies TV = 0.567, which matches the table — so the historical Mistral row is internally consistent under that assumption (just inconsistent with the on-disk JSON, which records different action triplets). For Qwen-A (coop = 0.540) the equal-split TV is 0.207 vs claimed 0.333 (Δ = 0.126); for GPT-4o-mini-A (coop = 0.495) the equal-split TV is 0.162 vs claimed 0.223 (Δ = 0.061). The Qwen and GPT-4o-mini claimed B_RLHF values therefore required unrecorded non-equal action triplets to be valid — and without the event logs those triplets are unrecoverable.

**Required action before any further cross-model claim:** re-execute `scripts/run_cross_model_comparison.py` for all three model families under the patched code (post commit c73554e) to regenerate auditable per-cell artefacts. Until then, only the rule-based cross-cultural result (main paper §8.3) and the Condition D Gini anchor (main paper §8.2) are defensible cross-system findings; the cross-model panel is a *historical* result with reduced evidential weight.

The current main-paper §6.6 reports only the Mistral on-disk numbers (with the historical −17.6 % narrative removed); the cross-family heterogeneity framing in §6.6.1 is reduced from "alignment methodology as a moderating variable (finding)" to "an open question pending re-execution."

Forensic JSON: `analysis/reports/cross_model_b_rlhf_forensic.json`.

---

## §A.4 Pilot-vs-Extension Attribution Gap

The §8.1 10-seed N=100 confirmatory extension does not reproduce the pilot magnitudes on cooperation rate, Gini, or mean wealth. We are unable, on the basis of these 20 runs alone, to attribute the gap to a single cause. The relevant axes that differ between the two are:

1. **Population scale.** Pilot N=50 (single seed) and short-horizon N=20 vs. extension N=100. The §8.1 protocol block as previously written referenced "N=500" in places and `--agents 50` in the launch command — neither matches what was actually executed. The N=500 LLM-scale run is the next data point; whether the pilot's effect re-emerges at N=500 is the open question §A.5 also addresses.
2. **Seed selection.** Pilot used seeds {42, 123, 7}; extension used seeds 1–10. The pilot seed set was selected pre-registration; we do not believe the gap is driven by seed cherry-picking but cannot rule it out without running the original three seeds at the new code revision.
3. **Code state.** Bugs L-1 (PromptLogger off-by-one) and L-2 (kernel batched path dropping `rag_context`) were patched between the pilot runs and these extension runs (§A.1). L-1/L-2 affected only the audit logs, not the inference path — Cond B prompts were still receiving RAG context at decision time via the unconditional construction in `scripts/run_config_simulation.py:285-304` — so they cannot explain Cond B regressing toward Cond A. They can, however, no longer be invoked to explain Cond A *not* exhibiting the 96 % cooperative collapse: that pilot behaviour now has no observed analogue at N=100. Bug C-1 (env-var batch-size override) was also patched; it affected throughput and OOM-thrashing only.
4. **OOM-driven batch halving.** The runs logged repeated `CUDA OOM on sub-batch ... halving to 6/7` events on both arms. The system's documented behaviour is to retry at smaller chunk sizes; no fallback action records were written (`fallback==0` across all 30,000 action events per cell), so decisions were produced by the LLM rather than by `LLMPolicyBase._fallback_action`. We therefore do not believe OOM-fallback collapsed the contrast, but a per-call latency-vs-temperature interaction (e.g., reduced effective sampling diversity at the smallest chunk sizes) is not ruled out.
5. **Empirical-population NaN substitution.** Every cell logged ~17–29 / 100 agents with NaN `income_decile` substituted by a default — "empirical marginal for this field is distorted". This affects both arms equally and is therefore unlikely to drive an A vs B gap, but it does indicate the empirical-population draws under-represent the income tails relative to a clean ESS sample.

**Rigor-pass changelog (commit-by-commit).** Two distinct revision passes were applied on 2026-05-24 ahead of this appendix being split out:

- **First rigor pass** (commit `0dc9427`) edited the abstract, Key Claim 2, the §4.1 power analysis, the §6.1 statistical-evidence + composite-BRM blocks, the Figure 2 / 8 / 17 captions, the §6.2 Bayesian posterior + meta-analytic synthesis paragraph, §7.1, Limitation 10, and the §10 conclusion to flag the non-replication and reframe the pilot magnitudes (96.2 % → 58.2 % cooperation, 0.625 → 0.260 Gini, ≈ 2.7× BRM ratio) as descriptive statistics of single runs rather than as confirmed treatment effects. The "10-seed extension transforms §§6.1–6.4 from pilot-level to confirmatory-level evidence" framing previously used in §8.1 was withdrawn at this pass.
- **Second rigor pass** (same date) additionally edited §6.6 Table 3 + surrounding prose + §6.6.1 + Figure 10 caption + the Conjecture support paragraph + the §10 conclusion opening + §10 contributions items 2 and 5 + the abstract §1 narrative cross-model sentence + §8.5 Table 7 reader callout + Figure 17 caption (now points to Table 8 only) + Limitation 14 mitigation note + Limitation 15 scope extension + new Limitation 18 (cross-model provenance), and updated audit-trail counts (tests: 1,441 → 1,538; experiments: 192 → 236; LOC: 71,500 → 72,475) in the abstract, §3.12.1, §4 setup table, and §10 contributions item 1.

The current main paper has been further restructured to delete the inline forensic apologetics and move the content to this appendix; the changelog above documents the intermediate state.

---

## §A.5 Historical Figure 17 and Pre-§8.1 Table 3 Prose

**Figure 17 (historical, pre-§8.1; retained for traceability).** The cached `analysis/figures/forest_plot.png` was generated by `analysis/forest_plot.py` from `analysis/tables/forest_plot.json` at pilot scale, before the §8.1 extension. It shows pre-extension point estimates that are now superseded: H1 (+0.235 pre-§8.1) has shrunk to +0.016 at N=100; H2 is falsified at N=100; the Mistral-7B / Qwen / GPT-4o-mini cross-model row depends on values whose on-disk provenance is now in doubt (§A.3). The canonical post-§8.1 hypothesis register is **Table 8 in main paper §8.1** — defer to Table 8 for any current-status claim. The cached PNG is preserved here for historical reference rather than as evidence; an updated render will follow an end-to-end re-execution of the cross-model and N=500 LLM-scale runs.

**Pre-§8.1 Table 3 narrative (historical).** Two paragraphs originally accompanied the historical Table 3 in main paper §6.6 and have been removed from the main text:

> **(Historical, pre-second-pass-audit.)** Mistral-7B and Qwen2.5-7B were reported to confirm H2 directionally, with Qwen showing a stronger bias reduction (−30.0 %) than Mistral (−17.6 %). Under the on-disk artefact for Mistral the reduction is much larger (≈ −85 % at N=20), but neither magnitude is reconciled with the §8.1 N=100 null (Δ_coop ≈ 0.006, MWU p = 0.91). The Qwen2.5-7B narrative magnitude is currently unverifiable.

> **(Historical, pre-second-pass-audit.)** GPT-4o-mini was reported to invert the sign (+40.3 %), motivating §6.6.1's "alignment methodology as a moderating variable" framing. With no on-disk artefact, the inversion is currently unverifiable; the §6.6.1 finding therefore stands or falls on a future cross-model re-execution. Until then, the "non-universal grounding response" claim should be read as *the most parsimonious explanation given the heterogeneity reported in the historical narrative* rather than as an audit-verified finding.

These paragraphs are retained here verbatim because subsequent papers, the LaTeX stub at `paper/paper_neurips.tex`, and external references may cite the historical magnitudes; this appendix is the canonical record of *why* they no longer appear in the main paper.

---

*Appendix end. The remainder of this paper (`docs/paper.md`) reports only the patched-code reality.*
