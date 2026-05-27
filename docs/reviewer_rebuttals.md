# Reviewer Rebuttal Pre-Prep — BGF / AAMAS 2027

Anticipates the top reviewer concerns based on (a) the paper's own self-disclosed limitations and (b) the on-disk evidence base as of 2026-05-27. Each entry: **likely critique → defense → supporting artefact**. Keep responses < 200 words each for AAMAS author response window.

---

## R1. "Your headline A-vs-B contrast is null (p > 0.85 on every primary metric). This is a negative result."

**Defense.** The dissociation *is* the contribution, not a failure mode. The paper foregrounds this explicitly in the abstract: "We read this dissociation as a substantive empirical claim about RLHF residual bias." The story arc is (i) `Φ` works at the policy layer that reads it directly (Condition D, Gini = 0.325, ρ = +1.000); (ii) the same `Φ` reaches the LLM prompt but does not propagate through the decision channel — a measurement *about the LLM*, not about BGF. The same dissociation has not been reported elsewhere because no prior LLM-agent paper held the population fixed across an RAG ablation with 10 seeds and a pre-registered hypothesis.

**Artefact.** §8.1; `analysis/tables/bootstrap_ci.csv`; `docs/hypothesis_preregistration.md` predates the run; the meta-analysis pooled estimate `analysis/tables/meta_analysis.json` is the formal pooled effect with τ² and I².

---

## R2. "Single LLM family (Mistral-7B-Instruct-v0.3) — you cannot generalise to RLHF as a class."

**Defense.** Agreed in scope. The §3.4 thesis is stated as a falsifiable prediction *about a property of single-agent RLHF training*, not a universal empirical claim. The existence proof for one family is on disk (`B_RLHF(A) ≈ 0.254`, audit-traceable). The benchmark spec released alongside the paper (§10) is structured for *per-model* submissions; the audit-traceable Qwen / GPT-4o-mini rows are withdrawn (Appendix A.3) and re-execution is queued. We commit to releasing the Qwen-2.5-7B and GPT-4o-mini rows in the camera-ready, contingent on GPU availability.

**Artefact.** Appendix A.3; `benchmark/spec.md`; queue: `scripts/run_cross_model_comparison.py`.

---

## R3. "Cooperation logistic regression is Austrian-only and AUC = 0.64. This barely beats chance."

**Defense.** Acknowledged in §3.X, line 195: "Fidelity scores should therefore be interpreted as rough indicators of directional consistency rather than precise behavioral calibration." The model is used as a *baseline for persona_decay scoring*, not as a primary causal claim. The replacement of the prior heuristic `0.2 + 0.6 · trust · (1−risk)` with this empirically grounded (if modest) model is itself a methodological improvement: trust as primary driver is *not* supported in the data (all three trust items have CIs overlapping zero). Reviewers should read this as a correction of an overconfident prior, not as a strong predictive claim.

**Artefact.** §6.6.2; bootstrap CIs in `data/cooperation_model.json`; cross-cluster validation against WVS Wave 7 and Herrmann-Thöni-Gächter (2008) is the independent behavioural benchmark.

---

## R4. "Memory ablation H8 was run under `policy: mock` — you cannot test memory with a deterministic policy."

**Defense.** Disclosed in §8.5 explicitly and reframed from "result" to "pre-registered prediction." The fix is not a paper-text revision; it is the LLM-policy rerun via `scripts/run_memory_ablation_llm.sh` (~6-8 GPU-h). We are running this for the camera-ready. The decision to relocate the prediction-vs-measurement split into §8.5 was made specifically to avoid the criticism the reviewer is identifying.

**Artefact.** §8.5; `scripts/run_memory_ablation_llm.sh`.

---

## R5. "Cross-cultural Spearman ρ = +1.000 looks suspiciously perfect."

**Defense.** Expected for n = 6 cluster ranks with a deterministic rule-based policy: the policy *reads* the ESS cluster-level cooperation rate ordering by construction, so ρ = +1.000 is the *consistency check that the policy implements its own definition*, not a generalisation claim. The paper acknowledges this is a within-survey result (§8.3). The independent behavioural benchmark — Herrmann-Thöni-Gächter (2008) per-city PGG contributions — is the non-circular evidence: Spearman ρ = +0.886, exact two-sided permutation p = 0.033, with PGG data BGF never ingests. The WVS Wave 7 replication (r = +0.977) adds further independent corroboration. The H9 result is the one to argue for in rebuttal; H5 in the rule-based proxy is a consistency check.

**Artefact.** `analysis/h9_behavioral_benchmark.py`; §6.5 vs §8.3 separation.

---

## R6. "N = 100 agents is too small to claim emergence."

**Defense.** Three responses. (i) The N=500 D condition (10 seeds, deterministic) is on disk — emergence-relevant macro-statistics (Gini = 0.325) reproduce at 5× scale with `Gini SD = 5 × 10⁻⁴`. (ii) The N=500 LLM A/B disambiguator is queued; current snapshot is the §8.1 N=100 cells. (iii) The phase-transition sweeps (§6.4) at N = 20 use scaling arguments + power-law fits with Clauset MLE / KS goodness-of-fit, exactly the rigorous emergence-testing protocol; they are explicitly described as "small-population pilots" requiring N=500 validation. We are not claiming emergence from N=100 alone.

**Artefact.** `mx_D_n500_s{1..10}`; `analysis/tables/phase_transition_*.json`; §6.4 caveat block.

---

## R7. "`B_RLHF` uses uniform reference — that is a strawman, no human cooperates uniformly."

**Defense.** §3.X already addresses this (line 189): "laboratory public-goods cooperation rates of 40–60 % (Chaudhuri 2011) imply a natural TV of 0.07–0.27 even for an unbiased human population." The uniform reference is the analytically tractable choice that bounds the **absolute** deviation from no-bias; the **directional** claim `B_RLHF(B) < B_RLHF(A)` is invariant to the reference choice (TV is translation-invariant in the relevant sense). A human-calibrated index `B_RLHF*(π) = TV(π, π_human)` requires the §8.4 human-subject study; the spec is in `docs/human_subjects_protocol.md` pending IRB.

**Artefact.** §3.2.1 Proposition 1; §3.X line 189; §8.4.

---

## R8. "Why should a public goods game generalise to other social dilemmas?"

**Defense.** Two-part response. (i) The thesis statement (line 44) explicitly says "any instruction-tuned LLM will exhibit B_RLHF > 0 in *any multi-agent social dilemma*" as a falsifiable cross-game prediction, not a within-PGG claim. Prediction (1) of the central thesis is exactly this; testing it is future work. (ii) Within-paper, the LPGG (§3.X with `c = 3`, `r = 12/N` constraints) is chosen because the social-dilemma condition `1/N < r < 1` is satisfied for 4 ≤ N ≤ 11, the regime in which cooperation is non-trivially costly. The construct-validity framework (C4 in §3.3) explicitly treats payoff design dependence.

**Artefact.** §3.X LPGG derivation; §3.3 C4; line 44 thesis.

---

## R9. "No human evaluation comparison — why should we believe BGF agents look like real humans?"

**Defense.** The infrastructure is complete and the protocol pre-registered (`docs/human_subjects_protocol.md`); IRB approval is the gating step, not engineering. Three pre-built Condition A vs B vignette pairs are deployed at `/human-eval` and the analysis pipeline (`scripts/analyze_human_eval_ratings.py`) emits a boxplot ready for paper insertion. Budget envelope is ~$120 Prolific. We commit to including these results in the camera-ready conditional on IRB and budget. In the interim, the independent behavioural benchmark against Herrmann-Thöni-Gächter (2008) PGG contributions (Spearman ρ = +0.886, p = 0.033) is the closest non-self-report cross-validation — see R5.

**Artefact.** `api/templates/human_eval.html`; H6 in `docs/hypothesis_preregistration.md`.

---

## R10. "Appendix A.3 discloses three infrastructure bugs (L-1, L-2, C-1). Why should we trust the post-patch numbers?"

**Defense.** Reproducibility is one of our core contributions. The full audit trail is preserved precisely *so reviewers can verify* what changed and why. (i) L-1 (PromptLogger off-by-one) affected only prompt logs, not action streams. (ii) L-2 (kernel batched path drops `rag_context`) is the bug that caused the pilot `B_RLHF` values to be impossible under the TV bound `B_RLHF ≤ 2/3`; this is why the pilot row was withdrawn. (iii) C-1 (env-var batch-size override) affected throughput, not correctness. The patched code is the reference; all current numbers in the paper are from the post-patch reality. The `bgf_logging/witness.py` cryptographic reproducibility witness allows any reviewer to verify a re-run matches what is reported.

**Artefact.** Appendix A.3; `bgf_logging/witness.py`; `scripts/reproduce_paper.sh`.

---

## R11 (general). "What is the take-home for the AAMAS community?"

**Defense (one-sentence summary).** Even when an empirically grounded population synthesis function `Φ` perfectly preserves joint ESS distributions across 15+ attributes, the inference-time grounding does not propagate through a 7B-instruction-tuned LLM into observable behavioural differentiation — so multi-agent simulation researchers using LLM-policy agents should not assume that prompt-level demographic grounding is equivalent to behavioural calibration. The measurement apparatus (BRM + B_RLHF) is the standalone tool we release.

---

## Quick-reference: Holm-Bonferroni status across H1–H9

(Lifted from §6.0 and line 1061.)

| H | Status | Note |
|---|---|---|
| H1 | Dirichlet weight-robust ✓ | Passes FWER α = 0.05 |
| H2 | **Falsified at N=100** (MWU p = 0.91) | Pending N=500 |
| H3 | Pending LLM scale | — |
| H4 | Directional only (ΔQ = +0.005, p = 0.68) | Pilot did not reproduce |
| H5 | Continuous post-hoc p < 0.0001 | Group-level pre-reg p = 0.167 (n=4 power ceiling) |
| H6 | Infra ready, awaiting IRB | — |
| H7 | Unaffected by §8.1 N=100 falsification | Cross-model design separate |
| H8 | Pre-reg only; mock-policy on disk | LLM rerun pending |
| H9 | Per-test p = 0.033 ✓ | Does *not* pass FWER α = 0.0056 |

**Family-wise confirmed at α = 0.05:** H1 only.
**Per-test confirmed:** H1, H5 (post-hoc), H9.
**Pending:** H3, H6, H8 at LLM scale; N=500 disambiguates H2 & H3.

---

## What still hurts us most (be honest with the AC)

1. **H2 falsification at N=100** is the single biggest narrative risk. Lean into it as a finding, not a bug. The dissociation framing in the abstract is the right move.
2. **N=500 LLM A/B stall (2026-05-24)** needs to be resolved before submission. This is the disambiguator.
3. **Cross-family panel** must have at least one non-Mistral row by camera-ready.
4. **Human-eval IRB** must be resolved before the §8.4 results can be claimed.
