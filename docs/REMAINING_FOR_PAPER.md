# What Remains for Paper Completion
> CS Bachelor's Capstone — Target: 100% approval
> Last updated: 2026-06-04

This document lists every item that must be resolved before `docs/paper.md` can be submitted as a final bachelor's capstone project. Items are ordered by blocking priority. Everything currently in progress or blocked is flagged.

---

## TIER 0 — Currently Running (wait for completion)

### T0-A: H8 Memory Ablation v2 — ACTIVE
- **What**: 24-cell LLM-policy ablation (M0–M3 × grounded/ungrounded × seeds 42,123,7; N=20, T=10)
- **Status**: Running in `tmux: h8_memory_ablation` since 2026-06-04 22:45 CEST. 23 cells scheduled; M0/s42 failed (partial events.jsonl — will re-run with `--skip-existing` once sweep finishes). ETA ~10–14 GPU-h from start.
- **When done**:
  1. Re-run M0/s42: `rm -rf experiments/ablation_M0_no_memory_grounded_s42_v2 && source venv/bin/activate && bash scripts/run_memory_ablation_llm.sh --skip-existing`
  2. Run analysis: `python scripts/analyze_memory_ablation.py --suffix _v2`
  3. Update paper §8.5 Table 7 with real measured values (replace predicted 0.609–0.742)
  4. Regenerate Figure 15: `python scripts/plot_memory_ablation.py --suffix _v2` (or equivalent)
  5. Update §8.5 status block from "ACTIVE" to measured results
  6. Update H8 row in hypothesis table with verdict (confirmed / falsified)
  7. Update abstract and conclusion with H8 finding

### T0-B: N=500 Cascade Terminal Runs — 2–3 rounds from T=30
- **What**: `mx_A_n500_s1` (condA, R27/30) and `mx_B_n500_s2` (condB, R28/30) completing their 30-round cascade
- **Status**: Both processes active at 100% CPU (PIDs 251909 and 252106). Expected to complete within ~4–6 hours.
- **When done**:
  1. Extract terminal metrics: `python3 -c "import json; [print(s, json.load(open(f'experiments/{s}/summary.json'))) for s in ['mx_A_n500_s1','mx_B_n500_s2']]"`
  2. Update paper §8.1.5 with terminal data (coop, Gini, B_RLHF at R30)
  3. Update abstract cascade sentence with final numbers
  4. Update Conclusion §10 cascade paragraph with terminal values
  5. Update Limitation 20 from "2–3 rounds from completion" to "complete"

---

## TIER 1 — Critical Before Submission (must be done)

### T1-A: Re-run M0/s42 After Sweep Completes
- **Command**: `source venv/bin/activate && bash scripts/run_memory_ablation_llm.sh --skip-existing`
- **Why**: The only failed cell in the 24-cell design. Without it, H8 analysis has 23/24 cells (missing M0 grounded baseline for s42).

### T1-B: Verify All 24 H8 Cells Have summary.json
```bash
for exp in ablation_M{0,1,2,3}_{no_memory,window,archive,full}_{grounded,ungrounded}_s{42,123,7}_v2; do
  [ -f "experiments/$exp/summary.json" ] && echo "OK: $exp" || echo "MISSING: $exp"
done 2>/dev/null
```

### T1-C: Run H8 Analysis and Verify Monotonicity
```bash
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2
```
- Check output for: persona_fidelity M0 < M1 < M2 < M3 (H8 prediction)
- If monotonicity confirmed → H8 supported; if violated → H8 falsified (report honestly)

### T1-D: Regenerate Figure 15
Figure 15 (`analysis/figures/memory_ablation_interaction.png`) currently shows mock-policy data. After H8 analysis completes, regenerate it:
```bash
python scripts/analyze_memory_ablation.py --suffix _v2 --plot analysis/figures/memory_ablation_interaction.png
```

### T1-E: Update Paper Table 7 and §8.5 with Real H8 Numbers
- Table 7 currently shows pre-registered *predicted* values (M0=0.609, M3=0.742)
- Replace with measured values from `analysis/tables/memory_ablation.json`
- The ⚠️ reader callout box must be updated or removed

### T1-F: Fix M0/s42 Partial Events.jsonl
The partial file must be deleted before re-running:
```bash
rm -rf experiments/ablation_M0_no_memory_grounded_s42_v2
```
This is already done as of 2026-06-04; just confirm it stays clean.

---

## TIER 2 — Important for Academic Rigor

### T2-A: Verify Test Suite Count in Paper (FIXED)
- Paper now correctly states **1,578 automated tests across 130 test files** (updated 2026-06-04)
- Verify before final submission: `source venv/bin/activate && python -m pytest tests/ --collect-only -q 2>&1 | tail -2`

### T2-B: Bad Apple Sweep at N=500
- Paper currently reports f* ≈ 0.023 from N=20 pilot; pre-registered N=500 re-run pending
- **CPU-only, ~30–60 min**: `bash scripts/run_bad_apple_sweep_n500.sh`
- Update §6.4 text with N=500 f* (expected to be higher, ≈0.05–0.20 at larger scale)
- This is academically desirable but not a hard blocker for bachelor submission

### T2-C: All Paper Numbers Verified
Run the verification script to confirm no stale numbers:
```bash
source venv/bin/activate
python scripts/compute_paper_numbers.py --verify
```
- All primary metrics (§8.1 N=100, §8.2 Condition D, §8.3 cross-cultural) should PASS
- Any FAIL flags must be investigated before submission

### T2-D: Ensure All Figures Are From Latest Data
Run each figure generator script to confirm figures are not stale:
- `python scripts/plot_empirical_analysis.py` → Figure 1
- `python scripts/fix_figure2_canonical.py` → Figure 2
- `python scripts/plot_networks.py` → Figures 3, 4
- `python scripts/plot_bad_apple.py` → Figure 5
- `python scripts/plot_cross_cultural.py` → Figure 16
- Memory ablation → Figure 15 (see T1-D)

### T2-E: Reproducibility Witness Check
```bash
source venv/bin/activate
python bgf_logging/witness.py --verify experiments/mx_A_s1 experiments/mx_B_s1
```
Confirms cryptographic integrity of primary experiment data.

### T2-F: Pre-Registration Final Update
- `docs/hypothesis_preregistration.md` should have deviation #7 (T=15 for most N=500 seeds) logged ✓
- Add deviation #8 if H8 v1 invalidation deviation is not already logged
- H8 v2 result should be added once measured

---

## TIER 3 — Desirable but Not Blocking

### T3-A: Padded Prompt Control at N=100 (Formal Closure of Limitation 8)
- Currently resolved at N=50 (3 seeds), paper characterises as "directionally resolved, not formally closed"
- **GPU-bound, ~3–5 hours**: Run `python scripts/run_padded_control.py --n-agents 100 --seeds 42 123 7`
- Updates §8.6 from "directionally resolved" to "formally closed"
- Strongly desirable for a rigorous capstone but not strictly blocking

### T3-B: Cross-Model Re-Execution (Qwen + GPT-4o-mini)
- On-disk cross-model artefact covers Mistral only; Qwen/GPT-4o-mini rows were withdrawn
- **GPU + API cost**: `python scripts/run_cross_model_comparison.py`
- Nice-to-have for the cross-model scope claim; not a blocker since the null cross-model rows are honestly disclosed

### T3-C: Human Evaluation Baseline (IRB-blocked)
- Infrastructure complete; IRB approval pending
- **Not feasible before bachelor submission** — document as future work (already done in §10)

### T3-D: Mediation Analysis (Persona × RAG Factorial)
- Code exists (`metrics/mediation.py`), output not aggregated
- **CPU-only, ~30 min**: `python analysis/mediation_summary.py`
- Fills gap in evidence audit row C.4; adds quantitative decomposition of grounding mechanism

### T3-E: Run Final Pipeline and Regenerate All Analysis Artifacts
```bash
source venv/bin/activate
python scripts/pipeline_full.py --no-gpu  # runs all CPU-only analysis
```
Ensures `analysis/tables/*.json` and `analysis/figures/*.png` are fully up-to-date.

---

## TIER 4 — Final Polish Before Submission

### T4-A: Paper Self-Consistency Check
- [ ] All section cross-references (§X.Y) point to real sections
- [ ] All table/figure numbers in text match actual table/figure numbers
- [ ] Abstract matches body (cooperation rates, B_RLHF values, dates)
- [ ] All "PENDING" / "ACTIVE" status blocks updated to actual results
- [ ] No stale dates (e.g., "as of 2026-06-03" → "as of 2026-06-04" or completion date)

### T4-B: Git Clean State
```bash
git status  # no untracked files except experiment outputs
git log --oneline -5  # clean commit messages
```

### T4-C: One-Command Reproduction Check
```bash
source venv/bin/activate
bash scripts/reproduce_paper.sh --dry-run  # should complete without error
```

### T4-D: README and CITATION.cff Up-to-Date
- README should describe the RLHF cascade finding (added at N=500 exploratory)
- CITATION.cff should have correct title, version, year

### T4-E: Limitations Section Complete
The 20 documented limitations (§9) are already comprehensive. Before final submission:
- Limitation 20 → update N=500 cascade status once T=30 complete
- Add any new limitations discovered during H8 analysis (e.g. small N=20 power)

---

## Summary Table

| Tier | Item | Status | Effort | Blocks |
|------|------|--------|--------|--------|
| T0-A | H8 ablation v2 runs | RUNNING | — | T1-A–E, H8 verdict |
| T0-B | N=500 cascade terminal | RUNNING | — | Cascade final numbers |
| T1-A | Re-run M0/s42 cell | After T0-A | <1h GPU | H8 completeness |
| T1-B | Verify 24 cells present | After T1-A | 5 min | H8 analysis |
| T1-C | H8 analysis + verdict | After T1-B | 5 min | Paper updates |
| T1-D | Regenerate Figure 15 | After T1-C | 5 min | Figure correctness |
| T1-E | Update Table 7 + §8.5 | After T1-C | 30 min | Paper correctness |
| T2-A | Test count verified | DONE ✓ | — | — |
| T2-B | Bad apple N=500 | Not started | 30–60 min CPU | §6.4 precision |
| T2-C | Paper numbers verify | After T0 | 10 min | Numbers integrity |
| T2-D | Figures from latest data | After T0 | 30 min | Figure integrity |
| T2-E | Reproducibility witness | After T0 | 10 min | Audit trail |
| T2-F | Pre-reg final update | After T1 | 15 min | Pre-registration integrity |
| T3-A | Padded N=100 | Optional | 3–5h GPU | Formal Lim. 8 closure |
| T3-B | Cross-model panel | Optional | GPU + API | Cross-family claim |
| T3-C | Human eval | BLOCKED (IRB) | — | Future work |
| T3-D | Mediation analysis | Optional | 30 min CPU | Causal decomposition |
| T4-* | Final polish | After T1-T3 | 2h | Submission ready |

---

## Execution Order for Fastest Completion

```
[NOW RUNNING] T0-A (H8 ablation, ~10h) + T0-B (N=500, ~4-6h)
     ↓ when T0-A done:
T1-A (re-run M0/s42, 30 min) → T1-B (verify) → T1-C (analyze) → T1-D (Figure 15) → T1-E (Table 7 + §8.5)
     ↓ when T0-B done:
Update N=500 terminal metrics in paper (cascade section, abstract, conclusion, Limitation 20)
     ↓ both complete:
T2-C (verify numbers) → T2-D (figures) → T2-E (witness)
     ↓ optional:
T2-B (bad apple N=500, 30–60 min) → T3-D (mediation, 30 min)
     ↓ final:
T4-A (self-consistency) → T4-B (git) → T4-C (reproduce) → T4-D (README) → T4-E (limitations)
     ↓
✅ SUBMISSION READY
```

---

## What Is Already Solid (No Action Needed)

- ✅ BGF framework `BGF = (A, E, G, P, Φ, T)` formally specified and implemented
- ✅ `B_RLHF = TV(π, π_uniform)` with Proposition 1 bound (≤ 2/3)
- ✅ BRM metric with Proposition 3 weight-robust ordering certificate
- ✅ N=100 primary LLM confirmatory extension (10 seeds, H1–H3 verdict)
- ✅ Condition D rule-based Gini = 0.325, within Eurostat range (10 seeds, N=500)
- ✅ Cross-cultural cooperation gradient Spearman ρ = +1.000 (H5/H9)
- ✅ WVS Wave 7 replication r = +0.977
- ✅ Herrmann–Thöni–Gächter 2008 behavioural benchmark replication ρ = +0.886
- ✅ Padded prompt control (Condition P, N=50, 3 seeds complete) — B_RLHF ordering confirmed
- ✅ N=500 cascade evidence through R27/R28 (condA=93.8%, condB=96.6%, B_RLHF 0.605/0.633)
- ✅ 1,578 automated tests across 130 test files
- ✅ 20 limitations fully documented with epistemic honesty
- ✅ Audit trail (L-1/L-2/C-1 bugs documented and patched)
- ✅ Pre-registration with Holm–Bonferroni family-wise correction
- ✅ SHA-256 cryptographic reproducibility witness
- ✅ ESS Round 11 cooperation baseline (logistic regression, AUC=0.640)
- ✅ Grounding stress tests (bad apple, macro shock, topology variation)
