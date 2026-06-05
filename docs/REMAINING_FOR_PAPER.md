# What Remains for Paper Completion
> CS Bachelor's Capstone — Submission deadline: ~2026-06-19 (2 weeks)
> Last updated: 2026-06-05

**Context for future Claude Code sessions:** This is a multi-agent LLM simulation paper (BGF framework). Primary working directory: `/mnt/sdb1/workspace/lucastourinho/SyntheticSocieties`. Paper at `docs/paper.md`. Key experiments live in `experiments/`. Active GPU jobs in tmux.

---

## 2-Week Timeline

```
Week 1 (Jun 5–12):
  Days 1–2  → Wait for H8 ablation to complete (tmux: h8_memory_ablation, ~10–16h remaining)
  Days 2–4  → Update §8.5 with real M1–M3 data as cells complete
  Days 4–5  → Bad apple N=500 sweep (CPU, 30–60 min)
  Days 5–6  → Pre-reg deviation table, figure regeneration, number verification

Week 2 (Jun 12–19):
  Days 7–9  → Padded control N=100 if GPU available (T3-A)
  Days 9–11 → n500 multi-seed if GPU available (T3-C, high value)
  Days 11–13 → Final self-consistency pass, all status blocks resolved
  Days 13–14 → Git clean, reproduce_paper.sh, README, submission
```

---

## TIER 0 — Currently Running (wait, then act)

### T0-A: H8 Memory Ablation v2 — 17/24 CELLS REMAINING
```
tmux: h8_memory_ablation
Log: logs/h8_memory_ablation_v2_full.log
ETA: ~10–16h from 2026-06-05 01:39 CEST (M1 grounded s123 started)
```
**Completed (7 cells):**
- M0 grounded s42/s123/s7 ✓ → mean terminal coop=0.583±0.085, Gini=0.218±0.037
- M0 ungrounded s42/s123/s7 ✓ → mean terminal coop=0.417±0.026, Gini=0.177±0.018
- M1 grounded s42 ✓ → terminal coop=0.300, Gini=0.209 (LOWER than M0 — watch for pattern)

**Remaining (17 cells):**
- M1 grounded s123 (RUNNING), s7 (pending)
- M1 ungrounded s42, s123, s7
- M2 grounded s42, s123, s7
- M2 ungrounded s42, s123, s7
- M3 grounded s42, s123, s7
- M3 ungrounded s42, s123, s7

**When sweep finishes, run immediately:**
```bash
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2
```
Output: `analysis/tables/memory_ablation.json` — this is the source of truth.

**Then update paper (T1-A through T1-E below).**

### T0-B: N=500 Cascade — DONE (R29 terminal, R30 OOM-stalled)
No action needed. `mx_A_n500_s1` and `mx_B_n500_s2` both at R29 (14,500 events each).
Paper already updated with R29 numbers as of 2026-06-05.

---

## TIER 1 — Critical (must complete before submission)

### T1-A: Run H8 Analysis and Extract Verdict
```bash
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2
```
Check for H8 monotonicity: is persona_fidelity M0 < M1 < M2 < M3 under grounding?
- **If confirmed** → H8 supported; update verdict in paper
- **If violated** → H8 falsified (report honestly); note which levels break ordering
- **Watch:** M1 grounded s42 (terminal coop=0.300) is already BELOW M0 s42 (0.550) — this
  may indicate M1 < M0 for cooperation, which would falsify H8's ordering claim.

### T1-B: Update Paper §8.5 With Real H8 Numbers
Location: `docs/paper.md` lines ~1320–1425

Replace the pre-registered *predicted* Table 7 values with measured ones.
Current predicted: M0=0.609→M3=0.742 (persona fidelity). These are placeholders.

For each completed level, add a results table following the M0 grounded template:
```markdown
| Seed | T | Coop (terminal) | Save | Work | Gini | B_RLHF_term |
|------|---|-----------------|------|------|------|-------------|
| 42   | ✓ | X.XXX           | ...  | ...  | X.XXX | X.XXX      |
```

Add contrast table (Grounded vs Ungrounded) per level — follow the M0 contrast table already in §8.5.

### T1-C: Update §8.5 Status Block
Change from:
`"Status: FIRST VALID H8 DATA — 7/24 CELLS COMPLETE"`
to:
`"Status: COMPLETE — all 24 cells run; H8 [confirmed/falsified] (§8.5.X)"`

### T1-D: Update H8 Hypothesis Table Row
Location: `docs/paper.md` line ~1080

Update the H8 row from "Partial (7/24)" to final verdict with key numbers.

### T1-E: Update Hypothesis Footnote
Location: `docs/paper.md` line ~1083

Update the H8 summary sentence with final verdict + key metric.

### T1-F: Regenerate Figure 15
```bash
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2 --plot analysis/figures/memory_ablation_interaction.png
# or if separate script:
python scripts/plot_memory_ablation.py --suffix _v2
```
Figure 15 currently shows pre-registered predictions (mock data). Replace with real data.

### T1-G: Update Abstract + Conclusion With H8 Finding
Search: `grep -n "H8\|memory ablation\|M0.*M3\|pending" docs/paper.md | head -20`

The abstract currently labels H8 as pending. Once verdict is in, integrate into the
RLHF attractor narrative: "even memory depth [does/does not] move the action distribution."

### T1-H: Final Self-Consistency Pass
```bash
# Find all remaining status placeholders
grep -n "PENDING\|ACTIVE\|in progress\|currently running\|in progress\|not yet\|awaits" docs/paper.md
```
Every "PENDING" / "ACTIVE" / "in progress" must be resolved before submission.
Primary ones to check:
- §8.5 status block (H8)
- §8.1.5 status block (cascade — already updated to R29 as of 2026-06-05)
- Figure 15 caption
- H8 row in hypothesis table

### T1-I: Verify Test Count
```bash
source venv/bin/activate
python -m pytest tests/ --collect-only -q 2>&1 | tail -2
```
Paper claims **1,578 automated tests across 130 test files**. Update if different.
Location: `docs/paper.md` line ~13 and §3.

---

## TIER 2 — Important for Academic Rigor

### T2-A: Pre-Registration Deviation Table (`docs/hypothesis_preregistration.md`)
Current deviations logged: 7
Need to add:
- **Deviation #8** (H8 v1 invalidation) — confirm it's logged; add if missing
- **Deviation #9** (H8 v2 results deviate from pre-registered predictions) — add once measured
  - M0 grounded actual (0.583) vs predicted (0.330): +0.253 deviation
  - Direction: RLHF overcooperation at all memory levels

### T2-B: Bad Apple Sweep @ N=500 (CPU-only, ~30–60 min)
```bash
source venv/bin/activate
bash scripts/run_bad_apple_sweep_n500.sh
```
Current paper reports f* ≈ 0.023 from N=20 pilot. Pre-registered N=500 re-run needed.
Expected: f* shifts toward 0.05–0.20 at N=500 (EGT prediction).
Updates §6.4 with confirmed N=500 phase transition.

### T2-C: Verify All Paper Numbers
```bash
source venv/bin/activate
python scripts/compute_paper_numbers.py --verify
```
Fix any FAIL flags. Primary metrics that must PASS:
- §8.1: coop A=0.461/B=0.455, Gini A=0.718/B=0.715, BRM A=0.832/B=0.848
- §8.2: Condition D Gini=0.325, BCa CI [0.324, 0.325]
- §8.3: ρ=+0.886, p=0.033
- §8.1.5: cascade R29 condA=93.8%, condB=95.6%

### T2-D: Regenerate Figures From Latest Data
```bash
source venv/bin/activate
python scripts/plot_empirical_analysis.py        # Figure 1
python scripts/fix_figure2_canonical.py          # Figure 2
python scripts/plot_cross_cultural.py            # Figure 16
python scripts/plot_memory_ablation.py --suffix _v2  # Figure 15 (after H8 done)
```
Check `docs/figure_status.md` for which figures are stale.

### T2-E: Pre-Registration Final Update
Confirm `docs/hypothesis_preregistration.md` reflects:
- H1: directional ✓
- H2: falsified at N=100 ✓
- H3: falsified at N=100 and N=500 ✓
- H4: not reproduced at N=100 ✓
- H5: post-hoc, ρ=0.800 ✓
- H6: pending LLM scale (document as future work)
- H7: non-universal ✓
- H8: update with v2 verdict
- H9: ρ=0.886, rule-based only ✓

---

## TIER 3 — Do If Time Allows (need 10+ days remaining)

### T3-A: Padded Control @ N=100 (3–5h GPU)
```bash
source venv/bin/activate
python scripts/run_padded_control.py --n-agents 100 --seeds 42 123 7
```
Upgrades §8.6 from "directionally resolved at N=50" → "formally closed at N=100".
Closes pre-registered Limitation 8.

### T3-B: Mediation Analysis (30 min CPU)
```bash
source venv/bin/activate
python analysis/mediation_summary.py
```
Fills evidence audit row C.4. Quantifies persona × RAG interaction decomposition.

### T3-C: N=500 Multi-Seed Cascade (20–30h GPU total — HIGH ACADEMIC VALUE)
Run seeds 2–10 for both conditions at T=15:
```bash
source venv/bin/activate
bash scripts/run_n500_gap_fill.sh --seeds 2 3 4 5 6 7 8 9 10 --rounds 15
```
This converts the cascade finding from "1-seed exploratory" to multi-seed confirmatory.
The condB > condA direction claim (Δ=1.8–4.8 pp) becomes statistically testable.
**This is the single highest-value experiment remaining if GPU time is available.**

---

## TIER 4 — Final Polish (last 2–3 days)

### T4-A: Abstract Matches Body
Spot-check these values in abstract vs. their source sections:
- N=100 cooperation rates (§8.1)
- N=500 cascade B_RLHF values (§8.1.5)
- H8 verdict (§8.5)
- Condition D Gini (§8.2)

### T4-B: Git Clean State
```bash
git status     # only untracked: experiment outputs, logs
git add docs/paper.md docs/REMAINING_FOR_PAPER.md docs/hypothesis_preregistration.md
git commit -m "update: H8 v2 complete, paper finalized for submission"
```

### T4-C: Reproducibility Check
```bash
source venv/bin/activate
bash scripts/reproduce_paper.sh --dry-run
```
Must complete without error.

### T4-D: README + CITATION.cff
- README: add 1-paragraph summary of RLHF cascade finding (N=500 scale-dependent B_RLHF=0.623)
- CITATION.cff: confirm `title`, `version`, `date-released`, `year`

### T4-E: Limitations Section Final Check
`docs/paper.md` §9 — 20 limitations currently. Scan for any that reference pending experiments
that are now complete. Key ones to update:
- Limitation 20 (cascade single-seed) — update text if multi-seed runs complete

---

## BLOCKED (do not attempt)

| Item | Reason |
|------|--------|
| H3 human validation (Prolific) | IRB approval pending — future work |
| Cross-model panel @ N=100 | GPU + API cost; N=20 data already in paper; not worth 2-week slot |
| ESS multi-country pooled refit | Awaits ESS R11 MD multi-country data release |

---

## ALREADY SOLID — No Action Needed

| Item | Evidence | Location |
|------|----------|----------|
| N=100 LLM confirmatory (10 seeds) | MWU p=0.91/0.85/0.35; BRM +0.016 | §8.1 |
| Condition D rule-based (N=500, 10 seeds) | Gini=0.325 BCa [0.324,0.325] | §8.2 |
| Cross-cultural gradient | ρ=+1.000 (ESS), ρ=+0.886 (HTG), r=+0.977 (WVS) | §8.3 |
| Padded control N=50 | 3-seed mean B_RLHF=0.255; ordering B_P>B_AB | §8.6 |
| N=500 cascade R29 terminal | condA 93.8%, condB 95.6%; agg_B_RLHF 0.513/0.543 | §8.1.5 |
| M0 ablation (6 cells) | Δ_coop=+0.167 grounded>ungrounded | §8.5 |
| M1 grounded s42 | terminal_coop=0.300 < M0 s42=0.550 (preliminary) | §8.5 |
| 1,578 automated tests | 130 test files | §3/README |
| Pre-registration (H1–H9) | Holm-Bonferroni; 7 deviations logged | docs/hypothesis_preregistration.md |
| Audit trail (L-1/L-2/C-1 bugs) | Documented and patched | docs/appendix_audit_trail.md |
| SHA-256 reproducibility witness | bgf_logging/witness.py | §3.8 |
| ESS cooperation baseline | logistic regression AUC=0.640 | §6.6.2 |

---

## Quick Reference: Key Numbers for Paper Consistency

```
N=100 LLM terminal (10 seeds):
  Coop: A=0.461, B=0.455 (MWU p=0.91)
  Gini: A=0.718, B=0.715 (p=0.85)
  BRM:  A=0.832, B=0.848 (Δ=+0.016, Hedges' g≈+0.78, p=0.089)
  B_RLHF: ≈0.195 both arms

N=500 cascade R29 (1 seed per arm, exploratory):
  condA (ablation_level=0): coop=93.8%, Gini=0.9632, B_RLHF=0.605, mean_wealth=182.43
  condB (ablation_level=5): coop=95.6%, Gini=0.9680, B_RLHF=0.623, mean_wealth=182.32
  condB peak: R27/R28 coop=96.6%, B_RLHF=0.633 (95% of 2/3 ceiling)
  agg_B_RLHF R1-R29: A=0.513, B=0.543
  condA plateau: R25-R29 invariant (93.8%, 0.605)
  Amplification over N=100: 3.2× terminal B_RLHF

Condition D (rule-based, N=500, 10 seeds):
  Gini=0.325 BCa CI [0.324,0.325] — within Eurostat range ✓

H8 M0 ablation v2 (N=20, T=10, terminal round):
  M0 grounded (n=3): coop=0.583±0.085, Gini=0.218±0.037, B_RLHF_term=0.256±0.096
  M0 ungrounded (n=3): coop=0.417±0.026, Gini=0.177±0.018, B_RLHF_term=0.150±0.077
  Δ_coop = +0.167 (predicted +0.094); Δ_gini = +0.041 (predicted -0.050, reversed)
  M1 grounded s42 (n=1): terminal_coop=0.300 < M0 s42=0.550 (preliminary counter-monotonicity)
```

---

## Critical Commands Reference

```bash
# H8 analysis (run after all 24 cells complete)
python scripts/analyze_memory_ablation.py --suffix _v2

# Check current H8 progress
for d in experiments/ablation_M{0,1,2,3}_{no_memory,window_only,archive,full}_{grounded,ungrounded}_s{42,123,7}_v2; do
  [ -f "$d/run_state.json" ] && python3 -c "import json; rs=json.load(open('$d/run_state.json')); print('$d:', rs['status'])" 2>/dev/null
done

# Check cascade status
cat experiments/mx_A_n500_s1/heartbeat.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('condA R'+str(d['round_id']), 'coop='+str(d['action_distribution'].get('cooperate')))"
cat experiments/mx_B_n500_s2/heartbeat.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('condB R'+str(d['round_id']), 'coop='+str(d['action_distribution'].get('cooperate')))"

# Paper consistency grep
grep -n "PENDING\|ACTIVE\|in progress\|currently running\|awaits" docs/paper.md

# Test count verification
source venv/bin/activate && python -m pytest tests/ --collect-only -q 2>&1 | tail -2

# Number verification
source venv/bin/activate && python scripts/compute_paper_numbers.py --verify

# Reproduce (dry run)
source venv/bin/activate && bash scripts/reproduce_paper.sh --dry-run
```
