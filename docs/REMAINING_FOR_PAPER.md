# What Remains for Paper Completion
> CS Bachelor's Capstone — Submission deadline: ~2026-06-19 (2 weeks)
> Last updated: 2026-06-05 (session 3)

**Context for future Claude Code sessions:** Multi-agent LLM simulation paper (BGF framework). Working dir: `/mnt/sdb1/workspace/lucastourinho/SyntheticSocieties`. Paper: `docs/paper.md`. Experiments: `experiments/`. Active GPU job: `tmux: h8_memory_ablation` (M3 cells pending).

---

## 2-Week Timeline

```
Week 1 (Jun 5–12):
  Days 1–2  → Wait for H8 M2U+M3 (8 cells remain); §8.5 updated through M2G already
  Days 2–3  → Run T1-A analysis when M3 complete; finalize H8 verdict in paper
  Days 3–4  → Pre-reg deviation table (#8, #9); number verification
  Days 4–5  → Figure regeneration (Figs 1, 2, 15, 16)

Week 2 (Jun 12–19):
  Days 7–9  → Padded control N=100 if GPU available (T3-A)
  Days 9–11 → n500 multi-seed if GPU available (T3-C, highest value)
  Days 11–13 → Final self-consistency pass, all status blocks resolved
  Days 13–14 → Git clean, reproduce_paper.sh, README, submission
```

---

## TIER 0 — Currently Running

### T0-A: H8 Memory Ablation v2 — 8/24 CELLS REMAINING
```
tmux: h8_memory_ablation
Log: logs/h8_memory_ablation_v2_full.log
```

**Completed (16 cells) — key findings already in paper §8.5:**
- M0G ✓ (n=3): mean coop=0.583±0.085, B_RLHF=0.256±0.096
- M0U ✓ (n=3): mean coop=0.417±0.026, B_RLHF=0.150±0.077 — Δ_M0=+0.167 (H8 direction ✓)
- M1G ✓ (n=3): mean coop=0.367±0.058, B_RLHF=0.128 — LOWER than M0G (counter-monotonicity!)
- M1U ✓ (n=3): mean coop=0.633±0.289 (s7 outlier=0.950), B_RLHF=0.306
- M2G ✓ (n=3): mean coop=0.367±0.058 — EQUAL to M1G (archive count adds nothing)
- M2U: s42 ✓, s123 running, s7 pending

**Remaining (8 cells):**
- M2U s123 (RUNNING), s7 (pending)
- M3G s42, s123, s7
- M3U s42, s123, s7

**When M3 complete, run immediately:**
```bash
cd /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2
```

**Key question for M3:** Does M3G exceed M0G (>0.583)? If not, H8 is falsified for grounded arm.
Current trajectory: M0G=0.583 → M1G=0.367 → M2G=0.367 (opposite of M0<M1<M2<M3).

### T0-B: Bad Apple N=500 Sweep — LAUNCHED (PID 470075)
```bash
# Check progress:
tail -f logs/bad_apple_n500_*.log
```
CPU-only, ~30–60 min. Updates §6.4 f* from N=20 pilot to N=500 confirmed.

### T0-C: N=500 Cascade — DONE (R29 terminal, no action)
Both condA s1 / condB s2 at R29 (14,500 events each). Paper updated.

---

## TIER 1 — Critical (must complete before submission)

### T1-A: Run H8 Full Analysis (when M3 done)
```bash
source venv/bin/activate
python scripts/analyze_memory_ablation.py --suffix _v2
```
Output: `analysis/tables/memory_ablation.json`

Then update paper:
- **T1-B**: §8.5 status block → "COMPLETE — [verdict]"
- **T1-C**: Add M2U and M3 tables (follow M1G/M2G template already in §8.5)
- **T1-D**: H8 hypothesis table row (line ~1080) → final verdict with numbers
- **T1-E**: H8 hypothesis footnote (line ~1083) → final verdict sentence
- **T1-F**: Abstract H8 count → final (currently "16/24")
- **T1-G**: Regenerate Figure 15: `python scripts/analyze_memory_ablation.py --suffix _v2 --plot`

**H8 verdict logic:**
- If M3G > M0G (>0.583): H8 partially saved for M3 step; M1/M2 still violated
- If M3G ≤ M0G: H8 falsified for grounded arm; report as falsified
- Check ungrounded arm separately (M0U<M1U<M2U<M3U?)

### T1-H: Update Abstract + Conclusion with H8 Finding
Once verdict locked, integrate into RLHF attractor narrative.
Abstract currently says "16/24 cells" — must be resolved before submission.

### T1-I: Final Self-Consistency Pass
```bash
grep -n "PENDING\|ACTIVE\|in progress\|currently running\|awaits\|16/24\|8/24" docs/paper.md
```
Every placeholder must be resolved.

### T1-J: Verify Test Count
```bash
source venv/bin/activate && python -m pytest tests/ 2>&1 | grep -E "passed|failed" | tail -1
```
Paper claims 1,578 tests. Current: 1578 total (1577 pass + 1 fail in test_llm_backend — fixture interference, passes solo). Update §3 and line ~13 if count changed.

---

## TIER 2 — Important for Academic Rigor

### T2-A: Bad Apple N=500 Results (after T0-B finishes)
```bash
tail logs/bad_apple_n500_*.log | grep -E "f\*|phase|transition|complete"
```
Update §6.4 with N=500 f* value. Currently: f*≈0.023 from N=20 pilot.

### T2-B: Pre-Registration Deviation Table
File: `docs/hypothesis_preregistration.md`
Need to add:
- **Deviation #8**: H8 v1 invalidation (confirm logged)
- **Deviation #9**: H8 v2 results deviate from prediction (add when measured)
  - M0G actual (0.583) vs predicted (0.330): Δ=+0.253 — RLHF overcooperation
  - Counter-monotonicity at M0→M1→M2 (predicted monotone increase)

### T2-C: Verify All Paper Numbers
```bash
source venv/bin/activate && python scripts/compute_paper_numbers.py --verify
```
Key: coop A=0.461/B=0.455, Gini A=0.718/B=0.715, BRM 0.832/0.848, cascade R29 A=93.8%/B=95.6%.

### T2-D: Regenerate Figures
```bash
source venv/bin/activate
python scripts/plot_empirical_analysis.py          # Fig 1
python scripts/fix_figure2_canonical.py            # Fig 2
python scripts/plot_cross_cultural.py              # Fig 16
python scripts/plot_memory_ablation.py --suffix _v2   # Fig 15 (after M3 done)
```
Check `docs/figure_status.md` for stale list.

### T2-E: Limitations §9 Final Scan
`grep -n "pending\|awaits\|future\|not yet" docs/paper.md | grep -i "§9\|Limitation"`
Update any that now have data (e.g., Limitation 20 — cascade single-seed).

---

## TIER 3 — Do If Time Allows (need 10+ days remaining)

### T3-A: Padded Control @ N=100 (3–5h GPU)
```bash
source venv/bin/activate
python scripts/run_padded_control.py --n-agents 100 --seeds 42 123 7
```
Upgrades §8.6: "directionally resolved at N=50" → "formally closed at N=100".

### T3-B: Mediation Analysis (30 min CPU)
```bash
source venv/bin/activate && python analysis/mediation_summary.py
```
Fills §C.4 evidence audit gap.

### T3-C: N=500 Multi-Seed Cascade (20–30h GPU — HIGHEST ACADEMIC VALUE)
Converts cascade from 1-seed exploratory to multi-seed confirmatory.
condB > condA direction claim becomes statistically testable.
```bash
bash scripts/run_n500_gap_fill.sh --seeds 2 3 4 5 6 7 8 9 10 --rounds 15
```

---

## TIER 4 — Final Polish (last 2–3 days)

- `git status` → clean commit
- `bash scripts/reproduce_paper.sh --dry-run` → must pass
- README: add RLHF cascade paragraph
- CITATION.cff: confirm title/year
- Abstract spot-check: coop rates, B_RLHF, Gini in abstract vs §8.1/§8.5 body

---

## BLOCKED

| Item | Reason |
|------|--------|
| H3 human validation (Prolific) | IRB approval — future work |
| Cross-model panel @ N=100 | GPU+API cost; N=20 data in paper |
| ESS multi-country pooled refit | Data not yet released |

---

## ALREADY SOLID — No Action Needed

| Item | Evidence | Location |
|------|----------|----------|
| N=100 LLM confirmatory (10 seeds) | coop 0.461/0.455, MWU p=0.91 | §8.1 |
| Condition D (N=500, 10 seeds) | Gini=0.325 BCa [0.324,0.325] | §8.2 |
| Cross-cultural gradient | ρ=+1.000 ESS, ρ=+0.886 HTG, r=+0.977 WVS | §8.3 |
| Padded control N=50 | 3-seed B_RLHF=0.255 (>0.195 both arms) | §8.6 |
| N=500 cascade R29 | condA 93.8%, condB 95.6%, agg 0.513/0.543 | §8.1.5 |
| H8 M0–M2G full characterisation | M0G(0.583)>M1G(0.367)=M2G(0.367); Δ_M0=+0.167 | §8.5 |
| H8 M1U all 3 seeds | 0.633 mean (s7=0.950 outlier documented) | §8.5 |
| 1,578 tests (130 files) | 1577 pass + 1 fixture-interference fail | §3/README |
| Pre-reg H1–H9 + 7 deviations | Holm-Bonferroni | docs/hypothesis_preregistration.md |
| SHA-256 reproducibility witness | bgf_logging/witness.py | §3.8 |

---

## Quick Reference: Key Numbers

```
N=100 LLM terminal (10 seeds):
  Coop: A=0.461, B=0.455 (MWU p=0.91)
  Gini: A=0.718, B=0.715 (p=0.85)
  BRM:  A=0.832, B=0.848 (Δ=+0.016, g≈+0.78, p=0.089)
  B_RLHF: ≈0.195 both arms

N=500 cascade R29 (1 seed per arm):
  condA: coop=93.8%, Gini=0.9632, B_RLHF=0.605, agg=0.513
  condB: coop=95.6%, Gini=0.9680, B_RLHF=0.623, agg=0.543
  condB peak R27: coop=96.6%, B_RLHF=0.633 (93% of 2/3 ceiling)

H8 v2 measured (N=20, T=10, terminal round):
  M0G: coop=0.583±0.085, Gini=0.218±0.037, B_RLHF=0.256±0.096
  M0U: coop=0.417±0.026, Gini=0.177±0.018, B_RLHF=0.150±0.077
  M1G: coop=0.367±0.058, Gini=0.198±0.031, B_RLHF=0.128±0.058
  M1U: coop=0.633±0.289, Gini=0.306±0.089, B_RLHF=0.306 (s7=0.950)
  M2G: coop=0.367±0.058, Gini=0.200±0.034, B_RLHF=0.144±0.067
  M2U: partial (s42=0.400, s123+s7 pending)
  M3:  pending
  Pre-reg predicted M0G=0.330, M1G=0.362, M2G=0.407 (all wrong — RLHF overcoop)
  Counter-monotonicity: M0G(0.583) > M1G(0.367) ≈ M2G(0.367)

Condition D rule-based (N=500, 10 seeds):
  Gini=0.325 BCa [0.324,0.325]
```

---

## Critical Commands

```bash
# H8 analysis (run when M3 done)
source venv/bin/activate && python scripts/analyze_memory_ablation.py --suffix _v2

# Check H8 tmux
tmux capture-pane -t h8_memory_ablation -p | tail -20

# Check completed v2 cells
ls experiments/ | grep "_v2" | grep -v smoke | sort

# Bad apple check
tail logs/bad_apple_n500_*.log | tail -5

# Paper PENDING grep
grep -n "PENDING\|ACTIVE\|currently running\|in progress" docs/paper.md

# Test count
source venv/bin/activate && python -m pytest tests/ 2>&1 | grep -E "passed|failed" | tail -1

# Number verification
source venv/bin/activate && python scripts/compute_paper_numbers.py --verify

# Cascade status
python3 -c "import json; [print(k, json.load(open(f'experiments/{v}/heartbeat.json'))) for k,v in [('condA','mx_A_n500_s1'),('condB','mx_B_n500_s2')]]"
```
