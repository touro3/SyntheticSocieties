# What Remains for Paper Completion
> CS Bachelor's Capstone — Submission deadline: ~2026-06-19 (2 weeks)
> Last updated: 2026-06-07 (session 7 — T2-E scan clean; reproduce_paper.sh PASSED; abstract spot-check ✓; s3/s4 running at R11→R12; ~6-8h until s3/s4 finish)

**Context for future Claude Code sessions:** Multi-agent LLM simulation paper (BGF framework). Working dir: `/mnt/sdb1/workspace/lucastourinho/SyntheticSocieties`. Paper: `docs/paper.md`. Active GPU jobs: `tmux: n500_condA` (mx_A_n500_s3, R11/15), `tmux: n500_condB` (mx_B_n500_s4, R11/15).

---

## 2-Week Timeline

```
Week 1 (Jun 5–12):
  Day 1 (done) → H8 complete + paper updated; bad apple N=500 run + §6.4 updated; figs 1,15,16 regenerated
  Days 2–3  → Pre-reg deviation table final check; limitations §9 scan (T2-E)
  Days 3–5  → ✅ s2/s3 DONE (2026-06-06); §8.1.5 updated; s3/s4 at R11/15 running → update §8.1.5 with 3-seed means when done

Week 2 (Jun 12–19):
  Days 7–9  → Padded control N=100 if GPU available (T3-A)
  Days 9–11 → n500 seeds s4–s10 if GPU available (T3-C, highest value)
  Days 11–13 → Final self-consistency pass; reproduce_paper.sh dry-run
  Days 13–14 → Git clean, README, CITATION.cff, submission
```

---

## TIER 0 — Currently Running

### T0-C: N=500 Cascade Multi-Seed — ▶ RUNNING (s3/s4)
```
DONE: mx_A_n500_s2 (R15 terminal: coop=88.8%, Gini=0.8338, BRM=0.7513, 2026-06-06)
DONE: mx_B_n500_s3 (R15 terminal: coop=88.0%, Gini=0.8206, BRM=0.7776, 2026-06-06)
RUNNING: tmux n500_condA → mx_A_n500_s3 (R11/15 as of 2026-06-07)
RUNNING: tmux n500_condB → mx_B_n500_s4 (R11/15 as of 2026-06-07)
```
KEY FINDING (2026-06-07): 2-seed multi-arm at R15 → condA mean 89.7%±1.3 pp, condB 89.8%±2.5 pp — H2 null consistent at N=500. T=30 condB>condA pattern not replicated.

**When s3/s4 finish:**
```bash
python3 -c "
import json, os
for tag, exp in [('condA s3','mx_A_n500_s3'),('condB s4','mx_B_n500_s4')]:
    if os.path.exists(f'experiments/{exp}/summary.json'):
        s = json.load(open(f'experiments/{exp}/summary.json'))
        m = s['metrics']
        ad_last = None
        with open(f'experiments/{exp}/round_metrics.jsonl') as f:
            for l in f:
                if l.strip(): ad_last = json.loads(l)
        coop = ad_last['action_distribution']['cooperate'] if ad_last else '?'
        print(f'{tag}: R15 coop={coop:.3f}, Gini={m[\"gini\"][\"final\"]:.4f}, BRM={m[\"brm\"]:.4f}')
"
```
Update §8.1.5 cascade table with R15 terminal values for s3/s4; recompute 3-seed means.

---

## TIER 1 — Critical (all done except T1-I final pass)

| Task | Status |
|------|--------|
| T1-A: H8 analysis | ✅ DONE — `analysis/tables/memory_ablation.json` |
| T1-B–G: §8.5 update (status, M3 tables, verdict, fig caption, abstract) | ✅ DONE |
| T1-H: Abstract + conclusion H8 narrative | ✅ DONE |
| T1-I: Final self-consistency pass | ✅ Done (2026-06-05 session 5) — 3 remaining PENDING markers all legitimately blocked (cross-cultural LLM, Prolific, cascade multi-seed). Re-run before final submission. |
| T1-J: Test count verify | ✅ 1578 passed (matches paper) |

**T1-I command (run before submission):**
```bash
grep -n "PENDING\|ACTIVE\|in progress\|currently running\|awaits\|16/24\|19/24\|M3 pending" docs/paper.md
```

---

## TIER 2 — Important for Academic Rigor

| Task | Status |
|------|--------|
| T2-A: Bad apple N=500 | ✅ DONE — f*=0.041, R²=0.996, scale reversal found; §6.4 updated |
| T2-B: Pre-reg deviation #9 (H8 final) | ✅ DONE — added to `docs/hypothesis_preregistration.md` |
| T2-C: Verify paper numbers | ✅ DONE (2026-06-07) — all 6 N=100 key metrics verified ✓; N=500 cascade seeds reported; script fixed to read summary.json |
| T2-D: Figure regeneration | ✅ Figs 1,15,16 regenerated; Fig 2 script errors (NoneType, not blocking); Fig 16 → `cross_cultural_expanded.png`; **forest_plot H8 fixed (2026-06-07): [verified] −0.130; n500_cascade_multiseed.pdf NEW (2026-06-07) — 4-seed trajectory** |
| T2-E: Limitations §9 scan | ✅ DONE — Limitation 20 title updated; line 1209 "awaits gap-fill" fixed; Limitations 8/13/17/22 legitimately blocked |

**T2-C: compute_paper_numbers.py** (no `--verify` flag, just run):
```bash
source venv/bin/activate && python scripts/compute_paper_numbers.py
# Output: analysis/paper_numbers.json + console diff
# Script reports "None" for metrics it can't locate — check manually
```

**T2-D: Fig 2 broken** (`fix_figure2_canonical.py` errors: `NoneType object is not subscriptable` at line 59 — reads `analysis/paper_numbers.json` but `action_counts` key is None). Not blocking submission.

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

### T3-C: N=500 Multi-Seed Cascade — s3/s4 RUNNING; s5–s10 pending
s2 (condA) and s3 (condB) DONE 2026-06-06 (R15). s3 (condA) and s4 (condB) at R11/15 now. When done, update §8.1.5 with 3-seed means.
s5–s10 would convert cascade from exploratory to confirmatory; 20–30h GPU per seed.
```bash
bash scripts/run_n500_gap_fill.sh --seeds 4 5 6 7 8 9 10 --rounds 15
```

---

## TIER 4 — Final Polish (last 2–3 days)

- `git status` → clean commit (13 fig/table updates from 2026-06-07 reproduce run — not yet committed)
- ✅ `bash scripts/reproduce_paper.sh --dry-run` → **PASSED** (exit 0, 2026-06-07); forest plot: H7/H8/H9 all [verified]; Condition D Gini=0.325 [0.324,0.325] ✓; zenodo.json OK
- ✅ README: RLHF cascade + H8 + bad apple scale reversal added; test count 1484→1578 fixed
- ✅ CITATION.cff: updated with R30 complete + H8 falsified + 1578 tests
- ✅ hypotheses.md: rewritten to match paper H1-H9 (was legacy H1-H8 draft)
- ✅ Abstract spot-check (2026-06-07): all key numbers consistent (N=100 coop/Gini/BRM ✓, N=500 cascade ✓, H8 verdict ✓, 1578 tests ✓)

---

## BLOCKED

| Item | Reason |
|------|--------|
| H3 human validation (Prolific) | IRB approval — future work |
| Cross-model panel @ N=100 | GPU+API cost; N=20 data in paper |
| ESS multi-country pooled refit | Data not yet released |
| H6 @ LLM scale | GPU run required |

---

## ALREADY SOLID — No Action Needed

| Item | Evidence | Location |
|------|----------|----------|
| N=100 LLM confirmatory (10 seeds) | coop 0.461/0.455, MWU p=0.91 | §8.1 |
| Condition D (N=500, 10 seeds) | Gini=0.325 BCa [0.324,0.325] | §8.2 |
| Cross-cultural gradient | ρ=+1.000 ESS, ρ=+0.886 HTG, r=+0.977 WVS | §8.3 |
| Padded control N=50 | 3-seed B_RLHF=0.255 (>0.195 both arms) | §8.6 |
| N=500 cascade R30 T=30 COMPLETE (s1/s2) | condA 94.0% B_RLHF=0.607, condB 96.0% B_RLHF=0.627, agg 0.516/0.545 | §8.1.5 |
| **H8 COMPLETE + FALSIFIED** | M0G(0.583)>M1G=M2G=M3G(0.367); M3U=0.450±0.000 | §8.5 |
| **Bad apple N=500** | f*=0.041, k=5.2, R²=0.996; Gini scale-reversal | §6.4 |
| 1,578 tests | 1578 passed (2026-06-05 run) | §3/README |
| Pre-reg H1–H9 + 9 deviations | Holm-Bonferroni | docs/hypothesis_preregistration.md |
| SHA-256 reproducibility witness | bgf_logging/witness.py | §3.8 |

---

## Quick Reference: Key Numbers

```
N=100 LLM terminal (10 seeds):
  Coop: A=0.461, B=0.455 (MWU p=0.91)
  Gini: A=0.718, B=0.715 (p=0.85)
  BRM:  A=0.832, B=0.848 (Δ=+0.016, g≈+0.78, p=0.089)
  B_RLHF: ≈0.195 both arms

N=500 cascade R30 terminal (T=30 COMPLETE, 1 seed per arm, 2026-06-05):
  condA s1: coop=94.0%, Gini=0.9653, B_RLHF=0.607, agg=0.516
  condB s2: coop=96.0%, Gini=0.9695, B_RLHF=0.627, agg=0.545
  condB peak R27/R28: coop=96.6%, B_RLHF=0.633 (95% of 2/3 ceiling)

N=500 T=15 multi-seed (2026-06-06 COMPLETE for s2/s3):
  condA s2: R15 coop=88.8%, Gini=0.8338, BRM=0.7513
  condB s3: R15 coop=88.0%, Gini=0.8206, BRM=0.7776
  2-seed means: condA 89.7%±1.3 pp / condB 89.8%±2.5 pp (Δ=+0.1 pp)
  KEY: H2 null consistent at N=500 (condB>condA pattern not replicated)
  condA s3 / condB s4: R11/15 running

H8 v2 FINAL (N=20, T=10, terminal round, 3 seeds each):
  M0G: coop=0.583±0.085, Gini=0.218±0.037, B_RLHF=0.256±0.080
  M0U: coop=0.417±0.024, Gini=0.177±0.018, B_RLHF=0.150±0.062
  M1G: coop=0.367±0.047, Gini=0.198±0.026, B_RLHF=0.128±0.048
  M1U: coop=0.633±0.232, Gini=0.306±0.074, B_RLHF=0.306±0.230 (s7=0.950)
  M2G: coop=0.367±0.047 = M1G exactly
  M2U: coop=0.633±0.232 = M1U exactly
  M3G: coop=0.367±0.062, Gini=0.221±0.048, B_RLHF=0.072±0.034 (global min)
  M3U: coop=0.450±0.000, Gini=0.354±0.028, B_RLHF=0.122±0.008
  VERDICT: H8 FALSIFIED — grounded arm monotone decrease; ungrounded inverted-U

Bad apple phase transition:
  N=20 pilot: f*=0.023, k=15.1, R²=0.97, Gini increases 0.243→0.330
  N=500 confirmatory: f*=0.041, k=5.2, R²=0.996, Gini DECREASES 0.246→0.180 (scale reversal)

Condition D rule-based (N=500, 10 seeds):
  Gini=0.325 BCa [0.324,0.325]
```

---

## Critical Commands

```bash
# n500 cascade progress (s3/s4 now running)
python3 -c "import json,os; [print(t,json.load(open(f'experiments/{e}/heartbeat.json'))) for t,e in [('A-s3','mx_A_n500_s3'),('B-s4','mx_B_n500_s4')] if os.path.exists(f'experiments/{e}/heartbeat.json')]"

# Paper PENDING grep
grep -n "PENDING\|ACTIVE\|currently running\|in progress\|16/24\|19/24" docs/paper.md

# Test count
source venv/bin/activate && python -m pytest tests/ 2>&1 | grep -E "passed|failed" | tail -1

# Number verification
source venv/bin/activate && python scripts/compute_paper_numbers.py

# Limitations scan
grep -n "pending\|awaits\|future\|not yet" docs/paper.md | grep -i "§9\|Limitation"

# Bad apple N=500 results
cat analysis/bad_apple_sweep_n500.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('f*=',d['sigmoid_fit']['f_star'],'R2=',d['sigmoid_fit']['r2'])"
```
