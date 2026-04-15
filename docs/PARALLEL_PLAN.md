# Parallel Implementation Plan (GPU Simulations Running)

---

## PROMPT FOR CLAUDE — paste this to start a new session

```
You are helping me finish a research paper called the Behavioral Grounding Framework (BGF).
Two GPU simulations are running in the background (10-seed A/B LLM comparison and Condition D
at N=500). While they run, I need you to work through the PARALLEL_PLAN.md file in this repo
root — executing every task in priority order, top to bottom.

Project context:
- Working directory: /mnt/sdb1/workspace/lucastourinho/SyntheticSocieties
- Paper: docs/paper.md (~800 lines, Markdown, targeting AAMAS 2027 / NeurIPS 2026 workshops)
- Tests: pytest tests/ -v (921+ passing, keep them green)
- Venv: source venv/bin/activate before running any Python
- Full architecture is in CLAUDE.md

Work through PARALLEL_PLAN.md in this exact priority order:

1. PRIORITY 2 FIRST — write the three analysis scripts
   (scripts/analyze_ab_results.py, scripts/analyze_condition_d.py, scripts/reformat_figures.py)
   so that results → paper tables the moment the GPU jobs finish. Test each script against
   existing experiment dirs under experiments/ before moving on.

2. PRIORITY 5 — reproducibility hardening:
   - Grep for hardcoded /mnt/raid/ paths and replace with BGF_MODEL_CACHE env var
   - Pin all versions in requirements.txt, split off requirements-dev.txt
   - Create a minimal Dockerfile (python:3.11-slim, installs requirements, runs full pipeline)
   - Stage and commit SECURITY.md, .env.example, and the three new test files currently untracked

3. PRIORITY 6 — launch padded control (no GPU):
   source venv/bin/activate && python scripts/run_padded_control.py --seeds 42,123,7 --agents 100 --rounds 30
   Then run scripts/analyze_padded_vs_grounded.py and add the padded-control row to the
   ablation table in docs/paper.md Section 6.

4. PRIORITY 1 — paper writing:
   Edit docs/paper.md directly.
   - Add the missing citations to Section 2 (Fehr & Gächter 2002, Bolton & Ockenfels 2000,
     Sharma et al. 2023, Perez et al. 2022, Santurkar et al. 2023, Bisbee et al. 2023)
   - Add B_RLHF operationalization justification to Section 3
   - Add threats-to-validity paragraph to Section 3.7
   - Rewrite Limitations as three-tier structured subsection

5. PRIORITY 3 — human evaluation stimuli:
   - Read 20–30 agent decision traces from experiments/*/rounds.jsonl (mix Condition A and B)
   - Write docs/human_eval_stimuli.json with the 2AFC vignette format
   - Write docs/human_eval_protocol.md with Prolific study spec and OSF pre-registration plan

6. PRIORITY 4 — LaTeX stub:
   Create docs/paper_aamas.tex with ACM sigconf template headers, abstract pasted in,
   and section stubs. Do not convert the full paper — just make the file ready to fill.

After each priority block: run pytest tests/ -v and confirm it stays green.
Commit after each block with a descriptive message.

Check PARALLEL_PLAN.md as you go and mark completed items with [x].
When done, report which tasks are fully complete, which are partial, and what is left.
```

---

> Generated: 2026-04-14
> Context: 10-seed A/B LLM comparison + Condition D (N=500) running on GPU.
> Everything below requires no GPU and can be executed immediately.

---

## Simulations Currently Running

| Simulation | Command | Purpose | Paper Section |
|---|---|---|---|
| 10-seed A/B LLM | `python scripts/run_experiment_matrix.py --include-llm --conditions A B --seeds 1..10 --rounds 30 --agents 100` | Statistical power — bootstrap 95% CIs | Section 8.1, Table 2 |
| Condition D scale | `python scripts/run_full_pipeline.py --policy rule_based_ess --seeds 42,123,7 --rounds 30 --agents 500` | Rule-based ESS baseline vs. LLM | Section 8.2, Table 3 |

---

## Priority 1 — Paper Sections (no new results needed)

### 1.1 Strengthen Related Work (Section 2)

Missing citations that reviewers will flag:

- **Behavioral economics baselines**: Fehr & Gächter (2002), Bolton & Ockenfels (2000) — frame why 90% cooperation is empirically unrealistic
- **RLHF sycophancy**: Sharma et al. (2023), Perez et al. (2022) — reinforce Section 2.3 on alignment tax
- **Silicon sampling follow-ups**: Santurkar et al. (2023), Bisbee et al. (2023) — reviewers will expect these post-Argyle citations
- **Axelrod evolutionary dynamics** explicit cite for the game-theory kernel

### 1.2 Tighten Methodology (Section 3)

- [ ] Add 3–4 sentence justification for why `B_RLHF = TV(π, π_uniform)` is the right operationalization
- [ ] Cite token budget ablation from Section 6 in Section 3 prose
- [ ] Add "threats to validity" paragraph to Section 3.7: ESS sampling bias, synthetic vs. real individuals

### 1.3 Rewrite Limitations as Structured Subsection

Three tiers:

```
Acknowledged and mitigated:
  - Adversarial robustness (tested: bad apple injection, macro shock)
  - Seed variance (10-seed run in progress)

Acknowledged, not mitigated:
  - ESS coverage gaps (no Global South, single-economy model)
  - Single language (English prompts only)

Out of scope:
  - Individual-level validation
  - Causal identification
```

---

## Priority 2 — Analysis Scripts (write now, run when results land)

### 2.1 `scripts/analyze_ab_results.py`

Bootstrap CI computation over 10 seeds.

```
Input:  experiments/cmp_llm_s1..s10/metrics.json
Output: Table 2 — B_RLHF, BRM, Gini with 95% bootstrap CI (scipy.stats.bootstrap)
```

- [ ] Write script now, test on existing 3-seed experiments
- [ ] Output: `analysis/tables/table2_ab_bootstrap.csv`

### 2.2 `scripts/analyze_condition_d.py`

Condition comparison table builder.

```
Input:  Condition A/B/C dirs (exist) + Condition D dirs (pending)
Output: Table 3 — A/B/C/D comparison on BRM / B_RLHF / Gini / Coop
```

- [ ] Wire A/B/C rows now using existing experiments
- [ ] D rows auto-fill when simulation finishes
- [ ] Output: `analysis/tables/table3_condition_comparison.csv`

### 2.3 `scripts/reformat_figures.py`

Standardize all figures before submission.

```python
# Apply to all analysis/figures/*.png:
rcParams: font.size=10, figure.figsize=(3.5, 2.8)  # single-column ACM
Add (a)(b) subfigure labels where panels exist
LaTeX font rendering if texlive available
```

- [ ] Run on existing figures now — no results needed

---

## Priority 3 — Human Evaluation (start today, results in ~5 days)

**Impact**: "Validated by human judges" is a top-tier differentiator. Budget: ~$120.

### 3.1 Generate Stimuli

- [ ] Pull 20–30 agent decision traces from `experiments/*/rounds.jsonl`
  - 1 Condition A trace + 1 Condition B trace per scenario
  - Format: `Profile: [demographic snippet]. Situation: [world context]. Decision: [action + reasoning].`
- [ ] Create `docs/human_eval_stimuli.json`

### 3.2 Build Prolific Study

```
Platform:  Prolific Academic (prolific.com)
Design:    within-subjects, 2AFC — "which agent seems more like a real person?"
N:         40 participants
Duration:  ~15 min
Pay:       ~$3/participant → ~$120 total
Measure:   % correct identification of grounded (B) agent as more realistic
```

- [ ] Create study at prolific.com
- [ ] Pre-register on OSF (free, 30 min) — strengthens paper significantly

### 3.3 IRB / Ethics Statement

- [ ] Check if university requires IRB for judgment-of-fictional-agents study
- [ ] File for expedited review if required (1–2 week turnaround)
- [ ] Prepare ethics statement for AAMAS submission (required for human subjects)

---

## Priority 4 — Venue Formatting

### 4.1 Venue Decision

| Venue | Deadline | Page limit | Framing |
|---|---|---|---|
| AAMAS 2027 | ~Oct 2026 | 8 pages + refs | Multi-agent systems (perfect scope) |
| NeurIPS 2026 workshops (MASEC, LMRL) | ~Sep 2026 | 4–8 pages | LLM behavior / alignment |
| JASSS (journal) | Rolling | No limit | Computational social science |

- [ ] Confirm AAMAS 2027 deadline at ifaamas.org
- [ ] Choose primary venue

### 4.2 Start LaTeX Conversion

- [ ] Create `docs/paper_aamas.tex` using ACM `acmart` / `sigconf` template
- [ ] Identify cuts for 8-page limit:
  - Section 2.1 (traditional ABMs): compress to 2 sentences
  - Section 2.4 (RAG background): cut to 1 paragraph
- [ ] Section headers + abstract first pass

### 4.3 AAMAS Submission Checklist

- [ ] Author keywords: agent-based modeling, LLM agents, behavioral realism, RLHF
- [ ] CCS concepts: Computing methodologies → Multi-agent systems
- [ ] Anonymized version for blind review
- [ ] Supplementary material pointer (code, full results table)

---

## Priority 5 — Reproducibility Hardening

Unlocks **reproducibility badge** at AAMAS/NeurIPS — a citation multiplier.

### 5.1 Docker Container

```dockerfile
# Dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "scripts/run_full_pipeline.py"]
```

- [ ] Create `Dockerfile`
- [ ] Test: `docker build . && docker run bgf` reproduces non-LLM baseline

### 5.2 `requirements.txt` Audit

- [ ] Pin all versions (`==`) — current file is unpinned
- [ ] Split into `requirements.txt` (runtime) + `requirements-dev.txt` (test/dev)

### 5.3 HuggingFace Cache Path Fix

Current hardcoded path breaks external reproducers.

```python
# In decision/llm_backend.py (and any other loader):
model_cache = os.environ.get("BGF_MODEL_CACHE", "/mnt/raid/workspace/lucastourinho/models")
```

- [ ] Grep for hardcoded `/mnt/raid/` paths and replace with env var
- [ ] Document `BGF_MODEL_CACHE` in README and `.env.example`

### 5.4 Commit Untracked Files

- [ ] `git add SECURITY.md .env.example` and commit
- [ ] Stage and commit `tests/test_adversarial_localization.py`, `test_gini_range_condition_ab.py`, `test_network_modularity_ab.py`

---

## Priority 6 — Padded Control (Phase 29.1, no GPU needed)

Answers reviewer question: *"couldn't you just pad the prompt instead of using ESS data?"*

```bash
source venv/bin/activate
python scripts/run_padded_control.py --seeds 42,123,7 --agents 100 --rounds 30
```

- [ ] Launch now (fast, no LLM)
- [ ] Run `scripts/analyze_padded_vs_grounded.py` when done
- [ ] Add padded control row to ablation table (Section 6)

---

## When Results Land — Immediate Actions

### A/B 10-seed run finishes

```bash
python scripts/analyze_ab_results.py   # → analysis/tables/table2_ab_bootstrap.csv
# Update paper Section 8.1 + Table 2 with CI numbers
# Regenerate Section 8 figures
git add analysis/ docs/paper.md && git commit
```

### Condition D finishes

```bash
python scripts/analyze_condition_d.py  # → analysis/tables/table3_condition_comparison.csv
# Update paper Section 8.2 + Table 3 D-row
git add analysis/ docs/paper.md && git commit
```

---

## Recommended Immediate Sequence

| # | Task | Est. Time | Unblocks |
|---|------|-----------|----------|
| 1 | Write `scripts/analyze_ab_results.py` | 45 min | Table 2 the moment A/B lands |
| 2 | Write `scripts/analyze_condition_d.py` (A/B/C rows) | 30 min | Table 3 partial now, D row auto-fills |
| 3 | Generate 20 human eval vignettes from `rounds.jsonl` | 60 min | Starts 5-day Prolific clock |
| 4 | Fix HF cache path → env var | 10 min | External reproducibility |
| 5 | Launch padded control | 5 min | Ablation table gap |
| 6 | Start `docs/paper_aamas.tex` (headers + abstract) | 30 min | Venue commitment |
| 7 | Run `scripts/reformat_figures.py` | 20 min | Publication-quality figures |
| 8 | Commit `SECURITY.md`, `.env.example`, new test files | 5 min | Clean repo |

---

## Success Criteria (after all parallel work + GPU results)

- [ ] Table 2: B_RLHF / BRM / Gini with 10-seed bootstrap 95% CIs
- [ ] Table 3: All four conditions (A/B/C/D) with real numbers
- [ ] Human evaluation: n=40, % realism judgment reported
- [ ] Padded control ablation row added
- [ ] LaTeX draft ready for target venue
- [ ] Docker container tested
- [ ] OSF pre-registration filed
- [ ] All figures publication-quality (single-column ACM size, consistent style)

**At this point: the paper is submission-ready for AAMAS 2027 or NeurIPS 2026 workshop track.**
