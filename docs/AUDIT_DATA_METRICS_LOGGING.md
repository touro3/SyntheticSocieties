# BGF Audit — Empirical Pipeline, Macro-Metrics, Logging

**Date:** 2026-05-20 · **Scope:** read-only audit · **Branch:** main @ 2d60d6c

Severity legend: 🔴 high (alters reported numbers / silently breaks a condition) · 🟠 medium (correctness latent or methodological) · 🟡 low (sentinel overload, reproducibility nit).

---

## Area 1 — Empirical data flow (ESS → agents → RAG)

### 🔴 A1.1 ESS survey weights silently dropped
- **Evidence:** raw `data/ESS11MD_e01_2.csv` ships `dweight, pspwght, pweight, anweight` (design / post-stratification / population / analysis). `data/ess_clean.parquet` schema (866 rows, 60 cols) contains **none** of these. `population/sampling.py:79` calls `rng.choice(len(df), size=n, replace=True)` with no `p=`.
- **Impact:** every "empirical" condition treats ESS respondents as i.i.d. draws from the target population, but ESS is a stratified weighted survey. Marginal moments (esp. trust, income decile) are biased by whichever country/age strata are over-represented in the unweighted parquet.
- **Where it leaks:** `generator.generate_empirical_population`, `persona_synthesizer.synthesize_ess_personas`, and the `sql_rag` peer-group baseline all consume this same unweighted frame.

### 🔴 A1.2 NaN→fixed-default substitution distorts marginals
- `population/generator.py:160` — `age = _safe_int(row.get("age"), default=sample_age(min,max))` replaces NaN ages with a **uniform draw**, smearing the age distribution.
- `population/persona_synthesizer.py:101` — `income_decile = _safe_int(row.get("income_decile"), 5)` collapses every NaN decile to the **fixed median (5)**, spiking that bin.
- `persona_synthesizer.py:113` — `(income_decile or 5) * 400.0` repeats the collapse via Python truthiness (also collapses `decile==0` if it ever appears).
- **Evidence of scale:** in `ess_clean.parquet`, `left_right` has 88/866 NaN (10%), `trust_eu_parliament` 33, `trust_un` 48, `satisfaction_education` 46 — non-trivial. Any column read with `_safe_float(... , default=X)` is silently re-injected with X.

### 🔴 A1.3 Two divergent empirical paths producing the same agent
The codebase has **two** generators that should agree but don't:

| Field | `generator.py:117` (empirical) | `persona_synthesizer.py:91` (ESS personas) |
|---|---|---|
| income | `_safe_float(decile,0.5) * base_income * 2` | `(decile or 5) * 400.0` |
| wealth | `50 + (decile/10) * wealth_step * 10` (= 50+decile*10) | `50 + (decile/10) * 100` (= 50+decile*10) ✓ |
| trust_institutions | mean of **3** cols (parl/legal/police) | `_mean_institutions` → mean of **4** cols (incl. `trust_institutions` self) |
| NaN income decile | safe-default `0.5` (decile=0.5) | safe-default `5` |

Same conceptual variable, different numerical result depending on which entry point was used to build the population. Downstream conditioning (e.g., comparing Conditions A/B/C that internally take different paths) is confounded.

### 🟠 A1.4 Sample mode defaults to "resample" (with replacement) regardless of n
`generator.py:147` — `sample_mode = data_cfg.get("sample_mode", "resample")`. For n ≤ |df| (866), the statistically defensible default is `subsample` (no replacement). Current default introduces duplicate rows and inflated variance for any small-population run.

### 🔴 A1.5 RAG silently degrades to static narrative without a log signal
`decision/sql_rag.py:46–50, 69, 124, 274` — when `data/ess_clean.parquet` is absent or any query throws, `get_peer_group_context` returns `self.static_context` (or a generic "no peer group data" string). No warning logged, no flag in the event payload. **Result:** a Condition B run with a missing/broken parquet on the deploy target (HF Space, cloud) is indistinguishable from Condition A in `events.jsonl`. There is no `validation.rag_active` field to filter on.

---

## Area 2 — Macro-metric mathematical correctness

### 🔴 A2.1 JSD log-base mismatch (BRM, calibration, stability all affected)
`metrics/distribution.py:66` — `stats.entropy((p+q)/2) - (entropy(p)+entropy(q))/2`. `scipy.stats.entropy` defaults to **natural log**, so JSD is returned in **nats**, but the docstring (L45) claims "JSD value in [0, 1] (base-2 logarithm)". JSD-nats ranges in [0, ln 2 ≈ 0.693], so:
- `compute_brm_jsd` (`behavioral_realism.py:66`) → `max(0, min(1, 1 - jsd))` floors at `1 - 0.693 = 0.307`, never at 0. Two completely-disjoint distributions score BRM ≈ 0.307 instead of 0, **compressing the metric's discrimination range to [0.307, 1.0]**.
- `composite_brm.jsd_component` same compression.
- `metrics/calibration.py:255 calibration_jsd` same — reported "JSD" is in nats.
- `temporal_stability_jsd` same.

**Fix direction (do not apply here):** pass `base=2` to both `stats.entropy` calls. Any cached/published BRM numbers were computed under the wrong base and should be regenerated.

### 🟠 A2.2 `calibration_jsd` double-normalizes (scale information destroyed)
`metrics/calibration.py:253–254` — `sim_n = _minmax01(sim_wealth)`, `ess_n = _minmax01(ess_ref)`. Each array is independently min-max scaled, so calibration_jsd is **only a shape divergence**, not a calibration in the absolute sense. A simulation where wealth saturates around 100k will calibrate identically to one stuck at 0–10 if their *shapes* match. Documented in the docstring (good) but readers will still interpret it as calibration in the standard sense. Either rename to `calibration_shape_jsd` or use Wasserstein on the raw scale.

### 🟠 A2.3 `ess_wealth_reference` silent anchor fallback
`metrics/calibration.py:206–220` — wrapped in `except (OSError, KeyError, ValueError, TypeError): pass`. If `data/empirical_distributions.json` is missing/malformed, the hardcoded `v_anchor=[1,2,3,5,7,8,10]` is used with **no log**. Every calibration_jsd in such a run is computed against the wrong reference, indistinguishably from a correct one.

### 🟠 A2.4 Action vocabulary excludes `steal` from B_RLHF and trajectories
- `metrics/behavioral_realism.py:38` — `_ACTIONS = ("work", "save", "cooperate")`.
- `metrics/trajectories.py:113` — `a_types = ["work", "save", "cooperate"]`.

But `environment/payoffs.py` and `InstitutionManager` define `steal` as a valid action for adversarial agents. In `pipeline_bad_apple.sh` runs (5% adversaries), the entire `steal` activity is silently zeroed from B_RLHF normalization and from `action_freqs`. The displayed action distribution does not sum to total population × rounds.

### 🟡 A2.5 Gini post-clip can mask off-by-one
`metrics/inequality.py:45` — `np.clip(gini, 0, 1)`. Formula at L40 is the standard 1-indexed form and is correct for the unclipped path. With `correct_bias=True` (L43), `gini * n/(n-1)` can exceed 1.0 for unusual distributions and is silently clamped. Same clip would hide a future off-by-one. Suggest asserting in-range instead of clipping.

### 🟡 A2.6 Spearman uses ordinal not average ranks
`metrics/trust_gradient.py:128` — `np.argsort(np.argsort(x))`. For ties, this assigns ordinal ranks (1,2,3) rather than averaged ranks (e.g. 1.5,1.5,3). Current `TRUST_GROUPS` has 4 strictly distinct `ess_reference_mean` values so this never bites in the canonical call, but the helper is reused elsewhere — any future tied input silently biases ρ.

### 🟡 A2.7 Population vs sample variance mixed
Several call sites use `np.std(...)` (ddof=0, population) while reporting "std":
- `metrics/calibration.py:75` (`std_wealth`)
- `metrics/trajectories.py:143–144` (`wealth_std`, `stress_std`)

Reported "std" is therefore biased downward by √((n-1)/n). For n=100, that's ~0.5% — negligible per-run; for n=10 ablations it's ~5%. Document or convert to `ddof=1`.

### 🟡 A2.8 Network metrics sentinel/determinism issues
- `metrics/network_metrics.py:55–68 modularity` — `greedy_modularity_communities` is non-deterministic across networkx versions and seeds; results are not reproducible without pinning.
- `network_metrics.py:71–94 diffusion_speed` — silently restricts to the largest CC when disconnected; the returned scalar carries no coverage flag, so a network that fragments over time appears to maintain stable diffusion speed.
- `network_metrics.py:33, 46 assortativity` — returns 0.0 for both *undefined* (degenerate) and *neutral* assortativity. Downstream cannot distinguish "no data" from "zero correlation".

---

## Area 3 — Logging alignment

### ✅ A3.0 What's correct
- `simulation/round_processor.py:179–202` writes the event with `state_after = agent.state.snapshot()` — explicit post-update state, no off-by-one with action.
- `scripts/run_config_simulation.py:564–577` recomputes `summary.json` from `events.jsonl` via `load_events` (not from in-memory `round_metrics`). Single source-of-truth pattern is honored at this layer.
- `events.jsonl` payload carries `round_id`, `agent_id`, `action`, `validation`, `result`, `state_after`, plus a `harness_substitutions` field — clean schema for downstream replay.

### 🔴 A3.1 Rotation hazard — bulk of long runs silently dropped
`bgf_logging/event_logger.py:65–80` rotates `events.jsonl` → `events.0001.jsonl` etc. when a shard exceeds 200 MB. But **no consumer is rotation-aware**:

| Reader | File | Reads |
|---|---|---|
| `metrics/event_metrics.py:10 load_events` | summary builder | single path |
| `metrics/trajectories.py:27` | trajectories | `exp_path / "events.jsonl"` |
| `metrics/cross_model.py:90,115,156` | cross-model | single path |
| `metrics/llm_diagnostics.py:20` | LLM diag | single path |
| `analysis/mediation_summary.py:49` | mediation | single path |
| `analysis/mechanism_analysis.py:148` | mechanism | single path |
| `scripts/plot_network_evolution.py:181` | network replay | single path |
| `scripts/analyze_padded_vs_grounded.py:46` | padded ablation | single path |

A 500-agent × 10 000-round run is documented in `event_logger.py:8` as producing ~2.5 GB → ~13 shards. The active `events.jsonl` after rotation contains **only the tail (~200 MB ≈ 8 %)** of the events. `summary.json` for any such run is computed from ~8 % of the data, with no warning. **Severity: critical for Phase D (500-agent) and `pipeline_phase_c.sh`.**

**Fix direction:** every reader should glob `events*.jsonl`, sort by suffix, and concatenate. `load_events` is the single chokepoint — fix there.

### 🟠 A3.2 Witness manifest likely shares the rotation blind spot
`bgf_logging/witness.py` (referenced from `scripts/run_config_simulation.py:579`) — likely hashes only `events.jsonl`, so any rotated content is **not covered by the reproducibility hash**. Verify and extend.

### 🟠 A3.3 No fsync on EventLogger
`bgf_logging/event_logger.py:61` opens with `buffering=1` (line-buffered) but never calls `fsync`. On a hard kill (SIGKILL, OOM, GPU panic), the OS page cache may discard the most recent rounds. Crash-recovery in `simulation/crash_recovery.py` then resumes from a checkpoint that's *ahead* of the persisted event log → silent gap between checkpoint and resumed events.

### 🟠 A3.4 Action-vocabulary drift between writer and metric reader
The writer emits whatever `proposed_action.action_type` returns — including `steal`. The readers (`_ACTIONS`, `trajectories.a_types`) only know three actions. No assertion that the event vocabulary is a subset of the metric vocabulary. Add a startup-time check or extend `_ACTIONS`.

---

## Summary — discrepancies between agent actions and ingested empirical data

The audit found **three failure modes** where simulated behavior is misaligned with the empirical ground truth it claims to encode:

1. **Input bias (A1.1, A1.2):** ESS rows enter the simulation unweighted and NaN-patched. Agents are not draws from the ESS target population; they are draws from a flat, NaN-filled approximation of it. Marginal moments do not match published ESS marginals.
2. **Path divergence (A1.3, A1.5):** Two ESS-to-agent paths exist with different formulas; RAG can silently fall through to a static narrative. Two runs nominally in "Condition B" can be operating on different empirical signals — or none.
3. **Metric compression (A2.1, A2.4, A3.1):** The headline realism metrics (BRM_JSD, composite_brm, calibration_jsd) are computed in the wrong log base and silently clamped; B_RLHF and trajectory action_freqs ignore `steal`; long-run summaries are computed from ~8 % of events after the first rotation. Even a perfectly aligned simulation cannot score above ~0.69 in BRM under the current implementation.

Recommended fix order: **A3.1 → A2.1 → A1.1 → A1.5 → A1.3 → A1.2 → A2.4 → A2.2/A2.3 → rest.** A3.1 and A2.1 silently corrupt the headline numbers in *all* published runs; A1.* changes are scientific, not bug fixes.
