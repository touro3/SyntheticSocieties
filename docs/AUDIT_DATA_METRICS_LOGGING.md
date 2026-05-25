# BGF Audit — Empirical Pipeline, Macro-Metrics, Logging

**Date:** 2026-05-20 · **Scope:** read-only audit · **Branch:** main @ 2d60d6c

Severity legend: 🔴 high (alters reported numbers / silently breaks a condition) · 🟠 medium (correctness latent or methodological) · 🟡 low (sentinel overload, reproducibility nit).

---

## Area 1 — Empirical data flow (ESS → agents → RAG)

### ✅ A1.1 ESS survey weights silently dropped — **FIXED 2026-05-25**
- **Original evidence:** raw `data/ESS11MD_e01_2.csv` ships `dweight, pspwght, pweight, anweight` (design / post-stratification / population / analysis). `data/ess_clean.parquet` schema (866 rows, 60 cols) contained **none** of these. `population/sampling.py:79` called `rng.choice(len(df), size=n, replace=True)` with no `p=`.
- **Original impact:** every "empirical" condition treated ESS respondents as i.i.d. draws from the target population, but ESS is a stratified weighted survey. Marginal moments (esp. trust, income decile) were biased by whichever country/age strata were over-represented in the unweighted parquet.
- **Fix:** added a `SURVEY_WEIGHTS` group (`anweight, pspwght, pweight, dweight`) to `data/ess_schema.py`, re-ran `scripts/ingest_ess.py` → `data/ess_clean.parquet` now ships all four weights. `population/sampling.py:96` auto-detects and applies `anweight` for the weighted draw. Verified: `sample_empirical_rows` logs `applying ESS survey weights from 'anweight'`. Also fixed a latent non-determinism in `generator.py` where the NaN→age fallback used the module-level `random` (caused reproducibility test to fail once weighted sampling pulled rows with NaN ages); now uses a seeded `Random(seed)`.

### 🔴 A1.2 NaN→fixed-default substitution distorts marginals
- `population/generator.py:160` — `age = _safe_int(row.get("age"), default=sample_age(min,max))` replaces NaN ages with a **uniform draw**, smearing the age distribution.
- `population/persona_synthesizer.py:101` — `income_decile = _safe_int(row.get("income_decile"), 5)` collapses every NaN decile to the **fixed median (5)**, spiking that bin.
- `persona_synthesizer.py:113` — `(income_decile or 5) * 400.0` repeats the collapse via Python truthiness (also collapses `decile==0` if it ever appears).
- **Evidence of scale:** in `ess_clean.parquet`, `left_right` has 88/866 NaN (10%), `trust_eu_parliament` 33, `trust_un` 48, `satisfaction_education` 46 — non-trivial. Any column read with `_safe_float(... , default=X)` is silently re-injected with X.

### 🟠 A1.3 Two divergent empirical paths producing the same agent — **PARTIALLY ADDRESSED 2026-05-25**
- **Original symptom:** `population/generator.py` and `population/persona_synthesizer.py` computed income (and historically also wealth) from the same ESS row with different formulas. Generator: ``decile * base_income(1000) * 2``; persona: ``(decile or 5) * 400`` — a 5× divergence at the same decile.
- **Fix:** both formulas are now routed through a single ``income_from_decile(decile, base_income, formula=...)`` and ``wealth_from_decile(decile, ...)`` in ``population/_helpers.py``. The two historical formulas are exposed as named ``formula="canonical"`` (persona, ``decile * base_income``) and ``formula="legacy_generator"`` (generator, ``decile * base_income * 2``) so a future unification sweep can flip the generator to canonical in a single deliberate change.
- **What remains:** flipping ``generator.py`` to ``formula="canonical"`` changes headline wealth/income numbers — deferred until after the in-flight N=500 LLM sweep finishes so reported pilot numbers stay bit-stable. Until then the divergence is documented and reviewable, not silent.

### 🔴 (original) A1.3 Two divergent empirical paths producing the same agent
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

### ✅ A2.1 JSD log-base mismatch — **FIXED** (commit f682af5, 2026-05-20)
- **Original symptom:** `metrics/distribution.py:66` called `stats.entropy(...)` with the scipy default (natural log), so JSD was reported in nats. `compute_brm_jsd` did `1 - jsd` and clamped to [0,1] → BRM compressed to [0.307, 1.0].
- **Fix in code:** `metrics/distribution.py:67` and `:108` now pass `base=2` to both `stats.entropy` calls; comment on L66 documents the rationale. JSD now in bits, BRM in [0, 1].
- **Caveat:** BRM/calibration_jsd numbers reported in pre-2026-05-20 runs are on the compressed scale and should be regenerated before any new paper figure ships.

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

### ✅ A3.1 Rotation hazard — **FIXED** (commit f682af5, 2026-05-20)
- **Original symptom:** rotated shards `events.0001.jsonl`, `events.0002.jsonl`, … were not read by any consumer; long runs were summarised from only the ~8% tail.
- **Fix in code:** `metrics/event_metrics.py:11 load_events` is now the rotation-aware chokepoint — globs `events.[0-9]*.jsonl`, sorts by shard index, concatenates with the active `events.jsonl`. All downstream readers (`trajectories.py`, `cross_model.py`, `llm_diagnostics.py`, `mediation_summary.py`, `mechanism_analysis.py`, network/padded plotters) go through it. Tested against synthetic multi-shard fixtures.
- **Caveat:** any `summary.json` written for a >200 MB run before 2026-05-20 is on a truncated event log; re-run `python scripts/build_summary.py <exp_id>` to refresh.

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

Recommended fix order: **A3.1 ✅ → A2.1 ✅ → A1.1 ✅ → A1.5 → A1.3 → A1.2 → A2.4 → A2.2/A2.3 → rest.** A3.1 and A2.1 silently corrupted the headline numbers in *all* published runs — both fixed 2026-05-20. A1.1 (ESS weights) fixed 2026-05-25. A1.* remaining changes are scientific, not bug fixes.
