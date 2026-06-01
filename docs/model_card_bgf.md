# Model Card — BGF Decision Layer

Following Mitchell et al. (2019) "Model Cards for Model Reporting."
Describes the **decision-layer wrapper** that BGF places around a base
LLM (Mistral-7B-Instruct-v0.3 by default). For the upstream Mistral
model card itself, refer to `mistralai/Mistral-7B-Instruct-v0.3` on
HuggingFace.

---

## Model Details

- **Developed by:** the BGF authors (`github.com/lucastourinho/SyntheticSocieties`).
- **Layer of analysis:** BGF wraps a base instruction-tuned LLM with
  (i) per-agent persona prompts derived from ESS Round 11 microdata,
  (ii) optional ESS-RAG context (SQL + Graph), (iii) hierarchical
  agent memory, (iv) anti-hallucination JSON parsing with
  rule-based fallback, (v) a deterministic institutional gate that
  rejects rule-violating actions. The base LLM is treated as a black
  box; BGF does no weight-level training and ships no checkpoints.
- **Base models exercised:** Mistral-7B-Instruct-v0.3 (primary;
  HuggingFace + vLLM backends), Qwen2.5-7B-Instruct (cross-model
  panel), GPT-4o-mini (cross-model panel, OpenAI API).
- **Model version:** BGF wrapper at repository revision recorded in
  every run's `witness.json`.
- **Type:** Decision-layer wrapper for instruction-tuned LLMs, used
  inside multi-agent economic simulations.
- **Licence:** MIT (BGF wrapper). Base-model licences apply to their
  respective weights (Mistral Apache-2.0; Qwen2.5 Apache-2.0;
  GPT-4o-mini OpenAI ToS).
- **Contact:** `tourinholucas123@gmail.com` / GitHub issue tracker.

---

## Intended Use

- **Primary intended use:** measuring whether grounding an LLM agent
  in empirical survey-microdata-derived personas changes its
  behaviour in a multi-agent economic simulation, relative to the
  same agent without grounding.
- **Primary intended users:** researchers comparing LLM-agent
  cooperation / inequality / network-topology outputs against
  empirical references; researchers benchmarking the RLHF cooperative
  bias of new instruction-tuned models.
- **Out-of-scope use cases:** policy forecasting; individual-level
  prediction; real-time decision systems; any deployment whose
  decisions affect real humans without an independent
  empirical validation step. See §9.A of `docs/paper.md` for the full
  list.

---

## Factors

- **Relevant population groups represented in the persona space:**
  ESS Round 11 demographic strata — age, gender, education level,
  income decile, country, urbanisation, political orientation,
  religiosity, life satisfaction, social activity. The local
  microdata parquet is **Austria-only (n = 866)**; cross-country
  claims rely on cluster-benchmark-level fallback rather than
  per-country microdata.
- **Instrumentation factors:** prompt token budget (default 3072,
  trim order: `social_context` → `population_context`), temperature
  (per `configs/`), batch size (default 16, `BGF_MAX_BATCH_SIZE`
  override), quantisation (none by default; vLLM optionally fp16),
  context window (depends on base model — 8192 for Mistral-7B v0.3).
- **Evaluation conditions:** four pre-registered conditions A / B /
  C / D (`docs/paper.md` §3.10).

---

## Metrics

- **Behavioural Realism Metric (BRM ∈ [0, 1])** — composite of
  wealth-JSD, Gini-gap, cooperation accuracy, temporal stability.
  See `metrics/behavioral_realism.py`. *Note:* pre-2026-05-20 BRM
  values are on a compressed [0.307, 1.0] scale because of a JSD
  log-base bug (audit A2.1, fixed); regenerate before reporting.
- **RLHF Cooperative Bias (`B_RLHF` ∈ [0, 2/3])** — total-variation
  distance from a uniform reference distribution over
  {work, save, cooperate}. See `metrics/behavioral_realism.py:
  compute_rlhf_bias_index`. **Limitation:** uniform reference
  overestimates true bias magnitude; the pre-registered §8.4 Prolific
  human-evaluation study (pending IRB) will provide a calibrated
  `π_human` reference.
- **Cross-cultural cooperation gradient** — Spearman ρ between
  simulated cooperation rate and ESS-11 cluster-mean interpersonal
  trust. See `metrics/cross_cultural.py`.
- **Standard reference statistics** — Gini, cooperation rate,
  network modularity (Newman Q), assortativity, mean wealth.

---

## Evaluation Data

- **ESS Round 11 microdata (AT subset, n = 866 respondents):**
  empirical anchor for `Φ: D_ESS → Profile` and for the per-cohort
  SQL-RAG queries. See `data/ess_clean.parquet` (regenerate from
  `scripts/ingest_ess.py`).
- **WVS Wave 7 cluster aggregates:** out-of-sample replication
  reference for the cross-cultural gradient (§8.3 of `docs/paper.md`).
- **Herrmann-Thöni-Gächter (2008) per-city PGG contributions:**
  external behavioural benchmark BGF never ingests, used as an
  independent replication target (Spearman ρ = +0.886, p = 0.033).

---

## Training Data

- **Base model:** trained by Mistral AI on its proprietary corpus;
  see the upstream Mistral-7B-Instruct-v0.3 model card. BGF does
  no further training, fine-tuning, or weight modification.
- **BGF wrapper:** no training data; the wrapper is a deterministic
  prompt-construction + parsing + institutional-gate pipeline whose
  behaviour is fully specified by `configs/base_config.yaml` and the
  Python source under `decision/`, `population/`, `environment/`.

---

## Quantitative Analyses

- **Unit-test coverage:** 1541 tests pass at HEAD (`pytest tests/
  --no-cov -p no:cacheprovider`, excluding GPU-only `test_llm_policy`
  / `test_cross_model`).
- **Hypothesis tests:** H1–H9 pre-registered (see
  `docs/hypothesis_preregistration.md`). Multiple-testing
  correction via Benjamini-Hochberg FDR at α = 0.05; permutation
  p-value used for the n = 6 cluster Spearman ρ; bootstrap 95% CIs
  (2000 resamples, fixed seed 42, percentile method) on every
  reported point estimate.
- **Latest status:** H5 (trust-gradient post-hoc) and H9
  (cross-cultural PGG) confirmed per-test; H2 (B_RLHF reduction at
  N=100) falsified; H1, H3, H4, H6, H8 pending the in-flight N=500
  LLM sweep.

---

## Ethical Considerations

- **Privacy:** no individual-level outputs are produced; ESS
  respondent IDs are dropped at ingestion.
- **Fairness:** the local microdata is AT-only — cross-country
  generalisations are gated through the cluster-benchmark fallback
  and labelled as such throughout the paper.
- **Safety:** the simulation has no real-world side effects; the
  framework's only outputs are JSONL traces and analysis figures.
- **Dual-use:** see §9.A "Broader Impacts and Responsible Use" in
  `docs/paper.md`. BGF is **hypothesis-generating, not
  policy-evaluating**.
- **Human-subjects work:** the §8.4 Prolific human-evaluation study
  is pending IRB approval and has not yet collected data.

---

## Caveats and Recommendations

- All pre-2026-05-20 BRM values are on a compressed [0.307, 1.0]
  scale and should be regenerated under the patched
  `metrics/distribution.py` before citation.
- All pre-2026-05-25 "empirical" runs used unweighted draws from
  the ESS parquet (audit A1.1). Marginal moments are biased; rerun
  with the patched `data/ess_clean.parquet` for unbiased estimates.
- The two historical income formulas in `population/generator.py`
  (legacy `decile × base_income × 2`) and
  `population/persona_synthesizer.py` (canonical `decile × 400`)
  differ by 5× at the same decile. Both routed through
  `population/_helpers.income_from_decile()` as of 2026-05-25; the
  generator path still uses `formula="legacy_generator"` to keep
  published numbers bit-stable until a deliberate unification sweep
  after the in-flight N=500 LLM run finishes.
- Cite the framework, the pre-registration, **and** the limitations
  section together; do not cite a single headline number in
  isolation.

---

**Citation:** see `CITATION.cff`. **Upstream model card:**
`huggingface.co/mistralai/Mistral-7B-Instruct-v0.3`. **Audit erratum:**
`docs/AUDIT_DATA_METRICS_LOGGING.md`.
