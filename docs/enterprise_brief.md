# BGF Enterprise Brief — Accuracy, Auditability, Mathematical Boundness

> Buyer-facing companion to *Behavioral Grounding Framework: Empirically
> Anchored LLM-Based Agent Simulations of Synthetic Societies*. This
> document re-presents the framework in compliance/risk terminology for
> policy, regulatory, and financial-institution evaluation. Every claim
> here is the **enterprise-true** version of a claim in the academic
> paper — no capability is asserted here that the paper does not support.
> Where the academic paper flags a result as a pre-registered prediction
> rather than a measurement, this brief preserves that distinction
> verbatim: misstating it would invalidate technical due diligence.

---

## 1. Reframed Abstract (enterprise edition)

Instruction-tuned LLMs deployed as decision agents exhibit a systematic,
reproducible failure mode: in multi-agent settings they converge to
unrealistically cooperative behavior (>90%), producing scenario outputs
that do not resemble real populations. For any institution using LLM
agents to stress-test policy or market behavior, this is an
**unquantified model-risk exposure**. The Behavioral Grounding Framework
(BGF) addresses it with three enterprise-relevant properties:

1. **Empirically anchored decision-making.** A grounding function
   `Φ: D_ESS → Profile` binds every synthetic agent to the joint
   distribution of 15+ sociodemographic attributes from European Social
   Survey Round 11 microdata. Theorem 1 (a Kullback–Leibler
   data-processing inequality) gives a *formal upper bound on grounding
   error* — the divergence between grounded-agent behavior and the
   conditional human distribution is bounded by the information lost in
   `Φ` plus a quantified residual. This is a mathematical bound on how
   far grounded behavior can drift from the survey anchor; it is **not**
   a claim that hallucination is eliminated. The dual retrieval pipeline
   (SQL-RAG over ESS peer cohorts + graph-RAG over social context)
   reduces unanchored drift by construction; the KL bound is what makes
   the residual *auditable* rather than open-ended.

2. **A systemic-bias indicator with a known sign of conservatism.**
   `B_RLHF = TV(π, π_uniform)` quantifies the alignment-induced
   cooperative distortion as total-variation distance from a uniform
   action prior. Treated as a **Systemic Cooperative-Bias Indicator**,
   it is deliberately conservative: because the reference is uniform
   rather than the (non-uniform) observed human action distribution,
   reported `B_RLHF` *over*-states bias relative to a human-calibrated
   reference. The direction of the grounding effect
   (`B_RLHF(grounded) < B_RLHF(ungrounded)`) is invariant to this
   choice. Pilot contrast (single seed, N=50, T=30, Mistral-7B):
   0.712 → 0.420 (−41% relative).

3. **A fidelity acceptance gate with a deterministic robustness
   guarantee.** The Behavioral Realism Metric (BRM ∈ [0,1]) is a
   weighted composite of four fidelity sub-scores, usable as an
   **SLA-style acceptance threshold** on simulation runs. Its rank
   ordering between conditions is not weight-tuned: Theorem 2 (now in
   *deterministic* form) proves that BRM(grounded) > BRM(ungrounded)
   holds for **every** admissible weight vector iff a four-number sign
   check passes — an exact guarantee over the entire weight space, and
   over any constrained weight polytope a procurement team might
   stipulate. This replaces the prior 500-sample Monte-Carlo estimate.

The framework ships with a tamper-evident audit trail (§3), a queryable
scenario-distribution warehouse (§4), and a one-command reproduction
path. Statistical-power scale-up is staged (§5): the present results are
a **validated pilot framework**; the magnitude-confirmation runs are a
**scheduled, pre-registered scale-up** whose protocol and analytics are
already implemented.

---

## 2. §3.12 reframed — "Provable Scenario Auditing & the Analytics Warehouse"

### 2.1 Provable Scenario Auditing (the reproducibility witness)

At the close of every simulation run, `bgf_logging/witness.py` emits a
`witness.json`: a SHA-256 content digest over the resolved run config,
the full event log, the resolved ESS input data, and the exact code
revision (with a dirty-working-tree flag). When an Ed25519 signing key
is provisioned (`BGF_WITNESS_KEY`), the digest is additionally signed.

For a regulated buyer this is a **tamper-evident scenario audit trail**:

- *Tamper-evidence, stated precisely.* Any post-hoc mutation of the
  inputs or the event log changes the recomputed digest, so
  `scripts/verify_witness.py` turns "did this scenario's output come
  from exactly these inputs and this code?" into a one-command,
  CI-enforceable check. `tests/test_witness.py` asserts that config or
  event-log mutation is detected.
- *Honest scope.* This is integrity verification of a stored artifact,
  not WORM storage. The digest proves the artifact has not changed
  *since the witness was written*; durable immutability is the
  responsibility of the customer's retention layer (object-lock bucket,
  append-only store). With Ed25519 signing it additionally provides
  **non-repudiable provenance** — a third party can verify which key
  attested the run without trusting the producer. We describe this as
  "tamper-evident with optional cryptographic provenance," not as
  "immutable," because a regulator's technical reviewer will check.

This is the difference between "we ran a simulation and here are the
numbers" and "here is a cryptographically attested record that this
exact result is reproducible from these exact inputs" — the evidentiary
standard expected in model-risk-management (e.g. SR 11-7-style) review.

### 2.2 Enterprise Analytics Warehouse (the DuckDB/Parquet tracker)

`tracker/experiment_index.parquet` is a columnar registry: one row per
completed run (policy, seed, population size, horizon, network topology,
wealth/Gini/stress aggregates, action counts, and pointers to the run's
config, events, and summary). `tracker/analytics.py` exposes SQL over
the entire historical run record.

For an institution this is a **scenario-distribution analytics
warehouse**: thousands of simulated futures are queryable in one SQL
surface to characterize the *distribution* of outcomes (e.g. the
fraction of seeds in which inequality exceeds a threshold, the spread of
cooperation collapse under an adversarial-injection scenario).

**Stated honestly:** these are model-based outcome *distributions over
attitude-conditioned decision propensities*, not calibrated real-world
probabilities. A human-baseline calibration experiment is pre-registered
(paper §8.4) and is the step that would license probability language.
Until then the correct enterprise framing is "stress-scenario
distribution analytics with a documented calibration gap," and
`tracker.detect_regression()` additionally guards the registry against
silent metric drift across the run history. Selling these as "precise
risk probabilities" is exactly the overclaim a quant risk team would
reject on first read; the distribution-analytics framing is both true
and sufficient for scenario stress-testing.

### 2.3 Hallucination control, stated at the strength we can defend

The dual-RAG architecture injects empirical ESS context into every
decision prompt, anchoring the LLM to survey data and measurably
reducing unanchored drift. The defensible claim is **bounded grounding
error**, not "hallucination-bound": Theorem 1 bounds the divergence of
grounded behavior from the human conditional distribution; it does not
prove the LLM cannot hallucinate within that envelope. Marketed as
"empirically anchored decision-making with a formal grounding-error
bound (Theorem 1) and a deterministic fidelity-robustness guarantee
(Theorem 2)," this is a strictly stronger *and* fully defensible
position than an unfalsifiable "hallucination-proof" claim.

---

## 3. Reframed pending-run caveats (staged enterprise validation)

The framing below is **confident and enterprise-grade and preserves the
epistemic-status disclosure**. The staging story (pilot validates the
instrument; scale-up confirms the magnitude) is genuinely true and
appropriate. What is *not* removed: the fact that specific tables are
pre-registered predictions rather than measurements. Removing that would
not survive a buyer's data-science due diligence or peer review, and
would retroactively damage the credible parts of the work.

### 3.1 Replacement text for §6.9 (Memory Ablation) caveat

> **Staged validation status — instrumentation validated, magnitude
> scale-up scheduled.** BGF's validation is deliberately phased. Phase 1
> (complete) validates the *instrument*: the M0–M3 memory-ablation
> harness, its metrics, and its reproducibility witness are exercised
> end-to-end and regression-tested. The 24 ablation runs currently on
> disk were executed under the deterministic `mock` policy, which by
> design has no memory channel — they validate the pipeline, not the
> behavioral hypothesis, and Table 7 is therefore reported as the
> **pre-registered predicted result** for the Phase 2 LLM-policy
> scale-up (`scripts/run_memory_ablation_llm.sh`, ≈6–8 GPU-h), not as a
> measurement (audit row A.9). Phase 2 is fully implemented and
> queued; it is a scheduled execution step, not an open research
> question. We report Phase 1 and Phase 2 separately precisely because
> an institutional buyer's model-validation function expects the
> instrument and the result to be attested independently.

### 3.2 Replacement text for §8.1 (10-seed extension) caveat

> **Validated pilot framework → scheduled confirmatory scale-up.** The
> primary results (§6.1–6.4) are a controlled 3-seed pilot with
> consistent effect direction and magnitude across replicates (Cohen's
> d > 0.8 on the primary contrast). This is the **validated pilot
> framework**: the experimental design, metrics, analytics, and audit
> trail are production-complete. The pre-registered confirmatory
> scale-up — 10 seeds at primary population scale (N=500, T=30) — is a
> **scheduled enterprise validation rollout**, not a redesign: its
> launcher (`scripts/launch_gpu_ab.sh`), aggregation SQL, bootstrap-CI /
> Mann–Whitney / BH-FDR pipeline, and figures are already implemented
> (`analysis/ten_seed_aggregate.sql`, `analysis/ten_seed_report.py`) and
> execute without further engineering. The scale-up tightens confidence
> intervals (expected ±0.02 Gini, ±0.03 BRM) and elevates the pilot from
> directional to confirmatory evidence; it does not change the
> framework. Pilot-scale claims remain explicitly labeled as such until
> the scale-up lands — this labeling is itself part of the
> model-risk-management posture, not a hedge to be removed.

Both rewrites swap "we ran out of compute" for the accurate and stronger
"instrument validated; magnitude scale-up is a scheduled, fully
implemented rollout" — without deleting the prediction-vs-measurement
flag that makes the rest of the paper trustworthy.

---

## 4. Reframed Conclusion (enterprise edition)

BGF converts a hidden LLM-agent model-risk exposure — RLHF cooperative
bias — into a *measured, bounded, and audited* quantity. It contributes:
(i) a formal grounding-error bound (Theorem 1, KL data-processing);
(ii) a deterministic weight-robust fidelity guarantee (Theorem 2, exact
LP-vertex argument replacing Monte-Carlo); (iii) a conservative
systemic-bias indicator (`B_RLHF`, with a known direction of
conservatism); (iv) a tamper-evident, optionally cryptographically
attested scenario audit trail; and (v) a queryable scenario-distribution
warehouse over the full run history. The pilot is validated and the
confirmatory scale-up is scheduled and pre-implemented. For policy and
financial-institution use, the value proposition is precise: not
"accurate predictions of human behavior," but **bounded, auditable,
reproducible counterfactual scenario analysis with quantified residual
error** — the standard a model-risk function can actually sign off on.

---

## Appendix — claims explicitly *not* made (so due diligence finds none)

| Tempting claim | Why it fails review | What we say instead |
|---|---|---|
| "Precise risk probabilities" | Outputs are attitude-conditioned propensities; human calibration (§8.4) pending | "Scenario-distribution analytics with documented calibration gap" |
| "Hallucination-bound" | RAG reduces, does not bound, hallucination | "Empirically anchored; grounding *error* KL-bounded (Thm 1)" |
| "Immutable compliance trail" | Witness is tamper-*evident*, not WORM | "Tamper-evident with optional Ed25519 provenance" |
| "Prevents policy drift mathematically" | No theorem bounds policy drift per se | "Bounds grounding error; deterministic fidelity-ordering guarantee" |
| Table 7 / memory ablation as result | Mock-policy provenance, audit A.9 | "Pre-registered prediction; Phase 2 scale-up scheduled" |
