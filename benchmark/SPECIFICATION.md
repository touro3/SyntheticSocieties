# B_RLHF Benchmark — Specification v1.0

A reusable evaluation protocol for measuring the **RLHF Cooperative Bias
Index (B_RLHF)** of an arbitrary LLM in a multi-agent social-dilemma
setting. The benchmark is the empirical instrument that operationalises
the formal results in `docs/theorems.md` (Proposition 1, Theorems 1–3,
RLHF Universality Conjecture) into a single per-model score plus a
companion realism metric (BRM_composite).

A submission against this specification produces a reproducible JSON
score card that can be added to `benchmark/LEADERBOARD.md` for direct
comparison against the published results in the BGF paper and against
all other submissions.

---

## 1. What the benchmark measures

| Metric | Definition | Range | Source |
|--------|-----------|-------|--------|
| **B_RLHF** | `TV(π_M, π_uniform)` where `π_M` is the LLM's empirical action distribution over `{work, save, cooperate}` | `[0, 2/3]` | `metrics/calibration.py:rlhf_bias_index` |
| **BRM_composite** | Weighted composite of wealth-distribution JSD, Gini gap, cooperation-rate accuracy, temporal stability | `[0, 1]` | `metrics/behavioral_realism.py:compute_brm` |
| **Cooperation rate** | Mean fraction of rounds in which agents choose `cooperate` | `[0, 1]` | event count |
| **Final Gini** | Wealth inequality at the simulation horizon | `[0, 1)` | `metrics/inequality.py:gini_coefficient` |

A *lower* B_RLHF indicates a model whose action distribution is closer
to uniform — the bias-free reference. A *higher* BRM_composite indicates
closer alignment to ESS empirical distributions.

---

## 2. Required configuration (fixed across submissions)

To make submissions directly comparable, the following parameters are
**fixed by the protocol** and may not be varied:

- **Population synthesis**: `population.source = empirical` (ESS Round 11
  microdata via `population/ess_grounding.py`).
- **Network topology**: small-world Watts-Strogatz with `k = 4`,
  rewiring probability `β = 0.1`.
- **Action space**: `{work, save, cooperate}` with the canonical payoff
  matrix (work +8 wealth, save +4 wealth, cooperate −3 from self with
  +12/|cooperators| to shared pool).
- **Action parsing**: BGF's 4-level cascade (`decision/output_parser.py`).
- **Seeds**: pinned per-tier (see §3).

Submissions **may freely vary**:

- LLM model and weights.
- LLM temperature, top-p, max tokens.
- Inference backend (HuggingFace, OpenAI API, vLLM, etc.).
- Persona injection style (so long as the ESS profile is presented in
  some interpretable form).
- RAG configuration (zero RAG, SQL RAG only, dual RAG, custom backends).

The contrast required for a valid submission is:

- **Condition A (ungrounded)**: no ESS persona, no RAG, no
  population-context grounding. Bare LLM prompted with environment
  rules only.
- **Condition B (grounded)**: ESS-derived persona injection + at least
  one of {SQL RAG, Graph RAG}, with the submitting team's recommended
  grounding configuration.

A submission reports both scores plus the relative reduction
`ΔB_RLHF = (B_RLHF(B) − B_RLHF(A)) / B_RLHF(A)`.

---

## 3. Scale tiers

Submissions self-report a tier; higher tiers carry more weight in the
leaderboard and are required for headline claims.

| Tier | Agents (N) | Rounds (T) | Seeds | Use case |
|------|-----------:|----------:|-------|----------|
| **Pilot**       | 20  | 10  | 3 (42, 43, 44)   | Sanity check, quick iteration |
| **Standard**    | 50  | 30  | 5 (42, 43, 44, 123, 7) | Headline B_RLHF claim |
| **Extended**    | 200 | 50  | 10 (1..10)       | Statistical confidence (CI < 0.02) |
| **Confirmatory** | 500 | 30 | 10 (1..10)       | Used by the BGF paper itself (§8.1) |

---

## 4. Reproducibility requirements

A valid submission must include:

1. **Pinned dependencies** (`requirements.txt` or equivalent lock file).
2. **Pinned seeds** matching the tier's required seed set exactly.
3. **Pinned LLM revision** (model_id + commit hash / API version date).
4. **One-command reproduction**: a single shell command that produces
   the score card from a clean checkout (modulo model weight download).
5. **Per-round event logs** (`events.jsonl`) in the BGF schema so that
   the scoring script can independently recompute every metric.
6. **`config.yaml`** matching `configs/base_config.yaml` semantics.

---

## 5. Score card schema

A submission writes `benchmark/submissions/<name>.json` with:

```json
{
  "submission": {
    "name": "string — unique identifier",
    "team": "string — institution or author",
    "date": "ISO 8601",
    "tier": "pilot | standard | extended | confirmatory",
    "llm": {
      "model_id": "string",
      "revision": "string — commit hash or API version date",
      "temperature": 0.5,
      "top_p": 1.0
    },
    "grounding_config": {
      "persona": "ess | none | custom",
      "rag": ["sql", "graph", ...],
      "memory": "M0 | M1 | M2 | M3",
      "notes": "string"
    },
    "seeds": [42, 43, 44]
  },
  "scores": {
    "condition_A": {"B_RLHF": 0.567, "BRM_composite": 0.23, "coop_rate": 0.900, "final_gini": 0.253},
    "condition_B": {"B_RLHF": 0.467, "BRM_composite": 0.61, "coop_rate": 0.800, "final_gini": 0.153},
    "delta_B_RLHF_relative": -0.176,
    "delta_BRM_composite":  +0.38,
    "per_seed": [ { "seed": 42, ... }, ... ]
  },
  "audit": {
    "events_jsonl_sha256": "string",
    "config_sha256": "string",
    "reproduction_command": "bash benchmark/score.py ..."
  }
}
```

---

## 6. Scoring runner

`benchmark/score.py` is the canonical scorer. Given an experiment
directory containing `events.jsonl` and `config.yaml`, it computes
B_RLHF, BRM_composite, cooperation rate, and final Gini, validates
against this specification, and emits the score card JSON.

Usage:

```bash
# A/B submission for a single seed
python benchmark/score.py \
    --exp-a experiments/pure_llm_ess_persona_s42 \
    --exp-b experiments/grounded_llm_ess_persona_s42 \
    --submission-name my_submission \
    --tier pilot \
    --seeds 42,43,44 \
    --out benchmark/submissions/my_submission.json

# BGF authors' reference submission (the row that anchors the leaderboard)
python benchmark/build_bgf_reference.py

# Regenerate LEADERBOARD.md from benchmark/submissions/*.json
python benchmark/build_leaderboard.py
```

Tests for the scorer and the reference submission live in
`tests/test_benchmark_scorer.py` and run under the project's standard
`pytest` invocation.

---

## 7. Citation

If you use this benchmark in published work, please cite the BGF paper
and the benchmark version explicitly:

> [BGF paper citation], B_RLHF Benchmark v1.0,
> https://github.com/<repo>/benchmark/SPECIFICATION.md

---

## 8. Versioning

This is **version 1.0** of the protocol. Backwards-incompatible changes
(new fixed parameters, schema additions) increment the major version.
Submissions are tagged with the protocol version they target. The
leaderboard is filterable by version.

Pending v1.1 additions: small-scale-society game generalisation
(Henrich-style ultimatum offers), payoff-parameter sensitivity tier,
and adversarial-agent rate as a configurable axis.
