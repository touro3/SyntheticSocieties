# Changelog

All notable changes to the Behavioral Grounding Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ABOUT.md` — one-pager project summary for discoverability.
- `docs/datasheet_ess_synthetic.md` — Gebru-template datasheet for the synthetic-population artefact and upstream ESS Round 11 microdata.
- `docs/model_card_bgf.md` — Mitchell-template model card for the BGF decision-layer wrapper around Mistral-7B-Instruct-v0.3.
- `docs/adr/` — Architecture Decision Records bootstrapped with the index + initial records for the `Φ` map, BRM definition, RAG architecture, and conditions A–D.
- `docs/api/openapi.yaml` — OpenAPI 3.1 specification covering the Flask API endpoints.
- `paper.md` §9.A — *Broader Impacts and Responsible Use* section: intended uses, out-of-scope uses, structural mitigations shipped in the framework, and recommendations for downstream users.
- `scripts/decide_n500_pivot.py` — automated H1 + H_alt analysis script that fires the pivot-framing decision once the §8.1.3 N=500 LLM cells complete; supports `--watch` polling mode.
- `scripts/plot_multi_seed_bands.py` + `analysis/figures/multi_seed_bands.png` — §8.1.4 multi-seed bootstrap confidence-bands figure.
- `simulation/kernel.py::_write_validation_manifest()` — emits `validation.json` at run start to surface silent RAG-degradation incidents (audit A1.5). Detects whether SQLRAG and GraphRAG are operational and writes the result so downstream tooling can detect a Condition B run that silently collapsed to Condition A.
- `population/sampling.py::build_marginal_samplers()` + `sample_from_marginal()` — replace fixed-default NaN substitution with seeded resampling from the column's own non-NaN empirical marginal (weighted by ESS survey weights when present). Removes the historical median-bin spike documented in audit A1.2.
- `population/_helpers.py::income_from_decile(formula=...)` + `wealth_from_decile()` — single source of truth for the ESS decile → income / wealth mapping. The two historical formulas are exposed as named `formula="canonical"` (persona path) and `formula="legacy_generator"` (generator path) so a future unification sweep can flip the generator deliberately rather than silently (audit A1.3).
- `data/ess_schema.py::SURVEY_WEIGHTS` group — re-introduces `anweight`, `pspwght`, `pweight`, `dweight` into the cleaned parquet so `population/sampling.py` applies weighted draws automatically (audit A1.1).
- `.devcontainer/devcontainer.json` — VSCode dev-container config for one-click reproducible local development.
- `.github/workflows/security.yml` — security workflow: `pip-audit` (dependency CVEs), `bandit` (Python SAST), weekly cron + on-PR.
- `Makefile` targets: `security`, `bandit`, `openapi-lint`, `adr-new`.
- `CHANGELOG.md` (this file) + `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1).

### Changed
- `docs/AUDIT_DATA_METRICS_LOGGING.md` — A1.1 / A2.1 / A3.1 are marked ✅ FIXED with citations to the patched lines; the recommended-fix-order list is updated to reflect remaining open items.
- `pyproject.toml` — added explicit `[tool.ruff]` and `[tool.mypy]` sections (line-length 120, selected rule families).
- `.github/workflows/ci.yml` — Python matrix expanded from 3.12-only to `[3.10, 3.11, 3.12]` so version-specific bugs are caught before submission.
- `docs/paper.md` — Table 1 + Figures 3 / 4 captions now carry explicit `(n=1 pilot, no CI)` annotations; the Table 1 caveat references the §8.1 N=100 falsification of H2.

### Fixed
- **Audit A1.1 — ESS survey weights silently dropped.** `data/ess_clean.parquet` now ships all four ESS weight columns; sampling auto-detects `anweight` and applies weighted draws. Every "empirical" agent population is no longer an unweighted draw from a stratified survey.
- **Latent reproducibility regression in `population/generator.py`** uncovered while patching A1.1: the NaN→age fallback called `sample_age(...)` without an rng, falling back to the module-level `random`. Two `generate_empirical_population(seed=42)` calls in the same process produced different agents whenever a sampled row had a NaN age. Now uses a seeded `random.Random(seed)`.

### Known issues
- §8.1.3 N=500 LLM A vs B sweep (10 seeds) still in flight on GPUs 0 / 2 / 3 (~5 % complete after 17h, ETA ~13 days). `scripts/decide_n500_pivot.py --watch` will fire the pivot-framing analysis automatically once cells complete.
- §8.4 Prolific human-evaluation study (calibrated `π_human` reference for B_RLHF) pending IRB approval — no data collected yet.
- Audit findings A1.2 partially addressed (NaN substitution now uses marginal resampling); A1.5 partially addressed (validation manifest emitted); A1.3 documented + routed through single helper but generator path still uses legacy formula to keep published numbers bit-stable until a deliberate unification sweep after N=500 finishes.

---

## [1.0.0] — 2026-04-14

Initial public release accompanying `docs/paper.md`.

### Highlights
- Formal framework `BGF = (A, E, G, P, Φ, T)` with type-safe `PolicyProtocol` interfaces, Pydantic configs, and 1500+ automated tests.
- Four experimental conditions: A (pure LLM), B (grounded LLM via ESS-RAG), C (generative-agents proxy), D (deterministic rule-based ESS policy).
- Behavioural Realism Metric `BRM ∈ [0, 1]` + RLHF Cooperative Bias `B_RLHF ∈ [0, 2/3]`.
- One-command paper reproduction via `bash scripts/reproduce_paper.sh --full`.
- Flask REST API + Vue SPA deployed on HuggingFace Spaces (`touro3/synthetic-societies`) for live agent interviews, anchor queries, AI-designed scenarios.
- Cross-cultural cooperation gradient recovered at Spearman ρ = +1.000 across six ESS clusters; independent WVS Wave 7 (r = +0.977) and Herrmann–Thöni–Gächter PGG (Spearman ρ = +0.886, p = 0.033) replications.
- 20+ evaluation modules: distributional similarity (JSD/KL/Wasserstein), network metrics, persona fidelity, trust gradient, mediation analysis, complexity (phase transitions + power laws), persona decay, behavioural realism.

### Limitations at 1.0
- All four critical audit bugs identified on 2026-05-20 (rotation, JSD base, ESS weights, ESS→agent path divergence) shipped silently; fixes land in the next minor release. Pre-2026-05-20 BRM values are on a compressed [0.307, 1.0] scale and should be regenerated.
- N=500 LLM A vs B and human-eval reference distribution both pending.

---

[Unreleased]: https://github.com/lucastourinho/SyntheticSocieties/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lucastourinho/SyntheticSocieties/releases/tag/v1.0.0
