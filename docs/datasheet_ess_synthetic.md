# Datasheet — BGF Synthetic-Population Dataset

Following Gebru et al. (2018) "Datasheets for Datasets." Describes the
synthetic-population artefact that BGF emits when run in empirical mode,
and the upstream ESS Round 11 microdata it is derived from.

---

## Motivation

**For what purpose was the dataset created?**
The synthetic-population artefact (`experiments/<exp_id>/events.jsonl` +
`agents.json`) is the agent-state trace of one BGF simulation run. It
exists to let other researchers replicate, audit, and re-analyse BGF
experiments without rerunning the simulation. The upstream ESS Round 11
microdata is the empirical grounding source — agent profiles are sampled
from it via `population/sampling.py`.

**Who created the dataset and on behalf of which entity?**
The synthetic outputs are produced by the BGF framework
(`github.com/lucastourinho/SyntheticSocieties`). The upstream ESS Round
11 microdata is collected and distributed by the European Social Survey
ERIC (`europeansocialsurvey.org`); BGF does not redistribute the raw ESS
microdata.

**Who funded the creation of the dataset?**
BGF is a capstone research project; no external funding. ESS Round 11
is funded by the European Commission, the European Science Foundation,
and participating national research-funding bodies (see ESS Round 11
documentation for the full funder list).

---

## Composition

**What do the instances represent?**
Each instance in `events.jsonl` is one (agent, round, action) tuple
emitted during a simulation run. Each instance in the synthesised
population (in-memory `Agent` objects) is one synthetic agent with 15+
ESS-derived attributes (age, gender, country, income decile, trust,
political orientation, life satisfaction, social activity, …) and a
randomly initialised wealth and memory state.

**How many instances?**
Per run: `n_agents × n_rounds × ~2.5` event records (one per agent per
round + housekeeping events). A 500-agent × 30-round run produces ~38k
events / ~30 MB. A 500-agent × 10 000-round run produces ~2.5 GB; the
event log auto-rotates to ≤200 MB shards (see
`docs/AUDIT_DATA_METRICS_LOGGING.md` A3.0–A3.1).

**Is the data a sample or the full population?**
The upstream ESS Round 11 microdata shipped in this repo is the **Austria
(AT)-only** subset (`data/ess_clean.parquet`, 866 respondents). The
synthesised population is drawn with replacement from this subset
weighted by `anweight` (analysis weight; see audit A1.1). The local
parquet is **not representative of ESS-11 as a whole** — it is a
single-country subset for which microdata redistribution was
permissible. Cross-country claims rely on the cluster-benchmark-level
fallback (`population/ood_split.py`) rather than per-country microdata.

**What data does each instance consist of?**
Synthetic agent: see `agents/profile.py:AgentProfile`. Event record: see
`bgf_logging/event_logger.py` (one JSON object per line, fields
`round_id`, `agent_id`, `action`, `wealth_delta`, plus per-policy
metadata).

**Is there a label or target associated with each instance?**
No — the framework is purely descriptive / generative. Behavioural
predictions are derived post-hoc from action distributions via
`metrics/behavioral_realism.py`.

**Is any information missing from individual instances?**
Yes — see audit A1.2 in `docs/AUDIT_DATA_METRICS_LOGGING.md`. The ESS
parquet has 10–24 % missingness on some fields (`left_right` 10 %,
`income_decile` ~24 %, `age` ~6 %). Missing values are now replaced via
**marginal resampling** weighted by `anweight` (audit A1.2 patched
2026-05-25); pre-patch runs collapsed NaNs to the median bin / uniform
age and should be regenerated.

**Are relationships between individual instances made explicit?**
Yes — the agent network topology (small-world or Erdős–Rényi) is fixed
at simulation start and recorded in `experiments/<exp_id>/network.json`.
Per-round interaction edges can be reconstructed from `events.jsonl`
via `/replay/<exp_id>` or `scripts/plot_network_evolution.py`.

**Are there recommended splits?**
For OOD validation: leave-one-cluster-out at the cluster-benchmark level
via `population/ood_split.py`. Microdata-level country holdouts are not
possible because the local parquet is single-country (AT).

**Are there errors, sources of noise, or redundancies?**
Yes — see all 🔴 / 🟠 entries in `docs/AUDIT_DATA_METRICS_LOGGING.md`.
Notable: (i) two historical income formulas in
`population/generator.py` and `population/persona_synthesizer.py`
produced 5× divergent absolute income for the same ESS row (audit A1.3,
unified through `population/_helpers.income_from_decile()` 2026-05-25
but the generator path still uses the legacy formula to preserve
already-published numbers). (ii) Pre-2026-05-20 BRM values are
compressed to [0.307, 1.0] because of a JSD log-base bug; rerun
`metrics/distribution.py` consumers to refresh.

**Is the dataset self-contained?**
The synthetic outputs are self-contained inside each
`experiments/<exp_id>/` directory (config snapshot, events, summary,
witness manifest). The upstream ESS microdata must be obtained from
ESS ERIC under their data-access terms; BGF does not redistribute it.

**Does the dataset contain confidential information?**
The synthetic outputs do not contain re-identifiable information about
any real individual. Each agent profile is sampled from ESS
respondents whose responses are already de-identified by ESS ERIC.
BGF additionally drops the ESS respondent ID (`idno`) before any agent
is written to disk.

---

## Collection Process

**How was the data acquired?**
Synthetic outputs are emitted by `simulation/kernel.py` during a run.
Upstream ESS microdata is downloaded from ESS ERIC's data portal
(`ess-search.nsd.no`) under their terms of use and ingested by
`scripts/ingest_ess.py` into `data/ess_clean.parquet`.

**What sampling strategy?**
Empirical mode: weighted resampling with replacement from the local ESS
parquet, weighted by `anweight` (see `population/sampling.py:96`).
Synthetic mode: rule-based draws from configured prior distributions.

**Who was involved in the collection?**
The ESS microdata is collected by trained interviewers in 31+ ESS
member countries via face-to-face interviews following the ESS Round
11 fieldwork protocol. BGF's automated ingestion adds no human labour
to the data after ESS ERIC's release.

**Over what timeframe?**
ESS Round 11 fieldwork: 2023–2024. BGF simulation runs are stamped
with UTC timestamps in `events.jsonl` and `experiments/<exp_id>/metadata.json`.

**Were ethics-review processes conducted?**
The ESS microdata is collected under ESS ERIC's institutional ethics
framework. The pending §8.4 Prolific human-evaluation study requires
its own IRB approval before launch (see Section 4.0 of `docs/paper.md`).
No human-subjects data is collected or processed by the synthetic-
society simulation itself.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing done?**
Yes, see `scripts/ingest_ess.py`. ESS missing-codes (55/66/77/88/99/…)
are replaced with NaN; Likert scales are normalised to [0, 1]; raw CSV
is converted to Parquet. Audit A1.1 (added 2026-05-25) preserves the
ESS survey weights (`anweight`, `pspwght`, `pweight`, `dweight`) so
downstream sampling is correctly weighted.

**Is the raw data also available?**
The raw ESS CSV is **not redistributed** in this repo (it is large and
ESS ERIC's terms govern redistribution). Obtain it directly from ESS
ERIC. The cleaned parquet (`data/ess_clean.parquet`, AT-only) ships
gitignored — regenerate it locally with `python scripts/ingest_ess.py`.

---

## Uses

**Has the dataset been used for any tasks already?**
Yes — the analyses reported in `docs/paper.md` use the synthetic-
society outputs end-to-end. See Sections 5–8 of the paper.

**Is there a repository linking to other uses?**
Yes — the GitHub issue tracker at
`github.com/lucastourinho/SyntheticSocieties/issues` lists open
extension threads.

**What (other) tasks could the dataset be used for?**
Calibrating LLM-agent simulations against empirical survey
distributions; measuring the RLHF cooperative bias of new instruction-
tuned models; benchmarking alignment-tax effects across model
families; methodological research on attitude-behaviour gap in LLM
personas. **Not** suitable for policy forecasting or individual-level
prediction — see §9.A Broader Impacts in `docs/paper.md`.

**Are there tasks the dataset should not be used for?**
See §9.A "Out-of-scope uses" in `docs/paper.md`. In summary: policy
forecasting, individual-level inference, cultural-cluster ranking
outside the six validated ESS clusters, real-time decision systems.

---

## Distribution

**Will the dataset be distributed to third parties?**
Synthetic outputs are distributed under MIT licence via the GitHub
repository. The upstream ESS microdata is **not** redistributed by
BGF — third parties must obtain it directly from ESS ERIC.

**How will it be distributed?**
GitHub releases (tagged versions) + Zenodo DOI snapshot (planned,
pending v1.1 release).

**When?**
v1.0 is on the `main` branch as of 2026-05-25; v1.1 (with the audit
A1.1 / A1.2 / A1.3 patches and the §8.1.4 multi-seed figure) tagged
after the in-flight N=500 LLM sweep completes.

**Will the dataset be distributed under a copyright or other IP
licence?**
Synthetic outputs and BGF code: MIT (see `LICENSE`). ESS Round 11
microdata: ESS ERIC's Conditions of Use (terms imposed by data
providers).

---

## Maintenance

**Who will be supporting / maintaining the dataset?**
The repository owner (Lucas Tourinho, `tourinholucas123@gmail.com`).

**How can the owner be contacted?**
Via the GitHub issue tracker or the email in `CITATION.cff`.

**Is there an erratum?**
Yes — `docs/AUDIT_DATA_METRICS_LOGGING.md` lists every silently-
corrupted-numbers incident, the date each was fixed, and which
pre-fix runs need regeneration. Treat that document as the canonical
erratum log.

**Will the dataset be updated?**
Yes — the active branch is `main`. Tagged releases (`v1.0`, `v1.1`,
…) snapshot stable points. Each release has a corresponding Zenodo
DOI (pending; see §3.12 of `docs/paper.md`).

---

**Maintainer contact, licence, and citation:** see `CITATION.cff` and
the top-level `LICENSE`.
