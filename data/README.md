# BGF Data Directory

This directory contains data assets for the Behavioral Grounding Framework.

## Released files (included in repo)

| File | Description |
|------|-------------|
| `empirical_distributions.json` | Pre-computed marginal and joint ESS-11 distributions (no raw microdata). Safe to redistribute. |
| `cross_cultural_benchmarks.json` | Published ESS-11 trust means per country cluster (Nordic / Southern / Eastern). |
| `cross_cultural_benchmarks_expanded.json` | Extended benchmarks with cooperation-rate targets per cluster. |
| `ess_clean.parquet` | Cleaned ESS Round 11 microdata subset. **Not for redistribution** — see licence below. |
| `ess_schema.py` | Column definitions and variable codings for the ESS clean dataset. |
| `dataset_registry.json` | Metadata registry for all data assets used in experiments. |

---

## Obtaining the raw ESS Round 11 data

The European Social Survey (ESS) Round 11 (2023) microdata are available free of charge
for academic use but **may not be redistributed** directly.

### Step-by-step download

1. Register for a free account at the ESS Data Portal:
   <https://ess.sikt.no/>

2. Navigate to **ESS Round 11 → Download**.

3. Select the **Integrated file** (all countries) in **SPSS (.sav)** or **Stata (.dta)** format.

4. Accept the ESS Data licence (non-commercial academic use).

5. Download to `data/raw/ESS11.sav` (or `.dta`).

6. Run the automated ingestion script:

```bash
python scripts/download_ess_data.py --input data/raw/ESS11.sav --output data/ess_clean.parquet
```

This will:
- Recode all variables to the BGF schema (see `data/ess_schema.py`)
- Drop respondents with missing values on key variables (trust_people, income_decile, etc.)
- Normalise continuous variables to [0, 1]
- Write the cleaned Parquet to `data/ess_clean.parquet`

### Country codes used in BGF

| Cluster | Countries | ESS codes |
|---------|-----------|-----------|
| Nordic | Norway, Sweden, Denmark | NO, SE, DK |
| Southern | Italy, Spain, Portugal | IT, ES, PT |
| Eastern | Poland, Czech Republic, Hungary | PL, CZ, HU |

---

## Licences

- **ESS microdata**: Non-commercial academic use only. Full terms at
  <https://www.europeansocialsurvey.org/data/licence.html>
- **Pre-computed distributions** (`empirical_distributions.json`, `cross_cultural_benchmarks*.json`):
  Released under CC BY 4.0 alongside the BGF codebase.
