# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774116156`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `66.225`
- Score bias: `11.627`
- Score MAE: `11.627`
- Dispersion ratio (synthetic / real): `0.653`
- Pearson correlation: `0.491`
- Spearman correlation: `0.524`

## Latent-Space Check

- PC1 Pearson correlation: `0.474`
- PC1 Spearman correlation: `0.357`
- PC1 bias: `-2.662`

## Recalibration

- Affine slope: `0.752`
- Affine intercept: `4.774`
- MAE after affine recalibration: `4.193`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.