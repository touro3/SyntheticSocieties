# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774125935`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `56.479`
- Score bias: `1.881`
- Score MAE: `5.169`
- Dispersion ratio (synthetic / real): `0.418`
- Pearson correlation: `0.375`
- Spearman correlation: `0.383`

## Latent-Space Check

- PC1 Pearson correlation: `0.323`
- PC1 Spearman correlation: `0.190`
- PC1 bias: `-0.133`

## Recalibration

- Affine slope: `0.898`
- Affine intercept: `3.889`
- MAE after affine recalibration: `4.684`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.