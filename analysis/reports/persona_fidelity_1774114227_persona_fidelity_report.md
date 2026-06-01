# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774114227`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `65.904`
- Score bias: `11.306`
- Score MAE: `11.306`
- Dispersion ratio (synthetic / real): `0.470`
- Pearson correlation: `0.498`
- Spearman correlation: `0.595`

## Latent-Space Check

- PC1 Pearson correlation: `0.452`
- PC1 Spearman correlation: `0.476`
- PC1 bias: `-2.538`

## Recalibration

- Affine slope: `1.060`
- Affine intercept: `-15.293`
- MAE after affine recalibration: `4.599`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.