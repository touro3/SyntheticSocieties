# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774119448`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `69.154`
- Score bias: `14.556`
- Score MAE: `14.556`
- Dispersion ratio (synthetic / real): `0.688`
- Pearson correlation: `0.573`
- Spearman correlation: `0.690`

## Latent-Space Check

- PC1 Pearson correlation: `0.550`
- PC1 Spearman correlation: `0.643`
- PC1 bias: `-3.383`

## Recalibration

- Affine slope: `0.833`
- Affine intercept: `-3.000`
- MAE after affine recalibration: `4.114`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.