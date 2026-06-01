# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774117312`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `66.746`
- Score bias: `12.148`
- Score MAE: `12.148`
- Dispersion ratio (synthetic / real): `0.613`
- Pearson correlation: `0.592`
- Spearman correlation: `0.571`

## Latent-Space Check

- PC1 Pearson correlation: `0.566`
- PC1 Spearman correlation: `0.452`
- PC1 bias: `-2.786`

## Recalibration

- Affine slope: `0.965`
- Affine intercept: `-9.833`
- MAE after affine recalibration: `3.917`
- Bias after affine recalibration: `-0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.