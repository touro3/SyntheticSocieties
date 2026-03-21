# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774124567`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `61.792`
- Score bias: `7.193`
- Score MAE: `7.885`
- Dispersion ratio (synthetic / real): `0.310`
- Pearson correlation: `-0.188`
- Spearman correlation: `-0.036`

## Latent-Space Check

- PC1 Pearson correlation: `-0.354`
- PC1 Spearman correlation: `-0.214`
- PC1 bias: `-1.451`

## Recalibration

- Affine slope: `-0.607`
- Affine intercept: `92.080`
- MAE after affine recalibration: `4.874`
- Bias after affine recalibration: `-0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.