# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774122891`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `70.017`
- Score bias: `15.418`
- Score MAE: `15.418`
- Dispersion ratio (synthetic / real): `0.691`
- Pearson correlation: `0.623`
- Spearman correlation: `0.643`

## Latent-Space Check

- PC1 Pearson correlation: `0.613`
- PC1 Spearman correlation: `0.476`
- PC1 bias: `-3.589`

## Recalibration

- Affine slope: `0.902`
- Affine intercept: `-8.581`
- MAE after affine recalibration: `3.779`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.