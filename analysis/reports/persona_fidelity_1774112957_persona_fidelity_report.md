# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774112957`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `59.938`
- Score bias: `5.339`
- Score MAE: `5.981`
- Dispersion ratio (synthetic / real): `0.453`
- Pearson correlation: `0.385`
- Spearman correlation: `0.310`

## Latent-Space Check

- PC1 Pearson correlation: `0.353`
- PC1 Spearman correlation: `0.071`
- PC1 bias: `-1.103`

## Recalibration

- Affine slope: `0.850`
- Affine intercept: `3.663`
- MAE after affine recalibration: `4.745`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.