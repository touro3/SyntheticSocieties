# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774115559`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `65.983`
- Score bias: `11.385`
- Score MAE: `11.385`
- Dispersion ratio (synthetic / real): `0.619`
- Pearson correlation: `0.519`
- Spearman correlation: `0.524`

## Latent-Space Check

- PC1 Pearson correlation: `0.475`
- PC1 Spearman correlation: `0.381`
- PC1 bias: `-2.539`

## Recalibration

- Affine slope: `0.838`
- Affine intercept: `-0.709`
- MAE after affine recalibration: `4.509`
- Bias after affine recalibration: `0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.