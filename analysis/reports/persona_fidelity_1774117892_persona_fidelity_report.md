# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774117892`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `67.238`
- Score bias: `12.639`
- Score MAE: `12.639`
- Dispersion ratio (synthetic / real): `0.633`
- Pearson correlation: `0.433`
- Spearman correlation: `0.500`

## Latent-Space Check

- PC1 Pearson correlation: `0.394`
- PC1 Spearman correlation: `0.381`
- PC1 bias: `-2.889`

## Recalibration

- Affine slope: `0.684`
- Affine intercept: `8.617`
- MAE after affine recalibration: `4.744`
- Bias after affine recalibration: `-0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.