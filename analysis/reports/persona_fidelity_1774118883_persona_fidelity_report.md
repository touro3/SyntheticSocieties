# Persona Fidelity Report

- Run dir: `artifacts/persona_fidelity/persona_fidelity_1774118883`
- Profiles: `8`

## Core Metrics

- Mean real score: `54.598`
- Mean synthetic score: `66.392`
- Score bias: `11.793`
- Score MAE: `11.793`
- Dispersion ratio (synthetic / real): `0.578`
- Pearson correlation: `0.570`
- Spearman correlation: `0.548`

## Latent-Space Check

- PC1 Pearson correlation: `0.533`
- PC1 Spearman correlation: `0.429`
- PC1 bias: `-2.696`

## Recalibration

- Affine slope: `0.987`
- Affine intercept: `-10.944`
- MAE after affine recalibration: `4.198`
- Bias after affine recalibration: `-0.000`

## Interpretation

This benchmark should be read as a fidelity check on persona-conditioned response generation, not as a replacement for real respondents.
High correlation with low dispersion ratio indicates rank-order preservation with variance compression, a pattern explicitly emphasized in the reference thesis.
Affine recalibration is reported to measure whether the synthetic profile grid can be corrected for level bias without destroying ordering.