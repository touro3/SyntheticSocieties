from __future__ import annotations

import numpy as np


def fit_affine(real: np.ndarray, synthetic: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)

    x_mean = synthetic.mean()
    y_mean = real.mean()
    x_var = ((synthetic - x_mean) ** 2).sum()

    if x_var == 0:
        slope = 1.0
    else:
        slope = float(((synthetic - x_mean) * (real - y_mean)).sum() / x_var)
    intercept = float(y_mean - slope * x_mean)

    return {"slope": slope, "intercept": intercept}


def apply_affine(values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return slope * values + intercept


def fit_variance_scaling(real: np.ndarray, synthetic: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)

    real_std = float(real.std(ddof=1)) if len(real) > 1 else 0.0
    synth_std = float(synthetic.std(ddof=1)) if len(synthetic) > 1 else 0.0
    scale = 1.0 if synth_std == 0 else real_std / synth_std
    real_mean = float(real.mean())
    synth_mean = float(synthetic.mean())

    return {
        "scale": scale,
        "real_mean": real_mean,
        "synthetic_mean": synth_mean,
    }


def apply_variance_scaling(values: np.ndarray, scale: float, synthetic_mean: float, real_mean: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - synthetic_mean
    return centered * scale + real_mean
