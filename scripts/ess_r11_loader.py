"""ESS R11 CSV ingest → logistic regression → cooperation model metrics.

Generalises the Austrian baseline (data/cooperation_model.json) to any
country or multi-country cohort provided as a raw ESS R11 CSV.

Usage (CLI)
-----------
    python scripts/ess_r11_loader.py \
        data.ess_csv=data/ESS11MD_e01_2.csv \
        loader.countries=[AT,DE,FR] \
        loader.output_json=data/cooperation_model_DE_FR.json

Usage (Python API)
------------------
    from scripts.ess_r11_loader import ESSR11Loader
    loader = ESSR11Loader(cfg)
    result = loader.run()
    print(result["auc_cv_mean"])

Hydra config keys (under `loader`)
------------------------------------
    countries       list[str]   ISO-2 country codes to include ([] = all)
    target_col      str         ESS variable used as cooperation proxy
                                Default: "volunfr" (volunteered in last 12 months)
    features        list[str]   Feature columns for the regression
    n_bootstrap     int         Bootstrap iterations for CI estimation (default 500)
    cv_folds        int         Cross-validation folds (default 10)
    output_json     str | null  Path to write model params JSON
    min_n           int         Minimum rows required after filtering (default 100)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ESS R11 column name → BGF canonical name
# ---------------------------------------------------------------------------

_ESS_TO_BGF: dict[str, str] = {
    "ppltrst": "trust_people",
    "pplfair": "trust_fairness",
    "pplhlp": "trust_helpfulness",
    "rskfree": "risk_taking",
    "sclmeet": "social_meeting_freq",
    "sclact": "social_activity",
    "gincdif": "reduce_inequality",
    "hinctnta": "income_decile",
    "agea": "age",
    "gndr": "gender",
    "cntry": "country",
    "volunfr": "volunteered",
}

_DEFAULT_FEATURES = [
    "trust_people",
    "trust_fairness",
    "trust_helpfulness",
    "risk_taking",
    "social_meeting_freq",
    "social_activity",
    "reduce_inequality",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LoaderResult:
    country_filter: list[str]
    n_rows_raw: int
    n_rows_model: int
    target_col: str
    features: list[str]
    positive_rate: float
    auc_cv_mean: float
    auc_cv_std: float
    auc_bootstrap_mean: float
    auc_bootstrap_ci_lo: float
    auc_bootstrap_ci_hi: float
    coef_original: list[float]
    intercept_original: float
    coef_standardized: list[float]
    intercept_standardized: float
    feature_means: list[float]
    feature_stds: list[float]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class ESSR11Loader:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        loader_cfg = cfg.get("loader", {})
        self.ess_csv = Path(cfg.data.ess_csv)
        self.countries: list[str] = list(loader_cfg.get("countries", []))
        self.target_col: str = loader_cfg.get("target_col", "volunteered")
        self.features: list[str] = list(loader_cfg.get("features", _DEFAULT_FEATURES))
        self.n_bootstrap: int = int(loader_cfg.get("n_bootstrap", 500))
        self.cv_folds: int = int(loader_cfg.get("cv_folds", 10))
        self.output_json: Optional[Path] = Path(loader_cfg["output_json"]) if loader_cfg.get("output_json") else None
        self.min_n: int = int(loader_cfg.get("min_n", 100))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        df_raw = self._load_csv()
        df = self._clean(df_raw)
        result = self._fit(df)
        if self.output_json:
            self._write_json(result, df)
        return asdict(result)

    # ------------------------------------------------------------------
    # Load via DuckDB (handles large CSVs without reading fully into RAM)
    # ------------------------------------------------------------------

    def _load_csv(self) -> pd.DataFrame:
        if not self.ess_csv.exists():
            raise FileNotFoundError(f"ESS CSV not found: {self.ess_csv}")

        wanted_ess = list(_ESS_TO_BGF.keys())
        col_sql = ", ".join(f'"{c}"' for c in wanted_ess)

        if self.countries:
            country_list = ", ".join(f"'{c}'" for c in self.countries)
            where = f"WHERE cntry IN ({country_list})"
        else:
            where = ""

        query = f"SELECT {col_sql} FROM read_csv_auto('{self.ess_csv}', header=True, ignore_errors=True) {where}"

        try:
            df = duckdb.query(query).df()
        except Exception as exc:
            raise RuntimeError(f"DuckDB query failed: {exc}") from exc

        df = df.rename(columns=_ESS_TO_BGF)
        logger.info("Loaded %d rows from %s", len(df), self.ess_csv)
        return df

    # ------------------------------------------------------------------
    # Clean: normalise ESS 0–10 scales to [0, 1], encode target
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        warnings: list[str] = []

        scale_10 = [
            "trust_people",
            "trust_fairness",
            "trust_helpfulness",
            "risk_taking",
            "social_meeting_freq",
            "reduce_inequality",
        ]
        for col in scale_10:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # ESS uses 77/88/99 as missing codes
                df[col] = df[col].where(df[col] <= 10)
                df[col] = df[col] / 10.0

        # social_activity is 1–5 in ESS; normalise to [0,1]
        if "social_activity" in df.columns:
            df["social_activity"] = pd.to_numeric(df["social_activity"], errors="coerce")
            df["social_activity"] = df["social_activity"].where(df["social_activity"] <= 5)
            df["social_activity"] = (df["social_activity"] - 1) / 4.0

        # income_decile: 1–10
        if "income_decile" in df.columns:
            df["income_decile"] = pd.to_numeric(df["income_decile"], errors="coerce")
            df["income_decile"] = df["income_decile"].where(df["income_decile"] <= 10)

        # target: volunteered (1=yes, 2=no in ESS) → binary
        if self.target_col in df.columns:
            raw_target = pd.to_numeric(df[self.target_col], errors="coerce")
            # ESS volunfr: 1=yes, 2=no; treat 1 as cooperation proxy
            df["_target"] = (raw_target == 1).astype(float)
            df["_target"] = df["_target"].where(raw_target.notna())
        else:
            raise ValueError(
                f"Target column '{self.target_col}' not found after renaming. Available: {list(df.columns)}"
            )

        return df

    # ------------------------------------------------------------------
    # Fit logistic regression + bootstrap CI
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> LoaderResult:
        warnings: list[str] = []

        available = [f for f in self.features if f in df.columns]
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            warnings.append(f"Features not found in CSV, skipped: {missing}")

        model_cols = available + ["_target"]
        df_model = df[model_cols].dropna()

        if len(df_model) < self.min_n:
            raise ValueError(
                f"Only {len(df_model)} complete rows after filtering (min_n={self.min_n}). "
                "Try wider country filter or looser feature set."
            )

        X = df_model[available].values
        y = df_model["_target"].values

        pos_rate = float(y.mean())
        logger.info("%d model rows, positive rate=%.3f", len(df_model), pos_rate)

        # Standardise for coefficient comparability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(penalty="l2", max_iter=1000, random_state=42)

        # Cross-validated AUC
        cv = StratifiedKFold(n_splits=min(self.cv_folds, int(y.sum())), shuffle=True, random_state=42)
        cv_aucs = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
        auc_cv_mean = float(cv_aucs.mean())
        auc_cv_std = float(cv_aucs.std())

        # Fit on full dataset for coefficients
        clf.fit(X_scaled, y)

        # Bootstrap AUC CI
        rng = np.random.default_rng(42)
        boot_aucs: list[float] = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, len(X_scaled), size=len(X_scaled))
            if len(np.unique(y[idx])) < 2:
                continue
            clf_b = LogisticRegression(penalty="l2", max_iter=500, random_state=0)
            clf_b.fit(X_scaled[idx], y[idx])
            preds = clf_b.predict_proba(X_scaled)[:, 1]
            try:
                boot_aucs.append(roc_auc_score(y, preds))
            except Exception:
                pass

        boot_arr = np.array(boot_aucs)
        ci_lo, ci_hi = (
            (
                float(np.percentile(boot_arr, 2.5)),
                float(np.percentile(boot_arr, 97.5)),
            )
            if len(boot_arr) > 10
            else (float("nan"), float("nan"))
        )

        # Recover original-scale coefficients: coef_orig = coef_std / std
        coef_std = clf.coef_[0].tolist()
        intercept_std = float(clf.intercept_[0])
        means = scaler.mean_.tolist()
        stds = scaler.scale_.tolist()
        coef_orig = [c / s for c, s in zip(coef_std, stds)]
        intercept_orig = intercept_std - sum(c * m / s for c, m, s in zip(coef_std, means, stds))

        if warnings:
            logger.warning("Loader warnings: %s", "; ".join(warnings))

        return LoaderResult(
            country_filter=self.countries,
            n_rows_raw=len(df),
            n_rows_model=len(df_model),
            target_col=self.target_col,
            features=available,
            positive_rate=pos_rate,
            auc_cv_mean=auc_cv_mean,
            auc_cv_std=auc_cv_std,
            auc_bootstrap_mean=float(boot_arr.mean()) if len(boot_arr) else float("nan"),
            auc_bootstrap_ci_lo=ci_lo,
            auc_bootstrap_ci_hi=ci_hi,
            coef_original=coef_orig,
            intercept_original=intercept_orig,
            coef_standardized=coef_std,
            intercept_standardized=intercept_std,
            feature_means=means,
            feature_stds=stds,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Write model params JSON (same schema as data/cooperation_model.json)
    # ------------------------------------------------------------------

    def _write_json(self, result: LoaderResult, df: pd.DataFrame) -> None:
        out = {
            "meta": {
                "description": (
                    "Logistic regression cooperate/not model fitted on ESS R11 "
                    f"data for countries: {result.country_filter or 'all'}."
                ),
                "ess_source": str(self.ess_csv),
                "ess_country": result.country_filter,
                "ess_round": 11,
                "target_variable": result.target_col,
                "features": result.features,
                "model_type": "LogisticRegression(L2)",
                "n_bootstrap": self.n_bootstrap,
                "cv_folds": self.cv_folds,
                "known_limitations": [
                    f"Countries: {result.country_filter or 'all'}; cross-country generalization untested",
                    "Volunteering is a cooperation proxy, not a direct measure",
                    "ESS does not include experimental trust-game outcomes",
                ],
            },
            "model_params": {
                "feature_means": result.feature_means,
                "feature_stds": result.feature_stds,
                "coef_standardized": result.coef_standardized,
                "intercept_standardized": result.intercept_standardized,
                "coef_original": result.coef_original,
                "intercept_original": result.intercept_original,
            },
            "validation": {
                "auc_cv_mean": result.auc_cv_mean,
                "auc_cv_std": result.auc_cv_std,
                "auc_bootstrap_mean": result.auc_bootstrap_mean,
                "auc_bootstrap_95ci": [result.auc_bootstrap_ci_lo, result.auc_bootstrap_ci_hi],
                "n_rows_model": result.n_rows_model,
                "positive_rate": result.positive_rate,
            },
            "warnings": result.warnings,
        }

        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Model params written to %s", self.output_json)


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

try:
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base=None, config_path="../configs", config_name="base_config")
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO)
        loader = ESSR11Loader(cfg)
        result = loader.run()
        print(json.dumps({k: v for k, v in result.items() if k != "warnings"}, indent=2))

    if __name__ == "__main__":
        main()

except ImportError:
    if __name__ == "__main__":
        import sys

        print("Install hydra-core to use CLI mode. Import ESSR11Loader directly for API use.")
        sys.exit(1)
