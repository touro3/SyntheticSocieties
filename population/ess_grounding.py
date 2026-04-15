from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from population.society_spec import GroundingReport, SocietySpec

BAND_RANGES = {
    "very_low": (0.0, 0.2),
    "low": (0.2, 0.4),
    "moderate": (0.4, 0.6),
    "high": (0.6, 0.8),
    "very_high": (0.8, 1.01),
}


@dataclass
class GroundingResult:
    matched_df: pd.DataFrame
    report: GroundingReport
    population_context: str


class ESSGrounder:
    def __init__(self, ess_path: str | Path = "data/ess_clean.parquet", min_cohort_size: int = 30) -> None:
        self.ess_path = Path(ess_path)
        self.min_cohort_size = min_cohort_size

    def load(self) -> pd.DataFrame:
        if not self.ess_path.exists():
            raise FileNotFoundError(f"ESS parquet not found: {self.ess_path}")
        df = pd.read_parquet(self.ess_path).copy()

        if "trust_institutions" not in df.columns:
            trust_cols = [c for c in ["trust_parliament", "trust_legal", "trust_police"] if c in df.columns]
            if trust_cols:
                df["trust_institutions"] = df[trust_cols].mean(axis=1)

        return df

    def ground(self, spec: SocietySpec) -> GroundingResult:
        base_df = self.load()
        requested_filters = self._requested_filters(spec)
        trace: list[dict] = []

        search_plan = [
            {"name": "strict", "numeric_padding": 0.00, "urban_slack": 0, "age_slack": 0, "drop_filters": []},
            {"name": "widen_numeric", "numeric_padding": 0.10, "urban_slack": 1, "age_slack": 5, "drop_filters": []},
            {"name": "widen_more", "numeric_padding": 0.20, "urban_slack": 1, "age_slack": 10, "drop_filters": []},
        ]

        best_df = base_df.copy()
        best_filters = {}
        final_df = None
        final_filters = None

        for step in search_plan:
            df, active_filters = self._apply_filters(
                base_df=base_df,
                spec=spec,
                numeric_padding=step["numeric_padding"],
                urban_slack=step["urban_slack"],
                age_slack=step["age_slack"],
                drop_filters=set(step["drop_filters"]),
            )
            trace.append(
                {
                    "step": step["name"],
                    "matched_rows": int(len(df)),
                    "numeric_padding": step["numeric_padding"],
                    "urban_slack": step["urban_slack"],
                    "age_slack": step["age_slack"],
                    "drop_filters": list(step["drop_filters"]),
                }
            )
            if len(df) > len(best_df):
                best_df = df
                best_filters = active_filters
            if len(df) >= self.min_cohort_size:
                final_df = df
                final_filters = active_filters
                break

        if final_df is None:
            drop_order = [
                "competitiveness_band",
                "risk_tolerance_band",
                "religiosity_band",
                "political_orientation_band",
                "social_activity_band",
                "trust_institutions_band",
                "trust_people_band",
                "urbanization",
                "age_profile",
                "countries",
            ]
            dropped: list[str] = []
            for filter_name in drop_order:
                dropped.append(filter_name)
                df, active_filters = self._apply_filters(
                    base_df=base_df,
                    spec=spec,
                    numeric_padding=0.15,
                    urban_slack=1,
                    age_slack=10,
                    drop_filters=set(dropped),
                )
                trace.append(
                    {
                        "step": f"drop_{filter_name}",
                        "matched_rows": int(len(df)),
                        "numeric_padding": 0.15,
                        "urban_slack": 1,
                        "age_slack": 10,
                        "drop_filters": list(dropped),
                    }
                )
                if len(df) > len(best_df):
                    best_df = df
                    best_filters = active_filters
                if len(df) >= self.min_cohort_size:
                    final_df = df
                    final_filters = active_filters
                    break

        warnings: list[str] = []
        if final_df is None:
            final_df = best_df if not best_df.empty else base_df
            final_filters = best_filters
            warnings.append(
                f"Unable to reach min_cohort_size={self.min_cohort_size}; using best available cohort with {len(final_df)} rows."
            )

        variable_summaries = self._build_variable_summaries(final_df)
        evidence_snippets = self._build_evidence_snippets(variable_summaries, final_df, spec, trace)
        population_context = self._build_population_context(variable_summaries, final_df, spec, trace)

        report = GroundingReport(
            dataset_name="ESS11",
            selected_datasets=["ESS11"],
            matched_rows=int(len(final_df)),
            base_rows=int(len(base_df)),
            min_cohort_size=self.min_cohort_size,
            requested_filters=requested_filters,
            active_filters=final_filters or {},
            variable_summaries=variable_summaries,
            evidence_snippets=evidence_snippets,
            relaxation_trace=trace,
            warnings=warnings,
        )
        return GroundingResult(matched_df=final_df.copy(), report=report, population_context=population_context)

    def _requested_filters(self, spec: SocietySpec) -> dict:
        out = {}
        for name in [
            "countries",
            "age_profile",
            "urbanization",
            "trust_people_band",
            "trust_institutions_band",
            "social_activity_band",
            "religiosity_band",
            "political_orientation_band",
            "risk_tolerance_band",
            "competitiveness_band",
        ]:
            value = getattr(spec, name)
            if value not in (None, [], ""):
                out[name] = value
        return out

    def _apply_filters(
        self,
        base_df: pd.DataFrame,
        spec: SocietySpec,
        numeric_padding: float,
        urban_slack: int,
        age_slack: int,
        drop_filters: set[str],
    ) -> tuple[pd.DataFrame, dict]:
        df = base_df.copy()
        active_filters: dict[str, str] = {}

        if spec.countries and "countries" not in drop_filters and "country" in df.columns:
            df = df[df["country"].isin(spec.countries)]
            active_filters["countries"] = ",".join(spec.countries)

        if spec.age_profile and "age_profile" not in drop_filters and "age" in df.columns:
            if spec.age_profile == "young":
                max_age = 35 + age_slack
                df = df[df["age"] <= max_age]
                active_filters["age_profile"] = f"young(age<={max_age})"
            elif spec.age_profile == "aging":
                min_age = max(18, 50 - age_slack)
                df = df[df["age"] >= min_age]
                active_filters["age_profile"] = f"aging(age>={min_age})"
            elif spec.age_profile == "elderly":
                min_age = max(18, 60 - age_slack)
                df = df[df["age"] >= min_age]
                active_filters["age_profile"] = f"elderly(age>={min_age})"

        if spec.urbanization and "urbanization" not in drop_filters and "urbanization" in df.columns:
            base_map = {
                "urban": [1, 2],
                "suburban": [2, 3],
                "rural": [4, 5],
                "mixed": [1, 2, 3, 4, 5],
            }
            slack_map = {
                "urban": [1, 2, 3],
                "suburban": [1, 2, 3, 4],
                "rural": [3, 4, 5],
                "mixed": [1, 2, 3, 4, 5],
            }
            values = slack_map[spec.urbanization] if urban_slack > 0 else base_map[spec.urbanization]
            df = df[df["urbanization"].isin(values)]
            active_filters["urbanization"] = f"{spec.urbanization}:{values}"

        df, desc = self._apply_band(
            df, "trust_people", spec.trust_people_band, numeric_padding, drop_filters, "trust_people_band"
        )
        if desc:
            active_filters["trust_people_band"] = desc

        df, desc = self._apply_band(
            df,
            "trust_institutions",
            spec.trust_institutions_band,
            numeric_padding,
            drop_filters,
            "trust_institutions_band",
        )
        if desc:
            active_filters["trust_institutions_band"] = desc

        df, desc = self._apply_band(
            df, "social_meeting_freq", spec.social_activity_band, numeric_padding, drop_filters, "social_activity_band"
        )
        if desc:
            active_filters["social_activity_band"] = desc

        df, desc = self._apply_band(
            df, "risk_taking", spec.risk_tolerance_band, numeric_padding, drop_filters, "risk_tolerance_band"
        )
        if desc:
            active_filters["risk_tolerance_band"] = desc

        df, desc = self._apply_band(
            df, "competitiveness", spec.competitiveness_band, numeric_padding, drop_filters, "competitiveness_band"
        )
        if desc:
            active_filters["competitiveness_band"] = desc

        if spec.religiosity_band and "religiosity_band" not in drop_filters and "religious_belonging" in df.columns:
            if spec.religiosity_band in {"high", "very_high"}:
                df = df[df["religious_belonging"] == 1]
            elif spec.religiosity_band in {"low", "very_low"}:
                df = df[df["religious_belonging"] == 2]
            active_filters["religiosity_band"] = spec.religiosity_band

        if (
            spec.political_orientation_band
            and "political_orientation_band" not in drop_filters
            and "left_right" in df.columns
        ):
            padding = numeric_padding
            if spec.political_orientation_band == "left":
                df = df[df["left_right"] < min(1.0, 0.30 + padding)]
            elif spec.political_orientation_band == "center_left":
                df = df[(df["left_right"] >= max(0.0, 0.30 - padding)) & (df["left_right"] < min(1.0, 0.45 + padding))]
            elif spec.political_orientation_band == "center":
                df = df[(df["left_right"] >= max(0.0, 0.45 - padding)) & (df["left_right"] < min(1.0, 0.55 + padding))]
            elif spec.political_orientation_band == "center_right":
                df = df[(df["left_right"] >= max(0.0, 0.55 - padding)) & (df["left_right"] < min(1.0, 0.70 + padding))]
            elif spec.political_orientation_band == "right":
                df = df[df["left_right"] >= max(0.0, 0.70 - padding)]
            active_filters["political_orientation_band"] = (
                f"{spec.political_orientation_band}(padding={numeric_padding:.2f})"
            )

        return df.dropna(subset=["age"]).copy(), active_filters

    def _apply_band(
        self,
        df: pd.DataFrame,
        column: str,
        band: Optional[str],
        numeric_padding: float,
        drop_filters: set[str],
        filter_key: str,
    ) -> tuple[pd.DataFrame, Optional[str]]:
        if not band or filter_key in drop_filters or column not in df.columns:
            return df, None

        low, high = BAND_RANGES[band]
        low = max(0.0, low - numeric_padding)
        high = min(1.01, high + numeric_padding)
        filtered = df[(df[column] >= low) & (df[column] < high)]
        if not filtered.empty:
            return filtered, f"{band}[{low:.2f},{high:.2f})"
        return df, None

    def _safe_mean(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns:
            return None
        series = df[column].dropna()
        if series.empty:
            return None
        return float(series.mean())

    def _safe_mode(self, df: pd.DataFrame, column: str) -> Optional[str]:
        if column not in df.columns:
            return None
        series = df[column].dropna()
        if series.empty:
            return None
        return str(series.mode().iloc[0])

    def _build_variable_summaries(self, df: pd.DataFrame) -> dict:
        return {
            "age_mean": self._safe_mean(df, "age"),
            "trust_people_mean": self._safe_mean(df, "trust_people"),
            "trust_institutions_mean": self._safe_mean(df, "trust_institutions"),
            "social_activity_mean": self._safe_mean(df, "social_meeting_freq"),
            "risk_tolerance_mean": self._safe_mean(df, "risk_taking"),
            "competitiveness_mean": self._safe_mean(df, "competitiveness"),
            "life_satisfaction_mean": self._safe_mean(df, "life_satisfaction"),
            "happiness_mean": self._safe_mean(df, "happiness"),
            "income_decile_mean": self._safe_mean(df, "income_decile"),
            "country_mode": self._safe_mode(df, "country"),
        }

    def _relaxation_summary(self, trace: list[dict]) -> str:
        used = [step for step in trace if step["matched_rows"] >= self.min_cohort_size]
        if used:
            chosen = used[0]
        else:
            chosen = trace[-1] if trace else None

        if not chosen:
            return "No relaxation details available."

        dropped = chosen.get("drop_filters") or []
        if dropped:
            return f"Final cohort obtained after relaxing filters: {', '.join(dropped)}."
        if chosen["step"] == "strict":
            return "Strict cohort matching was sufficient."
        return f"Final cohort obtained with widened filter bands ({chosen['step']})."

    def _build_evidence_snippets(
        self, summary: dict, df: pd.DataFrame, spec: SocietySpec, trace: list[dict]
    ) -> list[str]:
        snippets = [f"ESS cohort size: {len(df)} respondents."]
        snippets.append(self._relaxation_summary(trace))
        if summary["age_mean"] is not None:
            snippets.append(f"Average age is {summary['age_mean']:.1f}.")
        if summary["trust_people_mean"] is not None:
            snippets.append(f"Average interpersonal trust is {summary['trust_people_mean'] * 10:.1f}/10.")
        if summary["trust_institutions_mean"] is not None:
            snippets.append(f"Average institutional trust is {summary['trust_institutions_mean'] * 10:.1f}/10.")
        if summary["social_activity_mean"] is not None:
            snippets.append(f"Average social activity is {summary['social_activity_mean'] * 10:.1f}/10.")
        if summary["risk_tolerance_mean"] is not None:
            snippets.append(f"Average risk tolerance is {summary['risk_tolerance_mean'] * 10:.1f}/10.")
        if summary["income_decile_mean"] is not None:
            snippets.append(f"Average income decile is {summary['income_decile_mean']:.1f}.")
        if spec.countries:
            snippets.append(f"Country scope: {', '.join(spec.countries)}.")
        return snippets

    def _build_population_context(self, summary: dict, df: pd.DataFrame, spec: SocietySpec, trace: list[dict]) -> str:
        parts = [
            f"Target cohort derived from ESS11 ({len(df)} matched respondents).",
            f"Original society request: {spec.narrative}",
            self._relaxation_summary(trace),
        ]
        if summary["age_mean"] is not None:
            parts.append(f"Mean age is {summary['age_mean']:.1f}.")
        if summary["trust_people_mean"] is not None:
            parts.append(f"Mean interpersonal trust is {summary['trust_people_mean'] * 10:.1f}/10.")
        if summary["trust_institutions_mean"] is not None:
            parts.append(f"Mean institutional trust is {summary['trust_institutions_mean'] * 10:.1f}/10.")
        if summary["social_activity_mean"] is not None:
            parts.append(f"Mean social activity is {summary['social_activity_mean'] * 10:.1f}/10.")
        if summary["risk_tolerance_mean"] is not None:
            parts.append(f"Mean risk tolerance is {summary['risk_tolerance_mean'] * 10:.1f}/10.")
        if summary["competitiveness_mean"] is not None:
            parts.append(f"Mean competitiveness is {summary['competitiveness_mean'] * 10:.1f}/10.")
        if summary["life_satisfaction_mean"] is not None:
            parts.append(f"Mean life satisfaction is {summary['life_satisfaction_mean'] * 10:.1f}/10.")
        if summary["income_decile_mean"] is not None:
            parts.append(f"Mean income decile is {summary['income_decile_mean']:.1f}.")
        if summary["country_mode"] is not None:
            parts.append(f"Most frequent country code in the cohort is {summary['country_mode']}.")
        parts.append("Use this as empirical grounding, not as a rigid stereotype.")
        return " ".join(parts)
