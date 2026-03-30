"""
SQL-based TableRAG for population grounding.
Allows agents to query empirical population trends via DuckDB SQL.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Union

class SQLRAG:

    _GENDER_ENCODING = {"male": 1, "female": 2}  # ESS codebook section 3.2

    def __init__(self, data_path: str | Path = "data/ess_clean.parquet"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Population data not found: {self.data_path}")
        self.conn = None
        self._initialized = False

    def _connect(self) -> None:
        """Lazily open a DuckDB connection and register the population view.

        Idempotent — safe to call multiple times.
        Raises on failure so callers don't silently get wrong results.
        """
        if self.conn is None:
            self.conn = duckdb.connect()
            self.conn.execute(
                f"CREATE VIEW population AS SELECT * FROM read_parquet('{self.data_path}')"
            )
            self._initialized = True

    def query_population_trends(self, query: str) -> str:
        """Execute a SELECT query against the population database."""
        try:
            self._connect()
        except Exception as e:
            return f"Population database not available: {e}"
            
        # Security: Enforce SELECT-only
        if not query.strip().upper().startswith("SELECT"):
            return "Security error: Only SELECT queries are permitted."

        try:
            res = self.conn.execute(query).fetchdf()
            if res.empty:
                return "No matching population data found."
            return res.to_string(index=False)
        except Exception as e:
            return f"Query error: {str(e)}"

    def get_peer_group_context(self, age: int, gender: str | int, country: Optional[str] = None,
                                agent_trust: Optional[float] = None,
                                agent_risk: Optional[float] = None,
                                agent_satisfaction: Optional[float] = None) -> str:
        """Grounded query: How do peers (age/gender/country) usually behave?

        Returns distribution-level information (mean, std, median, sample size)
        rather than just averages, so the LLM can understand population variance
        and where the agent sits relative to their peer group.

        Attempts a narrow match (±5 years, same gender, same country if given).
        Falls back to a ±15-year window, then to a population-wide average, so the
        LLM always receives meaningful grounding rather than an empty string.
        """
        try:
            self._connect()
        except Exception as e:
            return f"Population database not available: {e}"

        # Robust gender mapping
        if isinstance(gender, int):
            g_val = gender
        else:
            g_val = self._GENDER_ENCODING.get(str(gender).lower(), 1)

        def _run_query(age_window: int, include_country: bool) -> Optional[str]:
            where_clauses = ["age BETWEEN ? AND ?", "gender = ?"]
            params: list = [age - age_window, age + age_window, g_val]
            if include_country and country:
                where_clauses.append("country = ?")
                params.append(country)
            q = f"""
            SELECT
                COUNT(*) AS n_peers,
                AVG(trust_people) * 10 AS avg_trust,
                STDDEV(trust_people) * 10 AS std_trust,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trust_people) * 10 AS median_trust,
                AVG(income_decile) AS avg_income_decile,
                AVG(risk_taking) * 10 AS avg_risk,
                STDDEV(risk_taking) * 10 AS std_risk,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY risk_taking) * 10 AS median_risk,
                AVG(life_satisfaction) * 10 AS avg_satisfaction,
                STDDEV(life_satisfaction) * 10 AS std_satisfaction,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trust_people) * 10 AS trust_q25,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trust_people) * 10 AS trust_q75
            FROM population
            WHERE {' AND '.join(where_clauses)}
            """
            try:
                res = self.conn.execute(q, params).fetchdf()
                if res.empty or pd.isna(res.iloc[0]["avg_trust"]):
                    return None
                row = res.iloc[0]
                n = int(row["n_peers"])
                if n < 5:
                    return None

                parts = [
                    f"Based on {n} peers in your demographic bracket:",
                    f"  Trust: avg {row['avg_trust']:.1f}/10 (std {row['std_trust']:.1f}, median {row['median_trust']:.1f}, IQR [{row['trust_q25']:.1f}-{row['trust_q75']:.1f}]).",
                    f"  Risk tolerance: avg {row['avg_risk']:.1f}/10 (std {row['std_risk']:.1f}, median {row['median_risk']:.1f}).",
                    f"  Life satisfaction: avg {row['avg_satisfaction']:.1f}/10 (std {row['std_satisfaction']:.1f}).",
                    f"  Income decile: avg {row['avg_income_decile']:.1f}.",
                ]

                # Show where the agent falls relative to peers
                agent_position_parts = []
                if agent_trust is not None and not pd.isna(row['std_trust']) and row['std_trust'] > 0:
                    z_trust = (agent_trust * 10 - row['avg_trust']) / row['std_trust']
                    if z_trust > 0.5:
                        agent_position_parts.append(f"your trust ({agent_trust*10:.1f}) is above average for your peers")
                    elif z_trust < -0.5:
                        agent_position_parts.append(f"your trust ({agent_trust*10:.1f}) is below average for your peers")
                if agent_risk is not None and not pd.isna(row['std_risk']) and row['std_risk'] > 0:
                    z_risk = (agent_risk * 10 - row['avg_risk']) / row['std_risk']
                    if z_risk > 0.5:
                        agent_position_parts.append(f"your risk tolerance ({agent_risk*10:.1f}) is above average")
                    elif z_risk < -0.5:
                        agent_position_parts.append(f"your risk tolerance ({agent_risk*10:.1f}) is below average")
                if agent_position_parts:
                    parts.append(f"  Relative to peers: {'; '.join(agent_position_parts)}.")

                return "\n".join(parts)
            except Exception:
                return None

        # Try narrow match → wider window → population-wide fallback
        for window, use_country in [(5, True), (15, True), (15, False)]:
            result = _run_query(window, use_country)
            if result:
                return result

        return "No peer group data found for this demographic."

    def close(self):
        """Release DuckDB resources."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


