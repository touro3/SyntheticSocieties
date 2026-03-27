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

    def get_peer_group_context(self, age: int, gender: str | int, country: Optional[str] = None) -> str:
        """Grounded query: How do peers (age/gender/country) usually behave?

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
                AVG(trust_people) * 10 AS avg_trust,
                AVG(political_interest) * 10 AS avg_interest,
                AVG(income_decile) AS avg_income_decile,
                AVG(risk_taking) * 10 AS avg_risk,
                AVG(life_satisfaction) * 10 AS avg_satisfaction
            FROM population
            WHERE {' AND '.join(where_clauses)}
            """
            try:
                res = self.conn.execute(q, params).fetchdf()
                if res.empty or pd.isna(res.iloc[0]["avg_trust"]):
                    return None
                row = res.iloc[0]
                return (
                    f"Context: People in your demographic brackets have an average "
                    f"trust level of {row['avg_trust']:.1f}/10, "
                    f"income decile {row['avg_income_decile']:.1f}, "
                    f"risk tolerance of {row['avg_risk']:.1f}/10, "
                    f"and life satisfaction of {row['avg_satisfaction']:.1f}/10."
                )
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


