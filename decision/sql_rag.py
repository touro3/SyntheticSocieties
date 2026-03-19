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

    def _connect(self):
        if self.conn is None:
            self.conn = duckdb.connect()
            self.conn.execute(f"CREATE VIEW population AS SELECT * FROM read_parquet('{self.data_path}')")
            self._initialized = True
        return True


    def query_population_trends(self, query: str) -> str:
        """Execute a SELECT query against the population database."""
        if not self._connect():
            return "Population database not available."
            
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
        """
        Grounded query: How do peers (age/gender/country) usually behave?
        """
        if not self._connect(): 
            return "Population database not available."
        
        # Robust gender mapping
        if isinstance(gender, int):
            g_val = gender
        else:
            g_val = self._GENDER_ENCODING.get(str(gender).lower(), 1)
        
        # Base query components
        where_clauses = ["age BETWEEN ? AND ?", "gender = ?"]
        params = [age - 5, age + 5, g_val]
        
        if country:
            where_clauses.append("country = ?")
            params.append(country)

        q = f"""
        SELECT 
            AVG(trust_people) * 10 AS avg_trust,
            AVG(political_interest) * 10 AS avg_interest,
            AVG(income_decile) AS avg_income_decile
        FROM population
        WHERE {' AND '.join(where_clauses)}
        """
        
        try:
            res = self.conn.execute(q, params).fetchdf()
            if res.empty or pd.isna(res.iloc[0]['avg_trust']):
                return "No peer group data found for this demographic."
            
            row = res.iloc[0]
            # Scaling note: ESS trust_people is normalized 0-1, so * 10 provides a 0-10 scale for prompts.
            return (f"Context: People in your age/gender bracket have an average "
                    f"trust level of {row['avg_trust']:.1f}/10 and income decile {row['avg_income_decile']:.1f}.")
        except Exception as e:
            return f"Data retrieval error: {str(e)}"

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


