"""
SQL-based TableRAG for population grounding.
Allows agents to query empirical population trends via DuckDB SQL.
"""

import atexit
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Cap on peer-context cache. Each entry is a small string; bound exists so
# long cross-model sweeps (~10⁴ distinct demographic tuples) don't grow
# unbounded inside long-lived policy objects.
_PEER_CACHE_MAX = 512


class SQLRAG:
    _GENDER_ENCODING = {"male": 1, "female": 2}  # ESS codebook section 3.2

    # ESS columns used in peer-group queries
    _PEER_COLS = {"age", "gender", "country", "income_decile", "trust_people", "risk_taking", "life_satisfaction"}

    def __init__(
        self,
        data_path: str | Path = "data/ess_clean.parquet",
        static_context: Optional[str] = None,
    ):
        self.conn = None
        self._initialized = False
        self._available_cols: set[str] = set()
        self._has_peer_cols: bool = False
        self.data_path = Path(data_path)
        self.static_context = static_context  # narrative from analysis sidecar
        # Don't raise on missing file — degrade gracefully to static_context.
        # The default ESS parquet is gitignored and absent on cloud deployments;
        # raising here kills LLM policy builds on Render/CI.
        # Per-agent demographic context is static across rounds — cache it once.
        # LRU-bounded so long sweeps don't leak.
        self._peer_cache: OrderedDict[tuple, str] = OrderedDict()
        atexit.register(self.close)
        # Fallback status — audit signal so callers / event logs can record
        # whether grounding actually fired or silently degraded to a static
        # narrative. Values: "ok", "no_data_file", "no_peer_cols",
        # "no_cohort_match", "query_error".
        self.last_status: str = "ok"
        self._warned_fallback: bool = False

    def _emit_fallback(self, reason: str) -> None:
        """Record fallback reason and emit a single warning per instance."""
        self.last_status = reason
        if not self._warned_fallback:
            logger.warning(
                "SQLRAG: degrading to static_context (reason=%s, data_path=%s). "
                "Condition B grounding is NOT active for this run.",
                reason,
                self.data_path,
            )
            self._warned_fallback = True

    def _connect(self) -> None:
        """Lazily open a DuckDB connection and register the population view.

        Idempotent — safe to call multiple times.
        Detects available columns so queries can adapt to arbitrary data files.
        When the data file is missing, marks the instance as having no peer
        columns so callers fall back to static_context instead of querying.
        """
        if self._initialized:
            return
        if not self.data_path.exists():
            # File absent — no SQL queries possible; callers use static_context.
            self._has_peer_cols = False
            self._initialized = True
            self._emit_fallback("no_data_file")
            return
        self.conn = duckdb.connect()
        # data_path is a server-controlled config value (default
        # data/ess_clean.parquet, optionally overridden via SQLRAG(data_path=...)).
        # We resolve to absolute and reject newline/null/quote so the literal
        # cannot break out — same guard as tracker/analytics.py:_connect.
        resolved = str(self.data_path.resolve())
        if any(c in resolved for c in ("\n", "\r", "\x00", "'")):
            raise ValueError(f"Illegal characters in SQLRAG data_path: {resolved!r}")
        sql = f"CREATE VIEW population AS SELECT * FROM read_parquet('{resolved}')"  # nosec B608
        self.conn.execute(sql)
        try:
            cols_df = self.conn.execute("DESCRIBE population").fetchdf()
            self._available_cols = set(cols_df["column_name"].tolist())
        except Exception:
            self._available_cols = set()
        self._has_peer_cols = bool(self._available_cols & self._PEER_COLS)
        self._initialized = True
        if not self._has_peer_cols:
            self._emit_fallback("no_peer_cols")

    def query_population_trends(self, query: str) -> str:
        """Execute a SELECT query against the population database."""
        try:
            self._connect()
        except Exception as e:
            return f"Population database not available: {e}"

        if self.conn is None:
            return self.static_context or "Population database not available (data file absent)."

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

    def get_peer_group_context(
        self,
        age: int,
        gender: str | int,
        country: Optional[str] = None,
        agent_trust: Optional[float] = None,
        agent_risk: Optional[float] = None,
        agent_satisfaction: Optional[float] = None,
        income_decile: Optional[float] = None,
    ) -> str:
        """Grounded query: How do peers (age/gender/country/income) usually behave?

        Results are cached per demographic tuple — agent attributes never change
        between rounds so the DuckDB queries only run once per unique agent profile.
        In a 100-agent × 100-round run this reduces ~40,000 queries to ~100.

        Adaptive: if the data file lacks ESS demographic columns, returns the
        static population narrative from the analysis sidecar (if available),
        or a generic fallback.  This ensures LLM agents always receive some
        population grounding even when a non-ESS file is uploaded.

        Returns distribution-level information (mean, std, median, sample size)
        rather than just averages, so the LLM can understand population variance
        and where the agent sits relative to their peer group.

        Four-tier fallback for cohort matching:
          1. age±5, gender, country, income band (±2.5 deciles)  — tightest
          2. age±10, gender, country                             — drops income
          3. age±15, country                                     — drops gender
          4. population-wide                                     — broadest

        This ensures high-income and low-income agents receive meaningfully
        different peer norms rather than collapsing to the same demographic bucket.
        """
        _cache_key = (age, gender, country, round(income_decile or 0, 1))
        if _cache_key in self._peer_cache:
            self._peer_cache.move_to_end(_cache_key)
            return self._peer_cache[_cache_key]

        try:
            self._connect()
        except Exception as e:
            return self.static_context or f"Population database not available: {e}"

        # If the data file has no ESS demographic columns, skip SQL queries and
        # return the narrative from the analysis sidecar as population context.
        if not self._has_peer_cols:
            if self.static_context:
                return self.static_context
            return "No peer demographic data available in the uploaded dataset."

        # Robust gender mapping
        if isinstance(gender, int):
            g_val = gender
        else:
            g_val = self._GENDER_ENCODING.get(str(gender).lower(), 1)

        def _run_query(age_window: int, include_country: bool, include_income: bool = False) -> Optional[str]:
            # Only add WHERE clauses for columns that actually exist in the data
            have_age = "age" in self._available_cols
            have_gender = "gender" in self._available_cols
            where_clauses = []
            params: list = []
            if have_age:
                where_clauses.append("age BETWEEN ? AND ?")
                params.extend([age - age_window, age + age_window])
            if have_gender:
                where_clauses.append("gender = ?")
                params.append(g_val)
            # Legacy path kept for callers that pass age_window without the new flags
            if include_country and country and "country" in self._available_cols:
                where_clauses.append("country = ?")
                params.append(country)
            if include_income and income_decile is not None and "income_decile" in self._available_cols:
                where_clauses.append("income_decile BETWEEN ? AND ?")
                params.extend([income_decile - 2.5, income_decile + 2.5])

            # Build SELECT only for columns that exist
            have_trust = "trust_people" in self._available_cols
            have_income = "income_decile" in self._available_cols
            have_risk = "risk_taking" in self._available_cols
            have_sat = "life_satisfaction" in self._available_cols

            if not (have_trust or have_risk or have_sat):
                return None  # nothing useful to report — caller will use static_context

            select_parts = ["COUNT(*) AS n_peers"]
            if have_trust:
                select_parts += [
                    "AVG(trust_people) * 10 AS avg_trust",
                    "STDDEV(trust_people) * 10 AS std_trust",
                    "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trust_people) * 10 AS median_trust",
                    "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY trust_people) * 10 AS trust_q25",
                    "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY trust_people) * 10 AS trust_q75",
                ]
            if have_income:
                select_parts.append("AVG(income_decile) AS avg_income_decile")
            if have_risk:
                select_parts += [
                    "AVG(risk_taking) * 10 AS avg_risk",
                    "STDDEV(risk_taking) * 10 AS std_risk",
                    "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY risk_taking) * 10 AS median_risk",
                ]
            if have_sat:
                select_parts += [
                    "AVG(life_satisfaction) * 10 AS avg_satisfaction",
                    "STDDEV(life_satisfaction) * 10 AS std_satisfaction",
                ]

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            q = f"SELECT {', '.join(select_parts)} FROM population {where_sql}"  # noqa: S608
            try:
                res = self.conn.execute(q, params).fetchdf()
                if res.empty:
                    return None
                row = res.iloc[0]
                n = int(row["n_peers"])
                if n < 5:
                    return None

                parts = [f"Based on {n} peers in your demographic bracket:"]
                if have_trust and not pd.isna(row.get("avg_trust")):
                    parts.append(
                        f"  Trust: avg {row['avg_trust']:.1f}/10 (std {row['std_trust']:.1f},"
                        f" median {row['median_trust']:.1f},"
                        f" IQR [{row['trust_q25']:.1f}-{row['trust_q75']:.1f}])."
                    )
                if have_risk and not pd.isna(row.get("avg_risk")):
                    parts.append(
                        f"  Risk tolerance: avg {row['avg_risk']:.1f}/10"
                        f" (std {row['std_risk']:.1f}, median {row['median_risk']:.1f})."
                    )
                if have_sat and not pd.isna(row.get("avg_satisfaction")):
                    parts.append(
                        f"  Life satisfaction: avg {row['avg_satisfaction']:.1f}/10"
                        f" (std {row['std_satisfaction']:.1f})."
                    )
                if have_income and not pd.isna(row.get("avg_income_decile")):
                    income_line = (
                        f"  Income decile: {income_decile:.1f} (peer avg {row['avg_income_decile']:.1f})."
                        if income_decile is not None
                        else f"  Income decile: avg {row['avg_income_decile']:.1f}."
                    )
                    parts.append(income_line)

                # Agent position relative to peers
                agent_position_parts = []
                if (
                    have_trust
                    and agent_trust is not None
                    and not pd.isna(row.get("std_trust", float("nan")))
                    and row["std_trust"] > 0
                ):
                    z_trust = (agent_trust * 10 - row["avg_trust"]) / row["std_trust"]
                    if z_trust > 0.5:
                        agent_position_parts.append(
                            f"your trust ({agent_trust * 10:.1f}) is above average for your peers"
                        )
                    elif z_trust < -0.5:
                        agent_position_parts.append(
                            f"your trust ({agent_trust * 10:.1f}) is below average for your peers"
                        )
                if (
                    have_risk
                    and agent_risk is not None
                    and not pd.isna(row.get("std_risk", float("nan")))
                    and row["std_risk"] > 0
                ):
                    z_risk = (agent_risk * 10 - row["avg_risk"]) / row["std_risk"]
                    if z_risk > 0.5:
                        agent_position_parts.append(f"your risk tolerance ({agent_risk * 10:.1f}) is above average")
                    elif z_risk < -0.5:
                        agent_position_parts.append(f"your risk tolerance ({agent_risk * 10:.1f}) is below average")
                if agent_position_parts:
                    parts.append(f"  Relative to peers: {'; '.join(agent_position_parts)}.")

                return "\n".join(parts)
            except Exception:
                return None

        # Four-tier fallback: tightest → broadest
        for window, use_country, use_income in [
            (5, True, True),  # tier 1: tight age, same country, same income band
            (10, True, False),  # tier 2: wider age, same country
            (15, True, False),  # tier 3: broad age, same country, no income filter
            (15, False, False),  # tier 4: population-wide
        ]:
            result = _run_query(window, use_country, use_income)
            if result:
                self._cache_put(_cache_key, result)
                return result

        _fallback = self.static_context or "No peer group data found for this demographic."
        self._cache_put(_cache_key, _fallback)
        self._emit_fallback("no_cohort_match")
        return _fallback

    def _cache_put(self, key: tuple, value: str) -> None:
        """Insert into the bounded LRU peer cache, evicting oldest if full."""
        self._peer_cache[key] = value
        self._peer_cache.move_to_end(key)
        while len(self._peer_cache) > _PEER_CACHE_MAX:
            self._peer_cache.popitem(last=False)

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
