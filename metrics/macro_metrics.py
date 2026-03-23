import polars as pl
import numpy as np
import json

class SocietyMacroMetrics:
    """Calculates macro-level emergent properties from simulation logs."""
    
    @staticmethod
    def calculate_gini(wealth_array: np.ndarray) -> float:
        """Calculates the Gini coefficient of a wealth distribution."""
        wealth_array = np.array(wealth_array, dtype=float)
        wealth_array = wealth_array[~np.isnan(wealth_array)]
        
        if len(wealth_array) == 0:
            return 0.0
            
        sorted_wealth = np.sort(wealth_array)
        n = len(wealth_array)
        index = np.arange(1, n + 1)
        
        sum_wealth = np.sum(sorted_wealth)
        if sum_wealth == 0:
            return 0.0
            
        return float((np.sum((2 * index - n  - 1) * sorted_wealth)) / (n * sum_wealth))

    @staticmethod
    def analyze_trajectory(events_parquet_path: str) -> pl.DataFrame:
        """Reads event logs and computes per-round macro metrics."""
        df = pl.read_parquet(events_parquet_path)
        
        # Extratores seguros em Python puro (lida com Dicionários ou Strings)
        def safe_extract_action(val):
            if isinstance(val, dict):
                return str(val.get("action_type", "")).lower()
            if isinstance(val, str):
                if "cooperate" in val.lower(): return "cooperate"
                if "work" in val.lower(): return "work"
                if "save" in val.lower(): return "save"
            return ""
            
        def safe_extract_wealth(val):
            if isinstance(val, dict):
                w = val.get("wealth")
                return float(w) if w is not None else None
            if isinstance(val, str):
                try:
                    return float(json.loads(val).get("wealth"))
                except:
                    pass
            return None

        # Aplica a extração segura
        df = df.with_columns(
            pl.col("action").map_elements(safe_extract_action, return_dtype=pl.String).alias("parsed_action"),
            pl.col("state_after").map_elements(safe_extract_wealth, return_dtype=pl.Float64).alias("parsed_wealth")
        )
        
        # Agrupa por rodada
        trends = df.group_by("round_id").agg([
            pl.col("parsed_wealth").drop_nulls().implode().alias("wealth_list"),
            pl.col("parsed_action").str.contains("cooperate").fill_null(False).sum().alias("coop_count"),
            pl.col("action").count().alias("total_actions")
        ]).sort("round_id")
        
        # Calcula Gini e Taxas
        trends = trends.with_columns(
            pl.col("wealth_list").map_elements(lambda x: SocietyMacroMetrics.calculate_gini(np.array(list(x))), return_dtype=pl.Float64).alias("gini_coefficient"),
            pl.col("wealth_list").map_elements(lambda x: float(np.sum(list(x))), return_dtype=pl.Float64).alias("total_wealth"),
            (pl.col("coop_count") / pl.col("total_actions")).alias("cooperation_rate")
        )
        
        return trends.drop("wealth_list")