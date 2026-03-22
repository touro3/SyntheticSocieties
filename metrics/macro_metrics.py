import polars as pl
import numpy as np

class SocietyMacroMetrics:
    """Calculates macro-level emergent properties from simulation logs."""
    
    @staticmethod
    def calculate_gini(wealth_array: np.ndarray) -> float:
        """Calculates the Gini coefficient of a wealth distribution."""
        if len(wealth_array) == 0:
            return 0.0
        sorted_wealth = np.sort(wealth_array)
        n = len(wealth_array)
        index = np.arange(1, n + 1)
        
        sum_wealth = np.sum(sorted_wealth)
        if sum_wealth == 0:
            return 0.0
            
        return (np.sum((2 * index - n  - 1) * sorted_wealth)) / (n * sum_wealth)

    @staticmethod
    def analyze_trajectory(events_parquet_path: str) -> pl.DataFrame:
        """
        Reads event logs and computes per-round macro metrics:
        Gini coefficient, total wealth, and cooperation rate.
        """
        df = pl.read_parquet(events_parquet_path)
        
        # Extração blindada da ação
        if isinstance(df.schema["action"], pl.Struct):
            action_col = pl.col("action").struct.field("action_type").cast(pl.String)
        else:
            action_col = pl.col("action").cast(pl.String)
        
        # Agrupamento
        trends = df.group_by("round_id").agg([
            pl.col("state_after").struct.field("wealth").implode().alias("wealth_list"),
            action_col.str.to_lowercase().str.contains("cooperate").sum().alias("coop_count"),
            pl.col("action").count().alias("total_actions")
        ]).sort("round_id")
        
        trends = trends.with_columns(
            pl.col("wealth_list").map_elements(lambda x: SocietyMacroMetrics.calculate_gini(np.array(list(x))), return_dtype=pl.Float64).alias("gini_coefficient"),
            pl.col("wealth_list").map_elements(lambda x: float(np.sum(list(x))), return_dtype=pl.Float64).alias("total_wealth"),
            (pl.col("coop_count") / pl.col("total_actions")).alias("cooperation_rate")
        )
        
        
        return trends.drop("wealth_list")