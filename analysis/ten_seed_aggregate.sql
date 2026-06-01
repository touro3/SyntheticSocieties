-- analysis/ten_seed_aggregate.sql
-- Per-seed, per-condition metric extraction for the pre-registered
-- 10-seed N=500 T=30 confirmatory A/B scale-up (scripts/launch_gpu_ab.sh).
--
-- Run IDs are emitted by scripts/run_experiment_matrix.py as
--   mx_A_s<seed>  (Condition A = ungrounded LLM)
--   mx_B_s<seed>  (Condition B = grounded LLM)
-- and registered in tracker/experiment_index.parquet by tracker/.
--
-- Metrics computed directly from registry columns (no run replay needed):
--   * cooperation_rate = num_cooperate / (num_work + num_save + num_cooperate)
--   * wealth_gini       (registry column, as-is)
--   * b_rlhf            = TV(pi, uniform) = 0.5 * Σ |p_a - 1/3|   over {work,save,cooperate}
--                         (matches paper §3.2 Proposition 1; identity p_coop = b_rlhf + 1/3
--                          holds only under the equal-split assumption and is NOT used here)
--
-- BRM_composite is intentionally NOT computed here: it requires the four
-- sub-component scores (wealth JSD, Gini gap, coop accuracy, temporal
-- stability) which are produced by metrics/ during run finalization, not
-- stored as registry columns. ten_seed_report.py joins BRM from each
-- run's metrics/summary artifact when present and degrades gracefully
-- otherwise (Theorem 2's certificate is the BRM robustness statement).
--
-- Usage:
--   duckdb -c ".read analysis/ten_seed_aggregate.sql"
--   -- or parameterized from Python (see ten_seed_report.py), which sets
--   -- the registry path and the N / T filter before .read.

-- Defaults; override by defining these before .read in a Python/CLI wrapper.
SET variable registry_path = 'tracker/experiment_index.parquet';
SET variable pop_size      = 500;
SET variable horizon       = 30;

WITH runs AS (
    SELECT
        experiment_id,
        seed,
        -- Condition is encoded in the matrix-runner experiment_id prefix.
        CASE
            WHEN experiment_id LIKE 'mx_A_s%' THEN 'A_ungrounded'
            WHEN experiment_id LIKE 'mx_B_s%' THEN 'B_grounded'
        END AS condition,
        population_size,
        rounds,
        wealth_gini,
        num_work,
        num_save,
        num_cooperate,
        (num_work + num_save + num_cooperate) AS num_actions
    FROM read_parquet(getvariable('registry_path'))
    WHERE policy_type = 'llm'
      AND population_size = getvariable('pop_size')
      AND rounds          = getvariable('horizon')
      AND (experiment_id LIKE 'mx_A_s%' OR experiment_id LIKE 'mx_B_s%')
),
per_run AS (
    SELECT
        condition,
        seed,
        experiment_id,
        wealth_gini,
        CASE WHEN num_actions > 0
             THEN num_cooperate * 1.0 / num_actions END AS cooperation_rate,
        -- B_RLHF = 0.5 * Σ_a |p_a - 1/3|
        CASE WHEN num_actions > 0 THEN
            0.5 * (
                abs(num_work      * 1.0 / num_actions - (1.0/3.0)) +
                abs(num_save      * 1.0 / num_actions - (1.0/3.0)) +
                abs(num_cooperate * 1.0 / num_actions - (1.0/3.0))
            )
        END AS b_rlhf
    FROM runs
    WHERE condition IS NOT NULL
)
-- Two result sets are materialized for the Python layer to consume.
SELECT * FROM per_run ORDER BY condition, seed;

-- Condition-level aggregates (point estimates; bootstrap CIs are added in
-- Python because DuckDB has no native bootstrap and we want BCa intervals).
CREATE OR REPLACE TEMP VIEW ten_seed_condition_summary AS
SELECT
    condition,
    count(*)                       AS n_seeds,
    avg(cooperation_rate)          AS coop_mean,
    stddev_samp(cooperation_rate)  AS coop_sd,
    avg(wealth_gini)               AS gini_mean,
    stddev_samp(wealth_gini)       AS gini_sd,
    avg(b_rlhf)                    AS b_rlhf_mean,
    stddev_samp(b_rlhf)            AS b_rlhf_sd
FROM (
    SELECT
        CASE WHEN experiment_id LIKE 'mx_A_s%' THEN 'A_ungrounded'
             WHEN experiment_id LIKE 'mx_B_s%' THEN 'B_grounded' END AS condition,
        wealth_gini,
        CASE WHEN (num_work+num_save+num_cooperate) > 0
             THEN num_cooperate*1.0/(num_work+num_save+num_cooperate) END AS cooperation_rate,
        CASE WHEN (num_work+num_save+num_cooperate) > 0 THEN
            0.5*( abs(num_work*1.0/(num_work+num_save+num_cooperate)-1.0/3.0)
                + abs(num_save*1.0/(num_work+num_save+num_cooperate)-1.0/3.0)
                + abs(num_cooperate*1.0/(num_work+num_save+num_cooperate)-1.0/3.0) ) END AS b_rlhf
    FROM read_parquet(getvariable('registry_path'))
    WHERE policy_type='llm'
      AND population_size = getvariable('pop_size')
      AND rounds          = getvariable('horizon')
      AND (experiment_id LIKE 'mx_A_s%' OR experiment_id LIKE 'mx_B_s%')
) t
WHERE condition IS NOT NULL
GROUP BY condition;
