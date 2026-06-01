#!/usr/bin/env bash
# =============================================================================
# run_phase_c_replication.sh — Re-run the primary LLM A/B comparison
#
# Reproduces experiments/phase_c_comparison/ (Condition A and B, N=50, T=30,
# seed=42), whose parquet files are missing from disk. The canonical summary
# statistics are preserved in analysis/paper_numbers.json.
#
# Paper §6.1 primary claims:
#   Condition A (ungrounded): coop=96.2%, Gini=0.625, B_RLHF=0.712
#   Condition B (grounded):   coop=58.2%, Gini=0.260, B_RLHF=0.420
#
# Requirements: GPU with ≥4 GB VRAM, Mistral-7B-Instruct-v0.3 cached
# Estimated runtime: ~3–5 hours on single P100 (4-bit quantization)
#
# Usage:
#   tmux new-session -d -s phase_c "bash scripts/run_phase_c_replication.sh 2>&1 | tee logs/phase_c_replication.log"
#   tmux attach -t phase_c
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "========================================================"
echo "  BGF Phase-C Replication — Primary LLM A/B Comparison"
echo "  N=50 agents | T=30 rounds | seed=42 | Mistral-7B"
echo "  $(date)"
echo "========================================================"

# ── Condition A — Ungrounded LLM (no ESS personas, no RAG) ──────────────────
echo ""
echo "── Condition A: Ungrounded LLM"
python scripts/run_full_pipeline.py \
    --include-llm \
    --agents 50 \
    --rounds 30 \
    --seeds 42 \
    --population.source synthetic \
    --policy.rag_enabled false \
    --policy.persona_grounding false \
    --exp-id phase_c_comparison_condition_a \
    2>&1 | tee logs/phase_c_condition_a.log

echo "── Condition A complete: $(date)"

# ── Condition B — ESS-Grounded LLM (empirical population + dual RAG) ────────
echo ""
echo "── Condition B: ESS-Grounded LLM"
python scripts/run_full_pipeline.py \
    --include-llm \
    --agents 50 \
    --rounds 30 \
    --seeds 42 \
    --population.source empirical \
    --policy.rag_enabled true \
    --exp-id phase_c_comparison_condition_b \
    2>&1 | tee logs/phase_c_condition_b.log

echo "── Condition B complete: $(date)"

# ── Verify results match paper_numbers.json ──────────────────────────────────
echo ""
echo "── Verifying against canonical values..."
python - <<'EOF'
import json
from pathlib import Path

canonical = json.loads(Path("analysis/paper_numbers.json").read_text())
ca = canonical["condition_a_ablated"]
cb = canonical["condition_b_grounded"]

tols = {"coop_rate_overall": 0.05, "gini_final": 0.05, "brlhf": 0.05}

for cond_label, cond_dir, canonical_cond in [
    ("A", "experiments/phase_c_comparison_condition_a", ca),
    ("B", "experiments/phase_c_comparison_condition_b", cb),
]:
    metrics_path = Path(cond_dir) / "metrics.json"
    if not metrics_path.exists():
        print(f"  WARNING: {metrics_path} not found — run may have failed")
        continue
    metrics = json.loads(metrics_path.read_text())
    print(f"\n  Condition {cond_label}:")
    for key, tol in tols.items():
        canon_val = canonical_cond.get(key, "N/A")
        new_val   = metrics.get(key, metrics.get(key.replace("_overall", ""), "N/A"))
        if isinstance(canon_val, float) and isinstance(new_val, float):
            diff = abs(new_val - canon_val)
            status = "✓" if diff <= tol else "⚠ DIFF"
            print(f"    {key}: canonical={canon_val:.3f}  new={new_val:.3f}  Δ={diff:.3f}  {status}")
        else:
            print(f"    {key}: canonical={canon_val}  new={new_val}")
EOF

echo ""
echo "========================================================"
echo "  Phase-C replication complete. Backup parquets:"
echo "    cp -r experiments/phase_c_comparison_condition_a experiments/phase_c_comparison/"
echo "    cp -r experiments/phase_c_comparison_condition_b experiments/phase_c_comparison/"
echo "========================================================"
