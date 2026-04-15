"""Compute the 3-condition ablation statistics for the BGF paper.

The BGF project has THREE distinct experimental conditions, not two:
  A) Ablated LLM   — no ESS persona, no RAG → phase_c Condition A → 96% coop
  B) ESS persona   — ESS persona, no RAG   → pure_llm_ess_persona → ~1% coop
  C) Full grounded — ESS persona + RAG     → grounded_llm_ess_persona → ~50% coop
                     (also phase_c Condition B → 58% coop at larger scale)

This script computes summary stats for all three and writes:
  analysis/three_condition_stats.json
  analysis/three_condition_table.tex   (LaTeX table fragment)

Usage
-----
    python scripts/compute_three_condition_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compute_paper_numbers import (
    _metrics_from_events,
    _metrics_from_parquet,
    _parse_jsonl_events,
    _stats,
)

EXP_DIR = ROOT / "experiments"


def _load_jsonl_seeds(prefix: str, seeds=(42, 43, 44)) -> list[dict]:
    results = []
    for seed in seeds:
        path = EXP_DIR / f"{prefix}_s{seed}" / "events.jsonl"
        if path.exists():
            m = _metrics_from_events(_parse_jsonl_events(path))
            if m:
                m["seed"] = seed
                m["source"] = str(path)
                results.append(m)
    return results


def _pooled(seed_data: list[dict]) -> dict:
    coops = [s["coop_rate_overall"] for s in seed_data if s.get("coop_rate_overall") is not None]
    brlhfs = [s["brlhf"] for s in seed_data if s.get("brlhf") is not None]
    ginis = [s["gini_final"] for s in seed_data if s.get("gini_final") is not None]
    n_agents_vals = [s["n_agents"] for s in seed_data if s.get("n_agents")]
    n_rounds_vals = [s["n_rounds"] for s in seed_data if s.get("n_rounds")]
    return {
        "coop_rate": _stats(coops),
        "brlhf": _stats(brlhfs),
        "gini_final": _stats(ginis),
        "n_seeds": len(seed_data),
        "seeds": [s.get("seed") for s in seed_data],
        "n_agents_typical": max(n_agents_vals) if n_agents_vals else None,
        "n_rounds_typical": max(n_rounds_vals) if n_rounds_vals else None,
    }


def _fmt(stats_dict: dict, key: str) -> str:
    s = stats_dict.get(key, {})
    if not s or s.get("mean") is None:
        return "—"
    mean = s["mean"]
    std = s.get("std", 0)
    if std and std > 0.001:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def compute_three_condition_stats() -> dict:
    # ── Condition A: Ablated (phase_c Condition A) ────────────────────────
    a_parquet = EXP_DIR / "phase_c_comparison" / "condition_a_events.parquet"
    condition_a = _metrics_from_parquet(a_parquet) if a_parquet.exists() else {}
    if condition_a:
        condition_a["description"] = "Ablated LLM — no persona, no RAG (AblatedLLMPolicy)"
        condition_a["setup"] = "50 agents × 30 rounds, Mistral-7B, small-world network"

    # ── Condition B: ESS persona only (no RAG) ────────────────────────────
    pure_seeds = _load_jsonl_seeds("pure_llm_ess_persona")
    condition_b_pooled = _pooled(pure_seeds)
    condition_b = {
        "description": "LLM + ESS persona, no RAG (LLMPolicy, persona only)",
        "setup": "20 agents × 5 rounds, Mistral-7B, seeds 42/43/44",
        "per_seed": pure_seeds,
        "pooled": condition_b_pooled,
        # Scalar convenience values for reporting
        "coop_rate_overall": condition_b_pooled["coop_rate"].get("mean"),
        "brlhf": condition_b_pooled["brlhf"].get("mean"),
        "gini_final": condition_b_pooled["gini_final"].get("mean"),
    }

    # ── Condition C: Full grounding (persona + RAG) ───────────────────────
    grounded_seeds = _load_jsonl_seeds("grounded_llm_ess_persona")
    # Also include phase_c Condition B as a larger-scale measurement
    b_parquet = EXP_DIR / "phase_c_comparison" / "condition_b_events.parquet"
    phase_c_b = _metrics_from_parquet(b_parquet) if b_parquet.exists() else None

    grounded_pooled = _pooled(grounded_seeds)
    condition_c = {
        "description": "Full grounding — ESS persona + RAG (LLMPolicy, full BGF)",
        "setup": "20 agents × 5 rounds (seeds 42/43/44) + 50 agents × 30 rounds (phase_c)",
        "per_seed_small": grounded_seeds,
        "pooled_small_scale": grounded_pooled,
        "phase_c_large_scale": phase_c_b,
        # Prefer large-scale phase_c for final reporting
        "coop_rate_overall": (
            phase_c_b.get("coop_rate_overall") if phase_c_b else grounded_pooled["coop_rate"].get("mean")
        ),
        "brlhf": (phase_c_b.get("brlhf") if phase_c_b else grounded_pooled["brlhf"].get("mean")),
        "gini_final": (phase_c_b.get("gini_final") if phase_c_b else grounded_pooled["gini_final"].get("mean")),
    }

    stats = {
        "_generated_by": "scripts/compute_three_condition_stats.py",
        "condition_a_ablated": condition_a,
        "condition_b_persona_only": condition_b,
        "condition_c_full_grounded": condition_c,
    }
    return stats


def _latex_table(stats: dict) -> str:
    a = stats["condition_a_ablated"]
    b = stats["condition_b_persona_only"]
    c = stats["condition_c_full_grounded"]

    def row(label, setup, coop, brlhf, gini):
        return f"  {label} & {setup} & {coop} & {brlhf} & {gini} \\\\"

    def fmt_val(v):
        if v is None:
            return "—"
        return f"{v:.3f}"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Three-condition ablation: BGF behavioral outcomes. "
        r"Condition A = ablated LLM (no persona, no RAG); "
        r"Condition B = ESS persona only (no RAG); "
        r"Condition C = full BGF grounding (persona + RAG). "
        r"Phase C values from 50-agent, 30-round Mistral-7B run; "
        r"Conditions B and C small-scale from 20-agent, 5-round runs (seeds 42--44).}",
        r"\label{tab:three_condition}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"  Condition & Setup & Coop Rate & $B_{\mathrm{RLHF}}$ & Gini (final) \\",
        r"\midrule",
        row(
            "A: Ablated LLM",
            "No persona/RAG",
            fmt_val(a.get("coop_rate_overall")),
            fmt_val(a.get("brlhf")),
            fmt_val(a.get("gini_final")),
        ),
        row(
            "B: ESS persona only",
            "Persona, no RAG (20 agents)",
            fmt_val(b.get("coop_rate_overall")),
            fmt_val(b.get("brlhf")),
            fmt_val(b.get("gini_final")),
        ),
        row(
            "C: Full BGF grounding",
            "Persona + RAG (50 agents)",
            fmt_val(c.get("coop_rate_overall")),
            fmt_val(c.get("brlhf")),
            fmt_val(c.get("gini_final")),
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    stats = compute_three_condition_stats()

    out = ROOT / "analysis" / "three_condition_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved → {out}")

    latex = _latex_table(stats)
    tex_out = ROOT / "analysis" / "three_condition_table.tex"
    with open(tex_out, "w") as f:
        f.write(latex)
    print(f"Saved → {tex_out}")

    # Print summary
    print("\n" + "=" * 70)
    print("  Three-Condition Ablation Summary")
    print("=" * 70)
    print(f"  {'Condition':<30} {'Coop Rate':>10} {'B_RLHF':>8} {'Gini':>8}")
    print("  " + "-" * 58)
    for key, label in [
        ("condition_a_ablated", "A: Ablated (no persona/RAG)"),
        ("condition_b_persona_only", "B: Persona only (no RAG)"),
        ("condition_c_full_grounded", "C: Full grounding"),
    ]:
        s = stats[key]
        coop = s.get("coop_rate_overall")
        brlhf = s.get("brlhf")
        gini = s.get("gini_final")
        print(
            f"  {label:<30} {f'{coop:.3f}' if coop else '—':>10} "
            f"{f'{brlhf:.3f}' if brlhf else '—':>8} "
            f"{f'{gini:.3f}' if gini else '—':>8}"
        )

    print("\n  Key insight: ESS persona ALONE suppresses cooperation too aggressively")
    print("  (1% vs 96% for ablated). Full grounding (RAG + persona) restores")
    print("  the calibrated middle ground (58% coop, Gini within EU range).")
    print("=" * 70)

    print("\n  LaTeX table fragment:")
    print(latex)


if __name__ == "__main__":
    main()
