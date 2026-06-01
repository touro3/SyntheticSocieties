"""Compute authoritative paper numbers from all BGF experiment data.

Reads every experiment directory and extracts the ground-truth metrics that
correspond to the paper's claims. Outputs analysis/paper_numbers.json.

Usage
-----
    python scripts/compute_paper_numbers.py
    python scripts/compute_paper_numbers.py --experiments-dir experiments/ --out analysis/paper_numbers.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.inequality import gini_coefficient

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_action_type(action) -> str:
    """Extract action_type string from raw action field (str, dict, or JSON)."""
    if isinstance(action, dict):
        return action.get("action_type", "unknown")
    try:
        d = ast.literal_eval(str(action))
        if isinstance(d, dict):
            return d.get("action_type", "unknown")
    except Exception:
        pass
    s = str(action).lower()
    for a in ("cooperate", "work", "save", "steal"):
        if a in s:
            return a
    return "unknown"


def _parse_wealth(state_after) -> Optional[float]:
    if isinstance(state_after, dict):
        return float(state_after.get("wealth", 0.0))
    try:
        d = ast.literal_eval(str(state_after))
        return float(d.get("wealth", 0.0))
    except Exception:
        return None


def _brlhf(action_counts: dict, n_actions: int = 4) -> float:
    """B_RLHF = 0.5 * sum |pi(a) - 1/|A|| (total variation from uniform)."""
    all_actions = ["cooperate", "work", "save", "steal"]
    total = sum(action_counts.values()) or 1
    tv = sum(abs(action_counts.get(a, 0) / total - 1 / n_actions) for a in all_actions)
    return round(0.5 * tv, 6)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "n": len(values),
    }


# ── Parsers ───────────────────────────────────────────────────────────────────


def _parse_jsonl_events(jsonl_path: Path) -> list[dict]:
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _metrics_from_events(events: list[dict]) -> dict:
    """Compute metrics from a list of event dicts (jsonl format)."""
    if not events:
        return {}

    action_types = [_parse_action_type(e.get("action", "")) for e in events]
    n = len(action_types)
    action_counts = {}
    for a in action_types:
        action_counts[a] = action_counts.get(a, 0) + 1

    coop_rate = action_counts.get("cooperate", 0) / n if n else 0.0

    # Per-round cooperation rates
    rounds = {}
    for e in events:
        rid = e.get("round_id", 0)
        at = _parse_action_type(e.get("action", ""))
        if rid not in rounds:
            rounds[rid] = []
        rounds[rid].append(at)

    coop_per_round = [sum(1 for a in acts if a == "cooperate") / len(acts) for acts in rounds.values() if acts]

    # Per-round Gini (from state_after wealth)
    gini_per_round = []
    for rid in sorted(rounds.keys()):
        round_events = [e for e in events if e.get("round_id") == rid]
        wealths = [_parse_wealth(e.get("state_after")) for e in round_events]
        wealths = [w for w in wealths if w is not None]
        if len(wealths) > 1:
            gini_per_round.append(gini_coefficient(wealths))

    n_agents = len({e.get("agent_id") for e in events})
    n_rounds = len(rounds)

    return {
        "coop_rate_overall": round(coop_rate, 6),
        "coop_rate_per_round": _stats(coop_per_round),
        "gini_per_round": _stats(gini_per_round),
        "gini_final": round(gini_per_round[-1], 6) if gini_per_round else None,
        "brlhf": _brlhf(action_counts),
        "action_counts": action_counts,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "n_events": n,
    }


def _metrics_from_parquet(parquet_path: Path) -> dict:
    """Compute metrics from a parquet file (phase_c_comparison / bad_apple format)."""
    try:
        import pandas as pd
    except ImportError:
        print(f"  [skip] pandas not available for {parquet_path}")
        return {}

    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        return {}

    df = df.copy()
    df["at"] = df["action"].apply(_parse_action_type)

    n = len(df)
    action_counts = df["at"].value_counts().to_dict()
    coop_rate = (df["at"] == "cooperate").mean()

    # Per-round coop rates
    if "round_id" in df.columns:
        coop_per_round = df.groupby("round_id")["at"].apply(lambda x: (x == "cooperate").mean()).tolist()
    else:
        coop_per_round = [coop_rate]

    # Per-round Gini
    gini_per_round = []
    if "state_after" in df.columns and "round_id" in df.columns:
        for rid in sorted(df["round_id"].unique()):
            rdf = df[df["round_id"] == rid].copy()
            rdf["wealth"] = rdf["state_after"].apply(_parse_wealth)
            wvals = rdf["wealth"].dropna().tolist()
            if len(wvals) > 1:
                gini_per_round.append(gini_coefficient(wvals))

    n_agents = df["agent_id"].nunique() if "agent_id" in df.columns else None
    n_rounds = df["round_id"].nunique() if "round_id" in df.columns else None

    return {
        "coop_rate_overall": round(float(coop_rate), 6),
        "coop_rate_per_round": _stats(coop_per_round),
        "gini_per_round": _stats(gini_per_round),
        "gini_final": round(gini_per_round[-1], 6) if gini_per_round else None,
        "brlhf": _brlhf(action_counts),
        "action_counts": {k: int(v) for k, v in action_counts.items()},
        "n_agents": int(n_agents) if n_agents is not None else None,
        "n_rounds": int(n_rounds) if n_rounds is not None else None,
        "n_events": int(n),
    }


# ── Experiment loaders ────────────────────────────────────────────────────────


def _load_experiment(exp_dir: Path) -> Optional[dict]:
    """Load metrics from a single experiment directory."""
    result = {"path": str(exp_dir), "name": exp_dir.name}

    # Try parquet first
    parquets = list(exp_dir.glob("*.parquet"))
    if parquets:
        combined = {}
        for pq in parquets:
            m = _metrics_from_parquet(pq)
            if m:
                combined[pq.name] = m
        if combined:
            result["format"] = "parquet"
            result["files"] = combined
            return result

    # Try events.jsonl
    jsonl = exp_dir / "events.jsonl"
    if jsonl.exists():
        events = _parse_jsonl_events(jsonl)
        m = _metrics_from_events(events)
        if m:
            result["format"] = "jsonl"
            result["metrics"] = m
            # Also load summary.json if present
            summary_path = exp_dir / "summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        result["summary"] = json.load(f)
                except Exception:
                    pass
            return result

    return None


# ── Main aggregation ──────────────────────────────────────────────────────────


def compute_paper_numbers(experiments_dir: Path) -> dict:
    """Compute all authoritative paper numbers from experiments."""

    # ── Phase C comparison (primary LLM result) ─────────────────��──────────
    phase_c = experiments_dir / "phase_c_comparison"
    condition_a = condition_b = None
    if phase_c.exists():
        pqs = list(phase_c.glob("*.parquet"))
        for pq in pqs:
            m = _metrics_from_parquet(pq)
            if not m:
                continue
            if "condition_a" in pq.name:
                condition_a = m
                condition_a["source"] = str(pq)
            elif "condition_b" in pq.name:
                condition_b = m
                condition_b["source"] = str(pq)

    # ── pure_llm_ess_persona (seeds 42, 43, 44) ───────────────────────────
    pure_seeds = []
    for seed in [42, 43, 44]:
        d = experiments_dir / f"pure_llm_ess_persona_s{seed}"
        if (d / "events.jsonl").exists():
            m = _metrics_from_events(_parse_jsonl_events(d / "events.jsonl"))
            if m:
                m["seed"] = seed
                pure_seeds.append(m)

    # ── grounded_llm_ess_persona (seeds 42, 43, 44) ───────────────────────
    grounded_seeds = []
    for seed in [42, 43, 44]:
        d = experiments_dir / f"grounded_llm_ess_persona_s{seed}"
        if (d / "events.jsonl").exists():
            m = _metrics_from_events(_parse_jsonl_events(d / "events.jsonl"))
            if m:
                m["seed"] = seed
                grounded_seeds.append(m)

    def _pool_seeds(seed_list: list[dict]) -> dict:
        """Pool per-seed metrics into mean ± std."""
        if not seed_list:
            return {}
        coops = [s["coop_rate_overall"] for s in seed_list if s.get("coop_rate_overall") is not None]
        brlhfs = [s["brlhf"] for s in seed_list if s.get("brlhf") is not None]
        gini_finals = [s["gini_final"] for s in seed_list if s.get("gini_final") is not None]
        return {
            "coop_rate": _stats(coops),
            "brlhf": _stats(brlhfs),
            "gini_final": _stats(gini_finals),
            "n_seeds": len(seed_list),
            "seeds": [s.get("seed") for s in seed_list],
            "n_agents": seed_list[0].get("n_agents"),
            "n_rounds": seed_list[0].get("n_rounds"),
        }

    # ── Compute B_RLHF reduction ──────────────────────────────────────────
    brlhf_reduction = None
    if condition_a and condition_b:
        a_b = condition_a.get("brlhf")
        b_b = condition_b.get("brlhf")
        if a_b and b_b and a_b > 0:
            brlhf_reduction = round((a_b - b_b) / a_b * 100, 1)

    # ── Behavioral ground truth alignment ─────────────────────────────────
    benchmarks = {
        "pgg": {"low": 0.35, "high": 0.55, "source": "Ledyard 1995; Zelmer 2003"},
        "trust_game": {"low": 0.35, "high": 0.65, "source": "Berg et al. 1995"},
        "iterated_pd": {"low": 0.40, "high": 0.65, "source": "Axelrod 1984"},
        "gini_eu": {"low": 0.20, "high": 0.38, "source": "Eurostat 2023"},
    }
    bgt_results = {}
    if condition_b:
        coop_b = condition_b.get("coop_rate_overall")
        gini_b = condition_b.get("gini_final")
        for name, bench in benchmarks.items():
            if name == "gini_eu":
                val = gini_b
            else:
                val = coop_b
            if val is not None:
                within = bench["low"] <= val <= bench["high"]
                bgt_results[name] = {
                    "value": val,
                    "range": [bench["low"], bench["high"]],
                    "within_range": within,
                    "verdict": "within_range" if within else ("above_range" if val > bench["high"] else "below_range"),
                }

    paper_numbers = {
        "_generated_by": "scripts/compute_paper_numbers.py",
        "_note": "All values from actual LLM experiments. phase_c_comparison = 50 agents x 30 rounds.",
        "condition_a_ablated": condition_a,
        "condition_b_grounded": condition_b,
        "pure_llm_ess_persona": {
            "per_seed": pure_seeds,
            "pooled": _pool_seeds(pure_seeds),
        },
        "grounded_llm_ess_persona": {
            "per_seed": grounded_seeds,
            "pooled": _pool_seeds(grounded_seeds),
        },
        "brlhf_reduction_pct": brlhf_reduction,
        "behavioral_ground_truth": bgt_results,
        "paper_corrections": {
            "coop_rate_condition_a": {
                "paper_claims": "≈0.74",
                "actual": condition_a.get("coop_rate_overall") if condition_a else None,
            },
            "coop_rate_condition_b": {
                "paper_claims": "≈0.31–0.38",
                "actual": condition_b.get("coop_rate_overall") if condition_b else None,
            },
            "brlhf_condition_a": {
                "paper_claims": "≈0.52",
                "actual": condition_a.get("brlhf") if condition_a else None,
            },
            "brlhf_condition_b": {
                "paper_claims": "≈0.21",
                "actual": condition_b.get("brlhf") if condition_b else None,
            },
            "gini_condition_a": {
                "paper_claims": "≈0.08 (near-zero, egalitarian)",
                "actual": condition_a.get("gini_final") if condition_a else None,
                "note": "WRONG DIRECTION: actual Gini A escalates, not stays low",
            },
            "gini_condition_b": {
                "paper_claims": "≈0.28–0.34",
                "actual": condition_b.get("gini_final") if condition_b else None,
            },
            "brlhf_reduction_pct": {
                "paper_claims": "≈60%",
                "actual": f"{brlhf_reduction}%" if brlhf_reduction is not None else None,
            },
        },
    }

    return paper_numbers


def _print_summary(paper_numbers: dict) -> None:
    print("\n" + "=" * 70)
    print("  BGF Paper Numbers — Authoritative Measurements")
    print("=" * 70)

    corrections = paper_numbers.get("paper_corrections", {})
    print("\n  PAPER vs ACTUAL (key discrepancies):")
    print(f"  {'Metric':<35} {'Paper claims':>15} {'Actual':>12}")
    print("  " + "-" * 64)
    for key, val in corrections.items():
        claim = str(val.get("paper_claims", "?"))
        actual = str(val.get("actual", "?"))
        note = val.get("note", "")
        flag = "  ← WRONG DIRECTION" if note else ""
        print(f"  {key:<35} {claim:>15} {actual:>12}{flag}")

    print()
    a = paper_numbers.get("condition_a_ablated") or {}
    b = paper_numbers.get("condition_b_grounded") or {}
    if a and b:
        print(f"  Phase C experiment: {a.get('n_agents')} agents × {a.get('n_rounds')} rounds")
        print(
            f"  Cond A: coop={a.get('coop_rate_overall'):.3f}, B_RLHF={a.get('brlhf'):.3f}, "
            f"Gini_final={a.get('gini_final'):.3f}"
        )
        print(
            f"  Cond B: coop={b.get('coop_rate_overall'):.3f}, B_RLHF={b.get('brlhf'):.3f}, "
            f"Gini_final={b.get('gini_final'):.3f}"
        )

    bgt = paper_numbers.get("behavioral_ground_truth", {})
    if bgt:
        print("\n  Behavioral Ground Truth (Condition B):")
        for name, r in bgt.items():
            status = "✓" if r["within_range"] else "✗"
            print(
                f"    {name:<15} {r['verdict']:<14} (value={r['value']:.3f}, "
                f"range=[{r['range'][0]:.2f},{r['range'][1]:.2f}]) {status}"
            )

    print()
    reduction = paper_numbers.get("brlhf_reduction_pct")
    print(f"  B_RLHF reduction (A → B): {reduction}%  (paper claims ≈60%)")
    print("=" * 70)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute authoritative paper numbers from BGF experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments-dir",
        default=str(ROOT / "experiments"),
        help="Path to experiments directory",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "analysis" / "paper_numbers.json"),
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    exp_dir = Path(args.experiments_dir)
    print(f"Reading experiments from: {exp_dir}")

    paper_numbers = compute_paper_numbers(exp_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(paper_numbers, f, indent=2)
    print(f"\nSaved → {out}")

    _print_summary(paper_numbers)


if __name__ == "__main__":
    main()
