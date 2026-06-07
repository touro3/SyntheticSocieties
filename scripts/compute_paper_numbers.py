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


# ── Experiment loaders for current mx_A/mx_B summary.json structure ──────────


def _load_summary_metrics(exp_dir: Path) -> Optional[dict]:
    """Read key metrics from summary.json (mx_A_s*/mx_B_s* structure)."""
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            s = json.load(f)
        m = s.get("metrics", {})
        cr = m.get("cooperation_rate", {})
        gini = m.get("gini", {})
        brm = m.get("brm")
        pf = m.get("persona_fidelity", {})

        def _val(x):
            if isinstance(x, dict):
                return x.get("final", x.get("mean"))
            return x

        # terminal cooperation from round_metrics if available
        rm_path = exp_dir / "round_metrics.jsonl"
        terminal_coop = None
        n_rounds = 0
        if rm_path.exists():
            rounds = []
            with open(rm_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            rounds.append(json.loads(line))
                        except Exception:
                            pass
            if rounds:
                n_rounds = len(rounds)
                last = rounds[-1]
                ad = last.get("action_distribution", {})
                terminal_coop = ad.get("cooperate")

        return {
            "coop_mean": _val(cr),
            "coop_terminal": terminal_coop,
            "gini_final": _val(gini),
            "brm": brm,
            "pf_mean": _val(pf) if isinstance(pf, dict) else pf,
            "n_rounds": n_rounds,
            "n_agents": s.get("num_agents"),
        }
    except Exception:
        return None


def _pool_summary_seeds(seed_list: list[dict]) -> dict:
    """Pool per-seed summary metrics into mean ± std."""
    if not seed_list:
        return {}

    def _arr(key):
        return [s[key] for s in seed_list if s.get(key) is not None]

    def _st(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        arr = np.array(vals)
        return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4), "n": len(vals)}

    return {
        "coop_mean": _st(_arr("coop_mean")),
        "coop_terminal": _st(_arr("coop_terminal")),
        "gini_final": _st(_arr("gini_final")),
        "brm": _st(_arr("brm")),
        "n_seeds": len(seed_list),
    }


def _load_n100_extension(experiments_dir: Path) -> dict:
    """Load N=100 10-seed LLM extension (mx_A_s{1..10}, mx_B_s{1..10})."""
    result = {}
    for cond, prefix in [("condA", "mx_A_s"), ("condB", "mx_B_s")]:
        seeds = []
        for i in range(1, 11):
            d = experiments_dir / f"{prefix}{i}"
            m = _load_summary_metrics(d)
            if m:
                m["seed"] = i
                seeds.append(m)
        result[cond] = {
            "per_seed": seeds,
            "pooled": _pool_summary_seeds(seeds),
            "n_found": len(seeds),
        }
    return result


def _load_n500_cascade(experiments_dir: Path) -> dict:
    """Load N=500 cascade seeds (mx_A_n500_s*, mx_B_n500_s*)."""
    result = {}
    for cond, prefix in [("condA", "mx_A_n500_s"), ("condB", "mx_B_n500_s")]:
        seeds = []
        for i in range(1, 11):
            d = experiments_dir / f"{prefix}{i}"
            m = _load_summary_metrics(d)
            if m:
                m["seed"] = i
                seeds.append(m)
        result[cond] = {"per_seed": seeds, "pooled": _pool_summary_seeds(seeds), "n_found": len(seeds)}
    return result


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

    # ── N=100 10-seed extension (primary confirmatory data) ───────────────────
    n100_ext = _load_n100_extension(experiments_dir)
    a10 = n100_ext.get("condA", {}).get("pooled", {})
    b10 = n100_ext.get("condB", {}).get("pooled", {})

    # ── N=500 cascade seeds ────────────────────────────────────────────────────
    n500_cascade = _load_n500_cascade(experiments_dir)

    # ── Paper-vs-actual verification (current paper claims §8.1) ─────────────
    def _mean(d):
        return d.get("mean") if d else None

    paper_corrections = {
        "n100_coop_condA": {
            "paper_claims": "0.461±0.042 (MWU p=0.91)",
            "actual_mean": _mean(a10.get("coop_mean")),
            "actual_std": a10.get("coop_mean", {}).get("std"),
            "n_seeds": a10.get("n_seeds"),
        },
        "n100_coop_condB": {
            "paper_claims": "0.455±0.044",
            "actual_mean": _mean(b10.get("coop_mean")),
            "actual_std": b10.get("coop_mean", {}).get("std"),
            "n_seeds": b10.get("n_seeds"),
        },
        "n100_gini_condA": {
            "paper_claims": "0.718±0.032",
            "actual_mean": _mean(a10.get("gini_final")),
            "actual_std": a10.get("gini_final", {}).get("std"),
        },
        "n100_gini_condB": {
            "paper_claims": "0.715±0.032",
            "actual_mean": _mean(b10.get("gini_final")),
            "actual_std": b10.get("gini_final", {}).get("std"),
        },
        "n100_brm_condA": {
            "paper_claims": "0.832±0.022",
            "actual_mean": _mean(a10.get("brm")),
            "actual_std": a10.get("brm", {}).get("std"),
        },
        "n100_brm_condB": {
            "paper_claims": "0.848±0.017",
            "actual_mean": _mean(b10.get("brm")),
            "actual_std": b10.get("brm", {}).get("std"),
        },
    }

    paper_numbers = {
        "_generated_by": "scripts/compute_paper_numbers.py",
        "_note": "Primary source: mx_A_s{1..10}/mx_B_s{1..10} summary.json (N=100 10-seed extension).",
        "n100_llm_extension": n100_ext,
        "n500_cascade": n500_cascade,
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
        "paper_corrections": paper_corrections,
    }

    return paper_numbers


def _print_summary(paper_numbers: dict) -> None:
    print("\n" + "=" * 70)
    print("  BGF Paper Numbers — Authoritative Measurements (N=100 10-seed)")
    print("=" * 70)

    corrections = paper_numbers.get("paper_corrections", {})
    print(f"\n  {'Metric':<25} {'Paper claims':<28} {'Actual mean':>12} {'Actual std':>11} {'Match?':>7}")
    print("  " + "-" * 85)
    for key, val in corrections.items():
        claim = str(val.get("paper_claims", "?"))
        am = val.get("actual_mean")
        as_ = val.get("actual_std")
        actual_str = f"{am:.4f}" if am is not None else "None"
        std_str = f"±{as_:.4f}" if as_ is not None else ""
        # Very rough match check: within 0.01
        pc = val.get("paper_claims", "")
        match = "?"
        try:
            import re

            nums = re.findall(r"\d+\.\d+", pc)
            if nums and am is not None:
                match = "✓" if abs(am - float(nums[0])) < 0.015 else "✗"
        except Exception:
            pass
        print(f"  {key:<25} {claim:<28} {actual_str:>12} {std_str:>11} {match:>7}")

    # N=100 extension summary
    n100 = paper_numbers.get("n100_llm_extension", {})
    for cond in ["condA", "condB"]:
        d = n100.get(cond, {})
        n = d.get("n_found", 0)
        p = d.get("pooled", {})
        print(f"\n  {cond} N=100 ({n} seeds found):")
        for k in ["coop_terminal", "gini_final", "brm"]:
            v = p.get(k, {})
            if v.get("mean") is not None:
                print(f"    {k:<20}: {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})")

    # N=500 cascade
    n500 = paper_numbers.get("n500_cascade", {})
    print(
        f"\n  N=500 cascade seeds found: condA={n500.get('condA', {}).get('n_found', 0)}, condB={n500.get('condB', {}).get('n_found', 0)}"
    )
    for cond in ["condA", "condB"]:
        for s in n500.get(cond, {}).get("per_seed", []):
            coop = s.get("coop_terminal") or s.get("coop_mean")
            gini = s.get("gini_final")
            nr = s.get("n_rounds")
            if coop:
                print(f"    {cond} s{s['seed']} T={nr}: coop={coop:.3f}, Gini={gini:.4f}")

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
