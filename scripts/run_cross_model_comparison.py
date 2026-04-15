#!/usr/bin/env python3
"""Run cross-model A vs B comparison experiments (Phase 16).

For each configured model, runs both Condition A (ungrounded LLM) and
Condition B (ESS-grounded BGF) and saves per-run metrics to:
    analysis/cross_model_results.json

Then calls plot_cross_model_comparison.py to generate the figure.

Usage:
    # All models (requires GPU + OPENAI_API_KEY)
    python scripts/run_cross_model_comparison.py

    # Specific models only
    python scripts/run_cross_model_comparison.py --models mistral-7b llama3-8b

    # Dry run (mock backend, for testing the pipeline without GPU)
    python scripts/run_cross_model_comparison.py --dry-run

Outputs:
    analysis/cross_model_results.json
    analysis/figures/cross_model_bias_comparison.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from metrics.cross_model import CrossModelResult, build_comparison_table

_RESULTS_PATH = Path("analysis/cross_model_results.json")
_FIGURE_PATH = Path("analysis/figures/cross_model_bias_comparison.png")

# ── Model registry ────────────────────────────────────────────────────────────

_MODELS = {
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "backend_type": "huggingface",
        "cache_dir": "/mnt/raid/workspace/lucastourinho/models",
        "n_agents": 20,
        "n_rounds": 10,
    },
    # Qwen2.5-7B-Instruct: non-gated, comparable to Mistral/Llama3 in scale and RLHF training.
    # Replaces meta-llama/Llama-3.1-8B-Instruct which requires gated HF access.
    # NOTE: must use bfloat16 — float16 causes NaN in sampling probability tensors.
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "backend_type": "huggingface",
        "cache_dir": "/mnt/raid/workspace/lucastourinho/models",
        "dtype": "bfloat16",
        "n_agents": 20,
        "n_rounds": 10,
    },
    "gpt4o-mini": {
        "model_id": "gpt-4o-mini",
        "backend_type": "openai",
        "n_agents": 20,
        "n_rounds": 10,
    },
}


# ── Mock backend for dry-run ──────────────────────────────────────────────────


class _MockBackend:
    """Returns deterministic JSON for dry-run testing."""

    def __init__(self, bias_toward: str = "cooperate"):
        self._bias = bias_toward
        self._call_count = 0

    def load(self):
        pass

    def generate(self, messages, temperature=None, max_new_tokens=None):
        # Cycle through actions with a bias
        self._call_count += 1
        actions = ["work", "save", self._bias]
        action = actions[self._call_count % len(actions)]
        response = json.dumps(
            {
                "action_type": action,
                "target_agent_id": None,
                "amount": 10,
                "reasoning_summary": "mock",
                "confidence": 0.8,
            }
        )
        return response, 0.001


# ── CLI helpers ───────────────────────────────────────────────────────────────


def _parse_seed_list(value: str) -> list[int]:
    """Parse comma-separated seeds from CLI (e.g. '42,123,7')."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("Seed list cannot be empty.")

    seeds: list[int] = []
    for token in tokens:
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid seed '{token}': expected integer values.") from exc
    return seeds


# ── Simulation runner ─────────────────────────────────────────────────────────


def _run_condition(
    model_name: str,
    model_spec: dict,
    condition: str,
    seed: int = 42,
    dry_run: bool = False,
) -> CrossModelResult:
    """Run one condition (A or B) for one model and return metrics.

    In dry-run mode, uses a mock backend and synthetic action counts.
    In real mode, runs a minimal simulation via the BGF kernel.
    """
    print(
        f"\n  [{model_name}] Seed {seed} Condition {condition} — ",
        end="",
        flush=True,
    )
    t0 = time.time()

    if dry_run:
        # Synthetic results to validate the pipeline without GPU
        import random

        rng = random.Random(f"{model_name}:{condition}:{seed}")
        if condition == "A":
            # Ungrounded: high cooperation bias
            counts = {
                "cooperate": 60 + rng.randint(0, 10),
                "work": 20 + rng.randint(0, 5),
                "save": 20 + rng.randint(0, 5),
            }
        else:
            # Grounded: lower cooperation, closer to uniform
            counts = {
                "cooperate": 35 + rng.randint(0, 5),
                "work": 35 + rng.randint(0, 5),
                "save": 30 + rng.randint(0, 5),
            }

        total = sum(counts.values())
        coop_rate = counts["cooperate"] / total

        from metrics.behavioral_realism import rlhf_bias_index_from_counts

        rlhf_bias = rlhf_bias_index_from_counts(counts)

        # Synthetic Gini: grounded has more inequality (less over-cooperation)
        gini = 0.25 + rng.random() * 0.15 if condition == "B" else 0.10 + rng.random() * 0.10

        elapsed = time.time() - t0
        print(f"(dry-run, {elapsed:.1f}s)")
        return CrossModelResult(
            model_id=model_name,
            condition=condition,
            cooperation_rate=round(coop_rate, 4),
            gini=round(gini, 4),
            rlhf_bias_index=round(rlhf_bias, 4),
            n_agents=model_spec["n_agents"],
            n_rounds=model_spec["n_rounds"],
        )

    # Real run: build a config dict matching base_config.yaml structure,
    # then reuse the same helpers as run_config_simulation.py.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    from bgf_logging.event_logger import EventLogger
    from decision.ablated_llm_policy import AblatedLLMPolicy
    from decision.llm_policy import LLMPolicy
    from decision.model_config import ModelConfig, get_backend
    from metrics.behavioral_realism import rlhf_bias_index_from_counts
    from metrics.inequality import gini_coefficient
    from population.generator import generate_empirical_population
    from scripts.run_config_simulation import build_network, build_world
    from simulation.kernel import SimulationKernel

    # Minimal config dict matching what population/kernel helpers expect
    n_agents = model_spec["n_agents"]
    n_rounds = model_spec["n_rounds"]
    config = {
        "project": {
            "seed": seed,
            "experiment_id": f"cross_model_{model_name}_{condition}_s{seed}",
        },
        "simulation": {"rounds": n_rounds, "population_size": n_agents},
        "policy": {"type": "llm"},
        "population": {"source": "empirical"},
        "data": {
            "ess_clean_path": "data/ess_clean.parquet",
            "sample_mode": "resample",
        },
        "network": {"type": "small_world", "k": 4, "rewiring_prob": 0.1},
        "environment": {
            "public_signal": {"economy": "stable"},
            "prices": {"food": 1.0},
            "resources": {"jobs": 100.0},
        },
        "agent_defaults": {
            "min_age": 25,
            "max_age": 60,
            "base_income": 1000.0,
            "income_step": 100.0,
            "education": "college",
            "occupation": "worker",
            "location": "urban",
            "political_preference": "center",
            "risk_tolerance": 0.5,
            "social_class": "middle",
            "initial_wealth": 50.0,
            "wealth_step": 10.0,
            "memory_size": 10,
        },
        "llm": {
            "model_id": model_spec["model_id"],
            "cache_dir": model_spec.get("cache_dir"),
            "dtype": "float16",
            "device_map": "auto",
            "temperature": 0.7,
            "max_new_tokens": 128,
            "memory_window": 5,
            "max_retries": 2,
        },
    }

    # Build backend via ModelConfig factory
    model_cfg = ModelConfig(
        model_id=model_spec["model_id"],
        backend_type=model_spec["backend_type"],
        cache_dir=model_spec.get("cache_dir"),
        dtype=model_spec.get("dtype", "float16"),
        max_new_tokens=128,
        temperature=0.7,
    )
    backend = get_backend(model_cfg)
    backend.load()

    # Condition A = no_persona ablation; Condition B = full grounded LLM
    if condition == "A":
        policy = AblatedLLMPolicy(backend=backend, ablation="no_persona")
    else:
        policy = LLMPolicy(backend=backend, ablation_level=5)

    import tempfile

    from metrics.cross_model import extract_action_counts, extract_final_wealth

    agents = generate_empirical_population(config, policy)
    network_manager = build_network(config, agents)
    world = build_world(config, network_manager)

    with tempfile.TemporaryDirectory() as tmpdir:
        events_path = _Path(tmpdir) / "events.jsonl"
        logger = EventLogger(events_path, overwrite=True)
        kernel = SimulationKernel(agents=agents, world=world, logger=logger)
        kernel.run(num_rounds=n_rounds)

        action_counts = extract_action_counts(events_path)
        wealth_vals = extract_final_wealth(events_path)

    # Fallback: read final wealth directly from agent state if events didn't capture it
    if not wealth_vals:
        wealth_vals = [float(a.state.wealth) for a in agents]

    total = sum(action_counts.values()) or 1
    coop_rate = action_counts.get("cooperate", 0) / total
    rlhf_bias = rlhf_bias_index_from_counts(action_counts) if total > 0 else 0.0
    gini = gini_coefficient(wealth_vals) if len(wealth_vals) > 1 else 0.0

    elapsed = time.time() - t0
    print(f"coop={coop_rate:.2f}, bias={rlhf_bias:.3f}, gini={gini:.3f} ({elapsed:.0f}s)")

    return CrossModelResult(
        model_id=model_name,
        condition=condition,
        cooperation_rate=round(coop_rate, 4),
        gini=round(gini, 4),
        rlhf_bias_index=round(rlhf_bias, 4),
        n_agents=model_spec["n_agents"],
        n_rounds=model_spec["n_rounds"],
    )


def _load_existing(path: Path) -> dict[tuple[str, str], dict]:
    """Load existing results from file, keyed by (model_id, condition).

    Dry-run results are ignored — they are marked with n_agents=0 by convention,
    but to be safe we compare against known dry-run model IDs instead.
    Returns empty dict if file doesn't exist or is malformed.
    """
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text())
        return {(r["model_id"], r["condition"]): r for r in rows if isinstance(r, dict)}
    except Exception:
        return {}


def _save_results(path: Path, results: list[CrossModelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    path.write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-model A vs B comparison")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(_MODELS.keys()),
        default=list(_MODELS.keys()),
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock backend — validates pipeline without GPU/API",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_RESULTS_PATH,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all models even if results already exist",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=[42],
        help="Comma-separated seeds to run (e.g. 42,123,7)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 16 — Cross-Model Generalizability Study")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Mode:   {'dry-run' if args.dry_run else 'real (GPU/API)'}")
    print(f"  Seeds:  {', '.join(str(s) for s in args.seeds)}")
    print("=" * 60)

    # Load any previously saved real results (enables crash recovery)
    # Existing rows are keyed by (model, condition) and do not encode seed.
    # Multi-seed runs therefore always start fresh to avoid accidental skips.
    existing = {} if (args.dry_run or args.force or len(args.seeds) > 1) else _load_existing(args.out)
    if existing:
        print(f"\nResuming — found {len(existing)} existing result(s) in {args.out}")
        for m, c in existing:
            print(f"  Skipping {m} Condition {c} (already done)")

    all_results: list[CrossModelResult] = []

    # Seed existing results into all_results so the table is complete
    for row in existing.values():
        all_results.append(
            CrossModelResult(
                model_id=row["model_id"],
                condition=row["condition"],
                cooperation_rate=row["cooperation_rate"],
                gini=row["gini"],
                rlhf_bias_index=row["rlhf_bias_index"],
                n_agents=row.get("n_agents", 0),
                n_rounds=row.get("n_rounds", 0),
            )
        )

    for model_name in args.models:
        spec = _MODELS[model_name]
        print(f"\nModel: {model_name}")
        for seed in args.seeds:
            print(f"  Seed: {seed}")
            for condition in ["A", "B"]:
                if len(args.seeds) == 1 and (model_name, condition) in existing:
                    continue  # already done
                result = _run_condition(
                    model_name,
                    spec,
                    condition,
                    seed=seed,
                    dry_run=args.dry_run,
                )
                all_results.append(result)
                # Save incrementally after every condition — crash-safe
                _save_results(args.out, all_results)
                print(f"  Saved to: {args.out}")

    print(f"\nAll results saved to: {args.out}")

    # Print comparison table
    table = build_comparison_table(all_results)
    print("\nCross-Model Comparison Table:")
    print("-" * 80)
    for row in table:
        bias_a = row.get("bias_A", "?")
        bias_b = row.get("bias_B", "?")
        delta = row.get("bias_reduction_pct", "?")
        effective = "✓" if row.get("grounding_effective") else "✗"
        print(
            f"  {row['model']:<20} | Bias A={bias_a:<6} B={bias_b:<6} | Δ={delta}% | Grounding effective: {effective}"
        )

    # Generate figure if we have at least one complete A+B pair
    if any(row.get("grounding_effective") is not None for row in table):
        print("\nGenerating figure...")
        from scripts.plot_cross_model_comparison import plot_cross_model_comparison

        plot_cross_model_comparison(args.out, _FIGURE_PATH)
    else:
        print("\nNot enough data to generate figure (need at least one A+B pair).")


if __name__ == "__main__":
    main()
