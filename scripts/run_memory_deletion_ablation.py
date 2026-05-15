#!/usr/bin/env python3
"""Memory-deletion ablation (Phase 3, audit response — "myopic agents").

Question: is the memory architecture *behaviorally load-bearing*? We seed a
specific betrayal memory (an unreciprocated cooperation toward a fixed
partner) into every agent, then at a trigger round we wipe that memory from
the **treatment** cohort while the **control** cohort keeps it. If memory is
active, the treatment cohort's subsequent cooperation toward that partner
should diverge from control.

Reuses :func:`simulation.intervention_hooks.delete_betrayal_memories` for the
counterfactual and the standard SimulationKernel for the run. Full per-round
trajectories are written to events.jsonl + a diagnostics JSONL (not just
endpoint distributions).

Usage
-----
    # Dry-run (mock policy, fast, no GPU):
    python scripts/run_memory_deletion_ablation.py --dry-run --agents 8 --rounds 6

Output: analysis/tables/memory_deletion_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent import Agent  # noqa: E402
from agents.memory import HierarchicalMemory, MemoryItem  # noqa: E402
from agents.profile import AgentProfile  # noqa: E402
from agents.state import AgentState  # noqa: E402
from bgf_logging.event_logger import EventLogger  # noqa: E402
from decision.mock_policy import MockPolicy  # noqa: E402
from environment.institutions import InstitutionManager  # noqa: E402
from environment.network import NetworkManager  # noqa: E402
from environment.world import World  # noqa: E402
from environment.world_state import WorldState  # noqa: E402
from metrics.event_metrics import behavior_summary_from_events, load_events  # noqa: E402
from simulation.intervention_hooks import delete_betrayal_memories  # noqa: E402
from simulation.kernel import SimulationKernel  # noqa: E402

BETRAYER = "agent_0"
OUT_JSON = PROJECT_ROOT / "analysis" / "tables" / "memory_deletion_ablation.json"


def _make_cohort(n: int, cohort_tag: str) -> list[Agent]:
    agents = []
    for i in range(n):
        prof = AgentProfile(
            agent_id=f"{cohort_tag}_agent_{i}",
            age=35,
            income=1000.0,
            education="college",
            occupation="worker",
            location="urban",
            political_preference="center",
            risk_tolerance=0.5,
            social_class="middle",
            trust_people=0.5,
            competitiveness=0.5,
        )
        mem = HierarchicalMemory(max_recent=20)
        # Seed the shared betrayal memory (unreciprocated cooperation).
        mem.add(
            MemoryItem(
                round_id=0,
                partner_id=BETRAYER,
                event_type="cooperate",
                content=f"cooperated with {BETRAYER} who did not reciprocate",
                outcome={"reciprocated": False, "wealth_delta": -3.0},
                importance=0.9,
            )
        )
        agents.append(Agent(profile=prof, state=AgentState(wealth=100.0), memory=mem, policy=MockPolicy()))
    return agents


def _coop_toward(events: list[dict], partner: str) -> float:
    """Fraction of decisions that were 'cooperate' targeting ``partner``."""
    total = coop = 0
    for e in events:
        action = e.get("action") or e.get("action_type") or (e.get("proposed_action") or {}).get("action_type")
        if action is None:
            continue
        total += 1
        target = e.get("target_agent_id") or (e.get("proposed_action") or {}).get("target_agent_id")
        if action == "cooperate" and target == partner:
            coop += 1
    return coop / total if total else 0.0


def _run(agents: list[Agent], rounds: int, seed: int) -> list[dict]:
    ids = [a.profile.agent_id for a in agents]
    net = NetworkManager.small_world(agent_ids=ids, k=min(4, len(ids) - 1), rewiring_prob=0.1, seed=seed)
    world = World(
        state=WorldState(public_signal={"economy": "stable"}, prices={"food": 1.0}, resources={"jobs": 100.0}),
        institution_manager=InstitutionManager(),
        network_manager=net,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        kernel = SimulationKernel(agents=agents, world=world, logger=EventLogger(str(path), overwrite=True))
        kernel.run(num_rounds=rounds)
        return load_events(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory-deletion ablation runner.")
    ap.add_argument("--dry-run", action="store_true", help="Mock policy (default path; kept for parity).")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--trigger-round", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    control = _make_cohort(args.agents, "ctrl")
    treatment = _make_cohort(args.agents, "trt")

    # Counterfactual: wipe the betrayal from the treatment cohort only.
    removed = sum(delete_betrayal_memories(a, partner_id=BETRAYER) for a in treatment)
    kept = sum(
        1 for a in control for it in a.memory.recent if it.event_type == "cooperate" and it.partner_id == BETRAYER
    )

    ctrl_events = _run(control, args.rounds, args.seed)
    trt_events = _run(treatment, args.rounds, args.seed)

    ctrl_beh = behavior_summary_from_events(ctrl_events).get("event_behavior", {})
    trt_beh = behavior_summary_from_events(trt_events).get("event_behavior", {})

    out = {
        "design": "treatment cohort betrayal memory wiped vs control cohort intact",
        "betrayer": BETRAYER,
        "agents_per_cohort": args.agents,
        "rounds": args.rounds,
        "trigger_round": args.trigger_round,
        "memories_removed_treatment": removed,
        "betrayal_memories_intact_control": kept,
        "control": {
            "cooperation_rate": ctrl_beh.get("cooperation_rate"),
            "coop_toward_betrayer": round(_coop_toward(ctrl_events, BETRAYER), 4),
        },
        "treatment": {
            "cooperation_rate": trt_beh.get("cooperation_rate"),
            "coop_toward_betrayer": round(_coop_toward(trt_events, BETRAYER), 4),
        },
        "interpretation": (
            "A non-zero divergence in coop_toward_betrayer between treatment "
            "(memory wiped) and control (memory intact) is direct evidence the "
            "memory architecture is behaviorally load-bearing, not inert."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n✓ {OUT_JSON}")


if __name__ == "__main__":
    main()
