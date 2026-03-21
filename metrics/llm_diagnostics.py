"""
LLM Diagnostics Module

Provides detailed runtime diagnostics for the LLM policy, focusing on:
1. Action histograms across agents and rounds.
2. Prompt-to-action contingency (how specific contexts influence outputs).
3. Context utilization (Percentage of prompts explicitly exposing 'trust' history).
4. Extracting raw JSON parsing errors or fallbacks.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import argparse


def load_llm_prompts_and_responses(experiment_id: str, experiments_root="experiments"):
    """Load prompts.jsonl and map them to their corresponding actions."""
    exp_path = Path(experiments_root) / experiment_id
    events_path = exp_path / "events.jsonl"
    prompts_path = exp_path / "prompts.jsonl"
    
    if not events_path.exists() or not prompts_path.exists():
        return [], []
        
    # Load events for action ground truth
    events = {}
    with events_path.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
                key = f"r{ev['round_id']}_a{ev['agent_id']}"
                events[key] = ev["action"]["action_type"]
            except Exception:
                pass
                
    # Load prompts
    diagnostics = []
    with prompts_path.open() as f:
        for line in f:
            try:
                p = json.loads(line)
                key = f"r{p['round_id']}_a{p['agent_id']}"
                action = events.get(key, "unknown")
                diag = {
                    "round": p["round_id"],
                    "agent": p["agent_id"],
                    "prompt_text": p["prompt"],
                    "raw_response": p["response"],
                    "parsed_action": action,
                    "has_social_context": "You have established several social ties" in p["prompt"] or "central figure" in p["prompt"] or "hop" in p["prompt"],
                    "has_trust_history": "Trust dictionary:" in p["prompt"] or "You have previously cooperated" in p["prompt"],
                    "has_stress_warning": "CRITICAL" in p["prompt"].upper() and "stress" in p["prompt"].lower()
                }
                diagnostics.append(diag)
            except Exception:
                pass
                
    return diagnostics


def diagnose_experiment(experiment_id: str):
    """Run diagnostics on a specific experiment and print the report."""
    diags = load_llm_prompts_and_responses(experiment_id)
    if not diags:
        print(f"No prompt logs found for {experiment_id}")
        return
        
    total = len(diags)
    actions = Counter(d["parsed_action"] for d in diags)
    
    # 1. Action Collapse
    print(f"\n--- LLM Diagnostics: {experiment_id} ---")
    print(f"Total Decisions: {total}")
    print("Action Distribution:")
    for a, c in actions.items():
        print(f"  {a:10s} : {c:4d} ({c/total:.1%})")
        
    # 2. Agent Level Histograms
    agent_actions = defaultdict(Counter)
    for d in diags:
        agent_actions[d["agent"]][d["parsed_action"]] += 1
        
    print("\nPer-Agent Action Histograms (Top 5 agents by activity):")
    sorted_agents = sorted(agent_actions.keys(), key=lambda k: sum(agent_actions[k].values()), reverse=True)[:5]
    for agent in sorted_agents:
        counts = agent_actions[agent]
        tot = sum(counts.values())
        print(f"  Agent {agent:4s}: Work={counts['work']/tot:.0%} Save={counts['save']/tot:.0%} Coop={counts['cooperate']/tot:.0%}")
        
    # 3. Context Utilization
    has_social = sum(1 for d in diags if d["has_social_context"])
    has_trust = sum(1 for d in diags if d["has_trust_history"])
    has_stress = sum(1 for d in diags if d["has_stress_warning"])
    
    print("\nContext Injection Rates:")
    print(f"  Social Topology (GraphRAG) : {has_social/total:.1%}")
    print(f"  Trust/Memory History     : {has_trust/total:.1%}")
    print(f"  Explicit Stress Warnings : {has_stress/total:.1%}")
    
    # 4. Contingency Table (Trust -> Action)
    if has_trust > 0:
        trust_actions = Counter(d["parsed_action"] for d in diags if d["has_trust_history"])
        no_trust_actions = Counter(d["parsed_action"] for d in diags if not d["has_trust_history"])
        
        print("\nContingency: Action rates WHEN Trust History is Present")
        for a in ["work", "save", "cooperate"]:
            t_rate = trust_actions[a] / has_trust if has_trust else 0
            n_rate = no_trust_actions[a] / (total - has_trust) if (total - has_trust) else 0
            print(f"  {a:10s} : With Trust={t_rate:.1%} | No Trust={n_rate:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", type=str, required=True, help="Experiment ID to diagnose (e.g. cmp_llm_s42)")
    args = parser.parse_args()
    diagnose_experiment(args.exp_id)


if __name__ == "__main__":
    main()
