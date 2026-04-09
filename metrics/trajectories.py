"""
Extraction of round-by-round trajectories from experiment event logs.
Parses events.jsonl to reconstruct time-series data for wealth, stress, and actions.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def extract_trajectories(exp_dir: str | Path) -> dict:
    """
    Extract wealth, stress, and action trajectories from an experiment directory.
    
    Returns:
        dict: {
            'rounds': list of round_ids,
            'wealth': { round_id: [agent_wealths] },
            'stress': { round_id: [agent_stress] },
            'actions': { round_id: { action_type: count } },
            'agent_trajectories': { agent_id: { 'wealth': [], 'stress': [] } }
        }
    """
    exp_path = Path(exp_dir)
    events_path = exp_path / "events.jsonl"
    
    if not events_path.exists():
        return {}
    
    wealth_traj = defaultdict(list)
    stress_traj = defaultdict(list)
    action_counts = defaultdict(lambda: defaultdict(int))
    agent_history = defaultdict(lambda: {"wealth": [], "stress": []})
    
    with events_path.open() as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            r_id = event["round_id"]
            state = event["state_after"]
            a_id = event["agent_id"]
            
            wealth_traj[r_id].append(state["wealth"])
            stress_traj[r_id].append(state["stress"])
            
            action = event["action"]["action_type"]
            action_counts[r_id][action] += 1
            
            # Note: this assumes events are ordered by round then agent
            # For a more robust version, we'd sort by round_id
            agent_history[a_id]["wealth"].append(state["wealth"])
            agent_history[a_id]["stress"].append(state["stress"])
            
    rounds = sorted(wealth_traj.keys())
    
    return {
        "rounds": rounds,
        "wealth": wealth_traj,
        "stress": stress_traj,
        "actions": action_counts,
        "agent_trajectories": agent_history
    }

def aggregate_seeds(policy: str, seeds: list[int], experiments_root: str | Path = "experiments") -> dict:
    """
    Aggregate trajectories across multiple seeds for a given policy.
    """
    root = Path(experiments_root)
    all_wealth = [] # list of [rounds x agents] arrays
    all_stress = []
    all_action_freqs = [] # list of [rounds x 3] arrays
    
    # Mapping to get expected ID prefix (llm -> cmp_llm, template -> cmp_template)
    # We'll try common prefixes
    prefix_candidates = ["cmp_", "ablation_", "pert_"]
    
    found_seeds = 0
    for seed in seeds:
        data = None
        for pref in prefix_candidates:
            # This is a bit brittle, relies on standard naming
            # Handle special cases if needed
            name_map = {"llm": "llm", "template": "template", "rule_based": "rule", "random": "random"}
            prefix_name = name_map.get(policy, policy)
            exp_id = f"{pref}{prefix_name}_s{seed}"
            exp_dir = root / exp_id
            if exp_dir.exists():
                data = extract_trajectories(exp_dir)
                if data:
                    break
        
        if not data:
            continue
            
        found_seeds += 1
        rounds = data["rounds"]
        
        # Wealth [ Rounds x Agents ]
        w_matrix = np.array([data["wealth"][r] for r in rounds])
        all_wealth.append(w_matrix)
        
        # Stress [ Rounds x Agents ]
        s_matrix = np.array([data["stress"][r] for r in rounds])
        all_stress.append(s_matrix)
        
        # Actions [ Rounds x 3 ] (work, save, cooperate)
        a_types = ["work", "save", "cooperate"]
        a_matrix = []
        for r in rounds:
            counts = data["actions"][r]
            total = max(sum(counts.values()), 1)
            a_matrix.append([counts.get(at, 0) / total for at in a_types])
        all_action_freqs.append(np.array(a_matrix))
        
    if found_seeds == 0:
        return {}
        
    # Standardize length across seeds (trim to min rounds found)
    min_rounds = min(m.shape[0] for m in all_wealth)
    all_wealth = [m[:min_rounds, :] for m in all_wealth]
    all_stress = [m[:min_rounds, :] for m in all_stress]
    all_action_freqs = [m[:min_rounds, :] for m in all_action_freqs]
    
    # Aggregate across seeds
    # Wealth Mean/Std [ Rounds ] (pooled agents*seeds)
    pool_wealth = np.concatenate(all_wealth, axis=1) # [ Rounds x (Agents * Seeds) ]
    pool_stress = np.concatenate(all_stress, axis=1)
    
    # Action Freqs [ Rounds x 3 ] (average across seeds)
    mean_actions = np.mean(all_action_freqs, axis=0)
    
    return {
        "rounds": list(range(min_rounds)),
        "n_seeds": found_seeds,
        "wealth_mean": np.mean(pool_wealth, axis=1),
        "wealth_std": np.std(pool_wealth, axis=1),
        "stress_mean": np.mean(pool_stress, axis=1),
        "stress_std": np.std(pool_stress, axis=1),
        "action_freqs": mean_actions, # [Rounds, 3]
        "action_labels": ["work", "save", "cooperate"],
        "pool_wealth": pool_wealth,   # [Rounds x (Agents * Seeds)]
        "pool_stress": pool_stress,   # [Rounds x (Agents * Seeds)]
    }

