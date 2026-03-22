import argparse
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agents.memory import HierarchicalMemory
from agents.state import AgentState

try:
    from agents.agent import Agent
except ModuleNotFoundError:
    from agents import Agent

from environment.world_state import WorldState
from environment.world import World
from environment.institutions import InstitutionManager
from environment.network import NetworkManager

from decision.llm_backend import LLMBackend
from decision.llm_policy import LLMPolicy
from decision.ablated_llm_policy import AblatedLLMPolicy
from population.profile_loader import EmpiricalProfileLoader

class SimpleLogger:
    def __init__(self, name):
        self.name = name
        self.events = []
    def log_event(self, event):
        self.events.append(event)
        
def build_society(profiles, policy):
    agents = []
    for p in profiles:
        state = AgentState(wealth=p.income)
        memory = HierarchicalMemory(max_recent=10)
        agents.append(Agent(profile=p, state=state, memory=memory, policy=policy))
    return agents

def main():
    parser = argparse.ArgumentParser(description="Phase C: Grounding Comparison Simulation")
    parser.add_argument("--artifact-dir", required=True, help="Path to Phase A fidelity artifact dir")
    parser.add_argument("--pop-size", type=int, default=50, help="Number of agents to simulate")
    parser.add_argument("--rounds", type=int, default=30, help="Number of rounds to run")
    args = parser.parse_args()

    print(f"Loading Empirical Profiles from {args.artifact_dir}...")
    loader = EmpiricalProfileLoader(args.artifact_dir)
    profiles = loader.load_population(target_size=args.pop_size)
    agent_ids = [p.agent_id for p in profiles]

    backend = LLMBackend(model_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.5)
    backend.load()

    network = NetworkManager.small_world(agent_ids, k=4, rewiring_prob=0.1, seed=42)
    institution = InstitutionManager()

    print("\n--- Starting Condition A: LLM Alone (Ablated) ---")
    ablated_policy = AblatedLLMPolicy(backend=backend, ablation="no_persona", temperature=0.5)
    agents_a = build_society(profiles, ablated_policy)
    
    world_a = World(WorldState(), institution, network)
    logger_a = SimpleLogger("Condition_A")
    
    from simulation.kernel import SimulationKernel
    sim_a = SimulationKernel(agents=agents_a, world=world_a, logger=logger_a)
    sim_a.run(num_rounds=args.rounds)
    print("Condition A completed.")

    print("\n--- Starting Condition B: Grounded LLM ---")
    grounded_policy = LLMPolicy(backend=backend, temperature=0.5)
    agents_b = build_society(profiles, grounded_policy)
    
    world_b = World(WorldState(), institution, network)
    logger_b = SimpleLogger("Condition_B")
    
    sim_b = SimulationKernel(agents=agents_b, world=world_b, logger=logger_b)
    sim_b.run(num_rounds=args.rounds)
    print("Condition B completed.")

    output_dir = Path("experiments/phase_c_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import polars as pl
    pl.DataFrame(logger_a.events).write_parquet(output_dir / "condition_a_events.parquet")
    pl.DataFrame(logger_b.events).write_parquet(output_dir / "condition_b_events.parquet")
    
    print(f"\nPhase C Complete! Logs saved to {output_dir}")

if __name__ == "__main__":
    main()