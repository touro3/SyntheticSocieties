import argparse
import os
import sys
import time
from pathlib import Path
import concurrent.futures

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agents.memory import HierarchicalMemory
from agents.state import AgentState
from agents.agent import Agent
from environment.world_state import WorldState
from environment.world import World
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from decision.fast_batched_backend import FastBatchedBackend
from population.profile_loader import EmpiricalProfileLoader

class ParallelPolicy:
    def __init__(self, backend):
        self.backend = backend
    
    def build_prompt(self, agent, context, round_id):
        return f"[SYSTEM] You are an actor... (persona: {agent.profile.agent_id}) [USER] Action for round {round_id}"

class ParallelLogger:
    def __init__(self): self.events = []
    def log_event(self, ev): self.events.append(ev)

def build_society(profiles, policy):
    agents = []
    for p in profiles:
        agents.append(Agent(profile=p, state=AgentState(wealth=p.income), memory=HierarchicalMemory(10), policy=policy))
    return agents

def main():
    parser = argparse.ArgumentParser(description="Phase D: High-Scale Simulation")
    parser.add_argument("--artifact-dir", required=True, help="Path to Phase A fidelity artifact dir")
    parser.add_argument("--pop-size", type=int, default=500, help="Number of agents (Escalado!)")
    parser.add_argument("--rounds", type=int, default=30, help="Number of rounds")
    args = parser.parse_args()

    print(f"Initializing Phase D... Scaling to {args.pop_size} agents.")
    loader = EmpiricalProfileLoader(args.artifact_dir)
    profiles = loader.load_population(target_size=args.pop_size)
    agent_ids = [p.agent_id for p in profiles]

    backend = FastBatchedBackend(temperature=0.5)
    backend.load()

    policy = ParallelPolicy(backend)
    agents = build_society(profiles, policy)
    world = World(WorldState(), InstitutionManager(), NetworkManager.small_world(agent_ids, k=4, rewiring_prob=0.1))
    logger = ParallelLogger()

    start_time = time.time()
    
    for round_id in range(args.rounds):
        print(f"--- Processing Round {round_id+1}/{args.rounds} ---")
        
        prompts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for agent in agents:
                ctx = world.get_agent_context(agent.profile.agent_id)
                futures.append(executor.submit(policy.build_prompt, agent, ctx, round_id))
            
            for f in concurrent.futures.as_completed(futures):
                prompts.append(f.result())

        responses = backend.generate_batch(prompts, batch_size=32)
        
    print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()