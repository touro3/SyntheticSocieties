import argparse
import concurrent.futures
import time
from pathlib import Path

from scripts._common import ParallelLogger, setup_gpu_env

setup_gpu_env()

from agents.agent import Agent  # noqa: E402
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from decision.fast_batched_backend import FastBatchedBackend
from population.profile_loader import EmpiricalProfileLoader
from utils.agent_factory import build_society


class ParallelPolicy:
    def __init__(self, backend):
        self.backend = backend

    def build_prompt(self, agent, context, round_id):
        return f"[SYSTEM] You are an actor... (persona: {agent.profile.agent_id}) [USER] Action for round {round_id}"

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