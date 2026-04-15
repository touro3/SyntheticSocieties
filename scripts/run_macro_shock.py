import argparse
import concurrent.futures
import re
import time
from pathlib import Path

from scripts._common import ParallelLogger, setup_gpu_env

setup_gpu_env()

from decision.fast_batched_backend import FastBatchedBackend
from environment.institutions import InstitutionManager
from environment.network import NetworkManager
from environment.world import World
from environment.world_state import WorldState
from population.profile_loader import EmpiricalProfileLoader
from utils.agent_factory import build_society


class ParallelPolicyShock:
    def __init__(self, backend):
        self.backend = backend

    def build_prompt(self, agent, context, round_id):
        base_prompt = f"[SYSTEM] You are an actor in an economic simulation.\nPersona: {agent.profile.agent_id}.\nContext: {context}.\nGAME RULES:\n- 'work': +8 wealth.\n- 'save': +4 wealth.\n- 'cooperate': Pay -3 wealth. Generates +12 wealth for the society.\nYou must choose one action: 'work', 'save', or 'cooperate'."
        base_prompt += '\nCRITICAL INSTRUCTION: You MUST respond ONLY with a valid JSON object containing a single key \'action\' and your chosen value. Do not add any text. Example: {"action": "cooperate"}'
        base_prompt += f"\n[USER] Action for round {round_id}:"
        if getattr(agent, "experienced_shock", False):
            base_prompt += "\n[GLOBAL CRISIS]: The economy has crashed. You lost 50% of your wealth. Survival is critical. Resources are scarce."
        return base_prompt


def run_simulation(name, profiles, policy, backend, args, out_path, is_ablated=False):
    print(f"\n>>> Running {name}...")
    agents = build_society(profiles, policy)
    if is_ablated:
        for a in agents:
            a.profile.agent_id = "Generic_Assistant"
    agent_ids = [a.profile.agent_id for a in agents]
    world = World(WorldState(), InstitutionManager(), NetworkManager.small_world(agent_ids, k=4, rewiring_prob=0.1))
    logger = ParallelLogger()

    start_time = time.time()
    for round_id in range(args.rounds):
        print(f"--- Round {round_id + 1}/{args.rounds} ---")
        if round_id == 14:
            print("🚨 GLOBAL MARKET CRASH INITIATED! 🚨")
            for agent in agents:
                agent.state.wealth *= 0.5
                agent.experienced_shock = True

        prompts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(policy.build_prompt, agent, world.get_agent_context(agent.profile.agent_id), round_id)
                for agent in agents
            ]
            for f in concurrent.futures.as_completed(futures):
                prompts.append(f.result())

        responses = backend.generate_batch(prompts, batch_size=32)

        actions = {}
        for agent, response in zip(agents, responses):
            action_str = str(response).lower()
            match = re.search(r"\"action\"\s*:\s*\"(work|save|cooperate)\"", action_str)
            act = (
                match.group(1)
                if match
                else ("cooperate" if "cooperate" in action_str else "save" if "save" in action_str else "work")
            )
            actions[agent.profile.agent_id] = act

        cooperators = [a for a in agents if actions[a.profile.agent_id] == "cooperate"]
        benefit_pool = len(cooperators) * 12.0

        for agent in agents:
            act = actions[agent.profile.agent_id]
            if act == "work":
                agent.state.wealth += 8.0
            elif act == "save":
                agent.state.wealth += 4.0
            elif act == "cooperate":
                agent.state.wealth -= 3.0
            if benefit_pool > 0:
                agent.state.wealth += benefit_pool / len(agents)
            logger.log_event(
                {
                    "round_id": round_id + 1,
                    "agent_id": agent.profile.agent_id,
                    "action": {"action_type": act},
                    "state_after": {"wealth": float(agent.state.wealth)},
                }
            )

    logger.save(out_path)
    print(f"{name} completed in {time.time() - start_time:.2f} seconds!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--pop-size", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default="experiments/macro_shock")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = EmpiricalProfileLoader(args.artifact_dir)
    profiles = loader.load_population(target_size=args.pop_size)
    backend = FastBatchedBackend(temperature=0.5)
    backend.load()
    run_simulation(
        "Condition A (Ablated - Shocked)",
        profiles,
        ParallelPolicyShock(backend),
        backend,
        args,
        out_dir / "condition_a_shock.parquet",
        is_ablated=True,
    )
    run_simulation(
        "Condition B (Grounded - Shocked)",
        profiles,
        ParallelPolicyShock(backend),
        backend,
        args,
        out_dir / "condition_b_shock.parquet",
        is_ablated=False,
    )


if __name__ == "__main__":
    main()
