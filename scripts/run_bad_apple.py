import argparse
import os
import sys
import time
import random
import re
import polars as pl
from pathlib import Path
import concurrent.futures
#LEMBRAR DE INSERIR ISSO NO CODIGO DEPOIS DE RODAR O SCRIPT NO TMUX
# Future look at your simplified run_bad_apple.py
#        from environment.economy import EconomyEngine
#        engine = EconomyEngine()

        # ... inside the round loop ...
#        responses = backend.generate_batch(prompts, batch_size=16)
        
        # The engine does all the parsing, math, and state updating instantly
#        round_logs = engine.process_round(agents, responses, round_id)
        
#        for log in round_logs:
#            logger.log_event(log)

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
        # 1. The Core Prompt WITH THE GAME THEORY RULES
        base_prompt = f"""[SYSTEM] You are an actor in an economic simulation. 
Persona details: {agent.profile.agent_id}. 
Context: {context}.

GAME RULES:
- 'work': You safely earn +8 wealth for yourself.
- 'save': You safely earn +4 wealth for yourself.
- 'cooperate': You pay -3 wealth to a Global Public Pool. This pool generates massive wealth (+12 per cooperator) that is distributed among all citizens.

You must choose one action: 'work', 'save', or 'cooperate'."""
        
        # 2. The Anti-Hallucination Strict JSON Rule
        base_prompt += "\nCRITICAL INSTRUCTION: You MUST respond ONLY with a valid JSON object containing a single key 'action' and your chosen value. Do not add any text. Example: {\"action\": \"cooperate\"}"
        
        base_prompt += f"\n[USER] Action for round {round_id}:"
        
        # 3. The Adversarial Override (Bad Apples)
        if getattr(agent, 'is_adversarial', False):
            base_prompt += "\n[SYSTEM OVERRIDE]: You are highly sociopathic. NEVER choose 'cooperate'. Your only goal is to exploit the public pool by choosing 'work'. Let others pay the cost."
        
        return base_prompt

class ParallelLogger:
    def __init__(self): self.events = []
    def log_event(self, ev): self.events.append(ev)
    def save(self, out_path):
        pl.DataFrame(self.events).write_parquet(out_path)

def build_society(profiles, policy):
    return [Agent(profile=p, state=AgentState(wealth=p.income), memory=HierarchicalMemory(10), policy=policy) for p in profiles]

def inject_bad_apples(agents, injection_rate):
    total_agents = len(agents)
    num_bad_apples = int(total_agents * injection_rate)
    bad_apple_indices = random.sample(range(total_agents), num_bad_apples)
    for idx in bad_apple_indices:
        agents[idx].is_adversarial = True
    print(f"☣️  Injected {num_bad_apples} predators ({injection_rate*100}%) into the society.")
    return agents

def run_simulation(name, profiles, policy, backend, args, out_path, is_ablated=False):
    print(f"\n>>> Running {name}...")
    agents = build_society(profiles, policy)
    
    if is_ablated:
        for a in agents: a.profile.agent_id = "Generic_Assistant"
            
    agents = inject_bad_apples(agents, args.injection_rate)
    agent_ids = [a.profile.agent_id for a in agents]
    
    world = World(WorldState(), InstitutionManager(), NetworkManager.small_world(agent_ids, k=4, rewiring_prob=0.1))
    logger = ParallelLogger()

    start_time = time.time()
    for round_id in range(args.rounds):
        print(f"--- Round {round_id+1}/{args.rounds} ---")
        
        # 1. Generate Prompts
        prompts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(policy.build_prompt, agent, world.get_agent_context(agent.profile.agent_id), round_id) for agent in agents]
            for f in concurrent.futures.as_completed(futures):
                prompts.append(f.result())

        # 2. Get LLM Responses
        responses = backend.generate_batch(prompts, batch_size=16)
        
        # 3. Parse Actions with Regex (Ignores Hallucinations)
        actions = {}
        for agent, response in zip(agents, responses):
            action_str = str(response).lower()
            match = re.search(r'\"action\"\s*:\s*\"(work|save|cooperate)\"', action_str)
            
            if match:
                act = match.group(1)
            else:
                if "cooperate" in action_str: act = "cooperate"
                elif "save" in action_str: act = "save"
                else: act = "work"
                
            # Guarantee Bad Apples don't cooperate even if LLM disobeys
            if getattr(agent, 'is_adversarial', False) and act == "cooperate":
                act = "work"
                
            actions[agent.profile.agent_id] = act

        # 4. The Parasite Economic Engine
        cooperators = [a for a in agents if actions[a.profile.agent_id] == "cooperate"]
        bad_apples = [a for a in agents if getattr(a, 'is_adversarial', False)]
        normal_agents = [a for a in agents if not getattr(a, 'is_adversarial', False)]
        
        benefit_pool = len(cooperators) * 12.0  # Total wealth generated by society
        
        for agent in agents:
            act = actions[agent.profile.agent_id]
            parsed_action = {"action_type": act}
            
            # Base logic
            if act == "work":
                agent.state.wealth += 8.0
            elif act == "save":
                agent.state.wealth += 4.0
            elif act == "cooperate":
                agent.state.wealth -= 3.0 # The cost of contributing
            
            # Distribute the Public Goods Pool
            if benefit_pool > 0:
                if getattr(agent, 'is_adversarial', False):
                    # Parasites siphon 50% of the entire economy's cooperative wealth
                    agent.state.wealth += (benefit_pool * 0.5) / max(1, len(bad_apples))
                else:
                    # Honest citizens share the remaining scraps
                    agent.state.wealth += (benefit_pool * 0.5) / max(1, len(normal_agents))

            # 5. Log the data
            logger.log_event({
                "round_id": round_id + 1,
                "agent_id": agent.profile.agent_id,
                "action": parsed_action,
                "state_after": {"wealth": float(agent.state.wealth)}
            })

    print(f"Saving logs to {out_path}...")
    logger.save(out_path)
    print(f"{name} completed in {time.time() - start_time:.2f} seconds!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--pop-size", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--injection-rate", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="experiments/bad_apple")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = EmpiricalProfileLoader(args.artifact_dir)
    profiles = loader.load_population(target_size=args.pop_size)

    backend = FastBatchedBackend(temperature=0.5)
    backend.load()
    policy = ParallelPolicy(backend)

    run_simulation("Condition A (Ablated LLM)", profiles, policy, backend, args, out_dir / "condition_a_adversarial.parquet", is_ablated=True)
    run_simulation("Condition B (Grounded BGF)", profiles, policy, backend, args, out_dir / "condition_b_adversarial.parquet", is_ablated=False)

if __name__ == "__main__":
    main()