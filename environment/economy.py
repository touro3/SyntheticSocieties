import re
from typing import List, Dict, Any

class EconomyEngine:
    """
    Handles parsing LLM actions, executing Game Theory economic rules, 
    and updating agent states efficiently.
    """
    def __init__(self):
        # Compiling the regex once at initialization saves CPU cycles
        self.action_regex = re.compile(r'\"action\"\s*:\s*\"(work|save|cooperate)\"')
        
        # Core Game Theory Payoffs
        self.PAYOFF_WORK = 8.0
        self.PAYOFF_SAVE = 4.0
        self.COST_COOPERATE = 3.0
        self.MULTIPLIER_COOPERATE = 12.0

    def parse_action(self, response: str, is_adversarial: bool) -> str:
        """Extracts the action from the LLM, enforcing adversarial constraints."""
        response_lower = str(response).lower()
        match = self.action_regex.search(response_lower)
        
        if match:
            act = match.group(1)
        else:
            # Fallback for LLM hallucinations
            if "cooperate" in response_lower: act = "cooperate"
            elif "save" in response_lower: act = "save"
            else: act = "work"
            
        # Hard mathematical override: Bad Apples can NEVER cooperate
        if is_adversarial and act == "cooperate":
            return "work"
            
        return act

    def process_round(self, agents: List[Any], responses: List[str], round_id: int) -> List[Dict[str, Any]]:
        """
        Executes a single economic round. 
        Updates agent wealth in-place and returns the formatted log events.
        """
        parsed_actions = {}
        cooperators_count = 0
        bad_apples_count = 0
        normal_agents_count = 0

        # 1. Parse all actions in O(N) time
        for agent, response in zip(agents, responses):
            is_adv = getattr(agent, 'is_adversarial', False)
            act = self.parse_action(response, is_adv)
            parsed_actions[agent.profile.agent_id] = act
            
            if act == "cooperate":
                cooperators_count += 1
            
            if is_adv:
                bad_apples_count += 1
            else:
                normal_agents_count += 1

        # 2. Calculate the Global Public Goods Pool
        benefit_pool = cooperators_count * self.MULTIPLIER_COOPERATE
        
        # 3. Distribute Wealth and Generate Logs
        round_logs = []
        for agent in agents:
            agent_id = agent.profile.agent_id
            act = parsed_actions[agent_id]
            is_adv = getattr(agent, 'is_adversarial', False)
            
            # Apply Base Payoffs
            if act == "work":
                agent.state.wealth += self.PAYOFF_WORK
            elif act == "save":
                agent.state.wealth += self.PAYOFF_SAVE
            elif act == "cooperate":
                agent.state.wealth -= self.COST_COOPERATE
                
            # Apply Pool Distribution (The Parasite Mechanic)
            if benefit_pool > 0:
                if is_adv:
                    # Predators steal 50% of the entire pool
                    agent.state.wealth += (benefit_pool * 0.5) / max(1, bad_apples_count)
                else:
                    # Honest citizens share the rest
                    agent.state.wealth += (benefit_pool * 0.5) / max(1, normal_agents_count)

            # Build the log event
            round_logs.append({
                "round_id": round_id + 1,
                "agent_id": agent_id,
                "action": {"action_type": act},
                "state_after": {"wealth": float(agent.state.wealth)}
            })

        return round_logs
