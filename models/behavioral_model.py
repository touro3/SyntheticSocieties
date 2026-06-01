import numpy as np

'''Literature: McFadden (1974) - "Conditional logit analysis of qualitative choice behavior"'''


class BehavioralModel:
    """Original hardcoded McFadden utility model (v0.1)."""

    def utilities(self, profile, state, world):

        income = profile.income
        stress = state.stress
        risk = profile.risk_tolerance

        food_price = world["prices"]["food"]

        income_pressure = food_price / (income + 1)

        U_work = 1.2 * income_pressure + 0.3 * risk - 0.2 * stress

        U_save = 0.8 * stress + 0.4 * (1 - risk)

        U_cooperate = 0.5 * (1 - stress) + 0.3 * risk

        return np.array([U_work, U_save, U_cooperate])

    def probabilities(self, utilities):

        exp = np.exp(utilities)
        return exp / exp.sum()

    def choose(self, probs):

        return np.random.choice(["work", "save", "cooperate"], p=probs)


class EmpiricalBehavioralModel(BehavioralModel):
    """
    ESS-grounded behavioral model (v0.2).

    Uses empirical attributes (trust, social activity, political orientation,
    competitiveness) from the enriched AgentProfile to compute utilities.
    Falls back to the base model's logic when ESS attributes are unavailable.
    """

    def utilities(self, profile, state, world):
        income = profile.income
        stress = state.stress
        risk = profile.risk_tolerance

        food_price = world["prices"]["food"]
        income_pressure = food_price / (income + 1)

        # ESS-derived attributes (with safe defaults)
        trust = getattr(profile, "trust_people", None) or 0.5
        social = getattr(profile, "social_activity", None) or 0.5
        competitiveness = getattr(profile, "competitiveness", None) or 0.5
        health = getattr(profile, "health_status", None) or 0.5
        satisfaction = getattr(profile, "life_satisfaction", None) or 0.5

        # Work utility: income pressure + competitiveness + risk - stress
        U_work = (
            1.0 * income_pressure
            + 0.4 * competitiveness
            + 0.3 * risk
            - 0.2 * stress
            - 0.1 * satisfaction  # satisfied people less driven to work
        )

        # Save utility: stress + low risk + health concerns
        U_save = (
            0.6 * stress
            + 0.4 * (1 - risk)
            + 0.2 * (1 - health)  # poor health → save more
            + 0.1 * (1 - trust)  # low trust → save rather than cooperate
        )

        # Cooperate utility: trust + social activity + low stress
        U_cooperate = (
            0.5 * trust
            + 0.4 * social
            + 0.2 * (1 - stress)
            + 0.1 * satisfaction
            - 0.1 * competitiveness  # competitive people cooperate less
        )

        return np.array([U_work, U_save, U_cooperate])
