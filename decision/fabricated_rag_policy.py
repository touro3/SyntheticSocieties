import random

from decision.sql_rag import SQLRAG


class FabricatedSQLRAG(SQLRAG):
    """
    Audit C.7: Fabricated Demographics Condition.
    Ignores the ESS parquet and emits a per-agent deterministic peer-context with
    ESS-plausible Gaussian priors, mirroring canonical sentence structure.
    """

    def __init__(self, db_connection, fabricate_seed=42, **kwargs):
        super().__init__(db_connection, **kwargs)
        self.fabricate_seed = fabricate_seed

    def get_peer_context(self, agent_id, demographics):
        random.seed(self.fabricate_seed + hash(agent_id))

        # Generate plausible Gaussian priors bounded [0, 10]
        trust_people = max(0, min(10, int(random.gauss(5.5, 2.0))))
        trust_institutions = max(0, min(10, int(random.gauss(4.8, 2.2))))
        risk_tolerance = max(0, min(10, int(random.gauss(5.0, 2.5))))

        context = (
            f"This agent has a generalized trust score of {trust_people} out of 10. "
            f"Their trust in institutions is {trust_institutions}/10, and their "
            f"risk tolerance evaluates to {risk_tolerance}/10. They prioritize "
            "security and conformity over self-direction."
        )
        return context
