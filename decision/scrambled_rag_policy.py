import random
from decision.sql_rag import SQLRAG


class ScrambledSQLRAG(SQLRAG):
    """
    Audit C.6: Scrambled-ESS Condition.
    Deterministically permutes the demographic tuple before delegating to the parent query.
    Agent self-report fields pass through unchanged so only the Φ mapping is broken.
    """

    def __init__(self, db_connection, scramble_seed=42, **kwargs):
        super().__init__(db_connection, **kwargs)
        self.scramble_seed = scramble_seed
        self.permutation_map = {}

    def get_peer_context(self, agent_id, demographics):
        # Deterministically pick a different agent's demographics based on the seed
        random.seed(self.scramble_seed + hash(agent_id))

        # Scramble logic: shuffle the keys of the demographic tuple
        scrambled_demographics = demographics.copy()
        keys = list(scrambled_demographics.keys())
        shuffled_keys = keys.copy()
        random.shuffle(shuffled_keys)

        for k, sk in zip(keys, shuffled_keys):
            scrambled_demographics[k] = demographics[sk]

        return super().get_peer_context(agent_id, scrambled_demographics)
