import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from decision.graph_rag import GraphRAG
from decision.sql_rag import SQLRAG


def test_graph_features():
    print("\n--- Testing GraphRAG ---")
    rag = GraphRAG()

    # Incremental update
    print("1. Adding cooperate event...")
    rag.add_event(
        {"agent_id": "Alice", "action": {"action_type": "cooperate", "target_agent_id": "Bob"}, "round_id": 1}
    )

    print(f"Context for Alice: {rag.get_social_context('Alice')}")
    print(f"Context for Unknown: {rag.get_social_context('Unknown')}")

    # k-hop test
    print("\n2. Adding chained interactions for k-hop...")
    rag.add_event({"agent_id": "Bob", "action": {"action_type": "cooperate", "target_agent_id": "Charlie"}})
    rag.add_event({"agent_id": "Charlie", "action": {"action_type": "cooperate", "target_agent_id": "David"}})

    print(f"Alice (2-hop) context: {rag.get_social_context('Alice', k_neighbors=2)}")
    print(f"Alice (3-hop) context: {rag.get_social_context('Alice', k_neighbors=3)}")


def test_sql_features():
    print("\n--- Testing SQLRAG ---")
    try:
        rag = SQLRAG(data_path="data/ess_clean.parquet")

        print("1. Peer group context (Age 31, Female, AT):")
        ctx = rag.get_peer_group_context(age=31, gender=2, country="AT")
        print(f"Result: {ctx}")

        print("\n2. Security Check (Injection prevention):")
        inj = rag.query_population_trends("SELECT * FROM population; DROP VIEW population;")
        print("Query: 'SELECT * FROM population; DROP VIEW population;'")
        if "DROP" in inj.upper():
            print("❌ WARNING: Possible injection leak!")
        else:
            print("✅ Successfully blocked or handled complex query.")

    except FileNotFoundError:
        print("❌ Error: data/ess_clean.parquet not found. Skipping SQL tests.")


if __name__ == "__main__":
    test_graph_features()
    test_sql_features()
    print("\nVerification Complete.")
