import argparse
import json
from pathlib import Path
import polars as pl
import networkx as nx

def main():
    parser = argparse.ArgumentParser(description="Exporta logs da simulação para grafos do Gephi (.gexf)")
    parser.add_argument("--data-dir", type=str, default="experiments/phase_c_comparison", help="Diretório com os parquets")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    file_a = data_dir / "condition_a_events.parquet"
    file_b = data_dir / "condition_b_events.parquet"

    if not file_a.exists() or not file_b.exists():
        print(f"Erro: Arquivos Parquet não encontrados em {data_dir}.")
        return

    out_dir = Path("analysis/networks")
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path, name in [(file_a, "Condicao_A_Ablated"), (file_b, "Condicao_B_Grounded")]:
        print(f"Processando a rede de {name}...")
        df = pl.read_parquet(file_path)
        
        edges_weight = {}
        nodes_wealth = {}

        # Varre cada linha de evento da simulação
        for row in df.iter_rows(named=True):
            agent_id = row['agent_id']
            action = row['action']
            state_after = row['state_after']

            # Guarda a riqueza final do agente para colorir/dar tamanho ao nó no Gephi
            if isinstance(state_after, dict) and 'wealth' in state_after:
                nodes_wealth[agent_id] = state_after['wealth']

            # Extração blindada do alvo da cooperação
            target = None
            action_type = ""

            if isinstance(action, dict):
                action_type = str(action.get('action_type', '')).lower()
                target = action.get('target_agent_id')
            elif isinstance(action, str):
                try:
                    act_dict = json.loads(action)
                    action_type = str(act_dict.get('action_type', '')).lower()
                    target = act_dict.get('target_agent_id')
                except json.JSONDecodeError:
                    pass

            # Se a ação foi cooperar e tem um alvo válido, cria/fortalece a aresta (edge)
            if "cooperate" in action_type and target:
                edge = (agent_id, target)
                edges_weight[edge] = edges_weight.get(edge, 0) + 1

        # Montagem do Grafo Direcionado
        G = nx.DiGraph()
        
        # Adiciona os nós (Cidadãos)
        for node_id, wealth in nodes_wealth.items():
            G.add_node(node_id, final_wealth=float(wealth))
            
        # Adiciona as arestas (Relações de Cooperação)
        for (src, dst), weight in edges_weight.items():
            if src in G and dst in G:
                G.add_edge(src, dst, weight=float(weight))

        out_file = out_dir / f"{name}.gexf"
        nx.write_gexf(G, out_file)
        print(f"-> Exportado: {out_file} | Nós: {G.number_of_nodes()} | Arestas: {G.number_of_edges()}")

    print("\nSucesso! Arquivos .gexf gerados para visualização no Gephi.")

if __name__ == "__main__":
    main()