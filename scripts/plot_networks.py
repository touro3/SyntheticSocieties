import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_network(gexf_path, output_path, title):
    print(f"Renderizando visualização para {gexf_path.name}...")
    G = nx.read_gexf(gexf_path)
    
    wealth_values = [float(data.get('final_wealth', 10.0)) for _, data in G.nodes(data=True)]
    wealth_values = np.array(wealth_values)
    
    weights = [float(data.get('weight', 1.0)) for _, _, data in G.edges(data=True)]
    
    min_w, max_w = wealth_values.min(), wealth_values.max()
    if max_w == min_w:
        node_sizes = np.full_like(wealth_values, 150)
    else:
        node_sizes = 100 + 700 * ((wealth_values - min_w) / (max_w - min_w))
        
    if weights:
        max_weight = max(weights)
        edge_widths = [0.5 + 4.0 * (w / max_weight) for w in weights]
    else:
        edge_widths = [0.75] * len(G.edges())

    pos = nx.spring_layout(G, k=0.2, iterations=2000, seed=42)
    
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')
    ax.set_title(title, fontsize=28, fontweight='bold', fontname='serif', pad=25)
    ax.axis('off')
    
    # Adicionado arrows=True e arrowsize=8 para habilitar as curvas
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color='dimgray',
        alpha=0.35,
        arrows=True,
        arrowsize=8,
        connectionstyle="arc3,rad=0.1" 
    )
    
    scatter = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=wealth_values,
        cmap=plt.cm.coolwarm,
        linewidths=0.5,
        edgecolors='darkgray' 
    )
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.3, pad=0.03, aspect=35)
    cbar.set_label('Capital Acumulado', fontsize=18, fontname='serif', rotation=270, labelpad=30)
    cbar.ax.tick_params(labelsize=14)
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92)

    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f" -> Arquivo salvo em: {output_path}\n")

def main():
    input_dir = Path("analysis/networks")
    file_a = input_dir / "Condicao_A_Ablated.gexf"
    file_b = input_dir / "Condicao_B_Grounded.gexf"
    
    out_dir = Path("analysis/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if file_a.exists():
        plot_network(file_a, out_dir / "grafo_A_ablated.png", "Condição A: Rede padrão com LLM")
    
    if file_b.exists():
        plot_network(file_b, out_dir / "grafo_B_grounded.png", "Condição B: Rede com LLM e memória de longo prazo")

if __name__ == "__main__":
    main()