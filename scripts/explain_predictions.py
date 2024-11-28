import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GNNExplainer
import matplotlib.pyplot as plt


def load_model_and_data():
    """
    Função para carregar o modelo e os dados. 
    Substitua esta função com o carregamento específico da sua arquitetura.
    """
    # Exemplo fictício de modelo e dados
    model = torch.load("path_to_your_model.pth")
    data = torch.load("path_to_your_data.pth")
    return model, data


def explain_predictions(model, data, node_index=None, edge_mask_threshold=0.5):
    """
    Gera explicações para as predições do modelo usando GNNExplainer.

    Args:
        model: Modelo GNN treinado.
        data: Dados de entrada.
        node_index: Índice do nó a ser explicado (opcional).
        edge_mask_threshold: Limite para exibição de máscaras de arestas.

    Returns:
        Explicações gráficas e métricas relacionadas às predições.
    """
    explainer = GNNExplainer(model, epochs=200, return_type='log_prob')

    # Explicação do nó específico ou do grafo inteiro
    if node_index is not None:
        node_feat_mask, edge_mask = explainer.explain_node(node_index, data.x, data.edge_index)
        visualize_node_explanation(data, node_index, edge_mask, edge_mask_threshold)
    else:
        graph_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
        visualize_graph_explanation(data, edge_mask, edge_mask_threshold)


def visualize_node_explanation(data, node_index, edge_mask, edge_mask_threshold):
    """
    Visualiza explicações para um nó específico.

    Args:
        data: Dados do grafo.
        node_index: Índice do nó a ser explicado.
        edge_mask: Máscara de arestas gerada pelo GNNExplainer.
        edge_mask_threshold: Limite para exibição de arestas.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    edge_weights = edge_mask > edge_mask_threshold

    plt.figure(figsize=(10, 8))
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G),
        with_labels=True,
        node_color='lightblue',
        edge_color=edge_weights.numpy(),
        width=edge_mask.numpy() * 5
    )
    plt.title(f"Explicação para o nó {node_index}")
    plt.show()


def visualize_graph_explanation(data, edge_mask, edge_mask_threshold):
    """
    Visualiza explicações para o grafo inteiro.

    Args:
        data: Dados do grafo.
        edge_mask: Máscara de arestas gerada pelo GNNExplainer.
        edge_mask_threshold: Limite para exibição de arestas.
    """
    import networkx as nx
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    edge_weights = edge_mask > edge_mask_threshold

    plt.figure(figsize=(10, 8))
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G),
        with_labels=True,
        node_color='lightblue',
        edge_color=edge_weights.numpy(),
        width=edge_mask.numpy() * 5
    )
    plt.title("Explicação do Grafo")
    plt.show()


if __name__ == "__main__":
    # Carrega modelo e dados
    model, data = load_model_and_data()

    # Escolha um nó para explicar (ou None para explicar o grafo inteiro)
    node_index_to_explain = None  # Altere para o índice do nó, se necessário

    # Executa explicação
    explain_predictions(model, data, node_index=node_index_to_explain)
