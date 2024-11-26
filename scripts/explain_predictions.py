# scripts/explain_predictions.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json
import os

# Definir a classe do modelo (deve ser igual à usada no treinamento)
class GNNClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        if x.size(0) == 0:
            out = torch.zeros((batch.max().item() + 1, self.lin.out_features)).to(x.device)
            return F.log_softmax(out, dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return F.log_softmax(x, dim=1)

def load_model(model_path, num_node_features, num_classes, device):
    model = GNNClassifier(num_node_features, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def explain_prediction(model, data, target_class=None):
    # Certifique-se de que o modelo está no modo de avaliação
    model.eval()

    # Habilitar gradientes para os parâmetros
    for param in model.parameters():
        param.requires_grad = True

    data = data.to(next(model.parameters()).device)
    data.x.requires_grad = True  # Habilitar gradientes para as features dos nós

    # Forward pass
    out = model(data)
    pred = out.argmax(dim=1).item()

    if target_class is None:
        target_class = pred

    # Calcular a perda em relação à classe alvo
    loss = -out[0, target_class]
    loss.backward()

    # Gradientes das features dos nós
    node_importance = data.x.grad.abs().detach().cpu().numpy()

    # Agregar importâncias das arestas com base nas importâncias dos nós conectados
    edge_importance = []
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        node_i = edge_index[0, i]
        node_j = edge_index[1, i]
        importance = (node_importance[node_i] + node_importance[node_j]) / 2.0
        edge_importance.append(importance[0])  # Assumindo que x tem dimensão [num_nodes, 1]

    return edge_importance, pred

def visualize_attributions(data, edge_importance, condition_to_label, output_path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Converter o objeto Data para um grafo NetworkX
    G = to_networkx(data, edge_importance)

    # Definir as posições dos nós
    pos = nx.spring_layout(G, seed=42)

    # Normalizar as importâncias para uso no colormap
    edge_importance_values = np.array([G[u][v]['importance'] for u, v in G.edges()])
    if len(edge_importance_values) > 0:
        norm = plt.Normalize(vmin=edge_importance_values.min(), vmax=edge_importance_values.max())
        edge_colors = edge_importance_values
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
        edge_colors = [0.0 for _ in G.edges()]

    # Criar a figura e os eixos
    fig, ax = plt.subplots(figsize=(8, 6))

    # Desenhar os nós
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', ax=ax)

    # Desenhar as arestas
    edges = nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds,
        edge_vmin=norm.vmin,
        edge_vmax=norm.vmax,
        width=2,
        ax=ax
    )

    # Desenhar os rótulos dos nós
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', ax=ax)

    # Adicionar a barra de cores
    if len(edge_colors) > 0:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Importância da Aresta', rotation=270, labelpad=15)

    ax.set_title('Atribuições das Arestas')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"A imagem foi salva em {output_path}")
    else:
        plt.show()

    plt.close()


def to_networkx(data, edge_importance):
    G = nx.Graph()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))

    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i]
        v = edge_index[1, i]
        importance = edge_importance[i]
        G.add_edge(u, v, importance=importance)
    return G

def load_data_object(data_path, device):
    # Carregar o objeto Data salvo previamente
    data = torch.load(data_path, map_location=device)
    return data

def main():
    # Definir o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carregar o mapeamento de condições para rótulos
    label_mapping_file = 'models/condition_to_label_mapping.json'
    with open(label_mapping_file, 'r') as f:
        condition_to_label = json.load(f)

    num_classes = len(condition_to_label)
    num_node_features = 1  # Conforme definido no treinamento

    # Carregar o modelo treinado
    model_path = 'models/gnn_classifier.pth'
    model = load_model(model_path, num_node_features, num_classes, device)

    # Carregar um objeto de dados para explicar
    # Você pode ajustar o caminho para o seu arquivo
    data_object_path = 'data/sample_data_object.pt'  # Exemplo
    data = load_data_object(data_object_path, device)

    # Obter as atribuições e a predição
    edge_importance, pred = explain_prediction(model, data)

    # Inverter o mapeamento de rótulos para obter o nome da condição predita
    label_to_condition = {v: k for k, v in condition_to_label.items()}
    predicted_condition = label_to_condition.get(pred, 'Desconhecida')
    print(f"Condição predita: {predicted_condition}")

    # Visualizar as atribuições
    output_image_path = 'outputs/attributions.png'
    visualize_attributions(data, edge_importance, condition_to_label, output_path=output_image_path)

if __name__ == '__main__':
    main()
