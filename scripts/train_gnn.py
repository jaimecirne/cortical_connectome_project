import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import json
import networkx as nx

from generate_connectomes import generate_connectome_from_data


def ensure_directory_exists(path):
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def create_data_objects(connectomes):
    """
    Cria objetos de dados (Data) para todos os grafos na estrutura hierárquica de conectomas.

    Args:
        connectomes (dict): Dicionário aninhado estruturado como:
            connectomes[session][condition][window] = (graph, cluster_legend)

    Returns:
        list: Lista de objetos Data, cada um representando um grafo do conectoma.
        dict: Mapeamento de rótulos de condições para valores numéricos.
    """
    data_objects = []
    all_conditions = set()

    # Coleta todas as condições para criar o mapeamento de rótulos
    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            all_conditions.add(condition)

    condition_to_label = {condition: idx for idx, condition in enumerate(sorted(all_conditions))}

    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            # Obtém o rótulo numérico da condição
            if condition not in condition_to_label:
                print(f"Condição '{condition}' não encontrada no mapeamento.")
                continue
            label = condition_to_label[condition]

            for window, (graph, cluster_legend) in windows.items():
                # Ignorar gráficos sem nós
                if graph.number_of_nodes() == 0:
                    continue

                # Extrair nós, arestas e atributos
                nodes = list(graph.nodes)
                edges = list(graph.edges)
                edge_weights = [graph[u][v]['weight'] for u, v in edges]

                # Mapear nós para índices
                node_to_index = {node: idx for idx, node in enumerate(nodes)}
                indexed_edges = [[node_to_index[u], node_to_index[v]] for u, v in edges]

                # Criar tensores de arestas e atributos
                if len(indexed_edges) > 0:
                    edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0,), dtype=torch.float)

                # Criar features dos nós (placeholder com 32 features por nó)
                x = torch.ones((len(nodes), 32), dtype=torch.float)

                # Criar objeto Data com rótulo
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long)  # Rótulo atribuído aqui
                )

                # Adicionar metadados (opcional)
                data.metadata = {
                    'session': session,
                    'condition': condition,
                    'window': window
                }

                data_objects.append(data)

    return data_objects, condition_to_label

class GNNClassifier(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch=None):
        print(f"x shape: {x.shape}")
        print(f"edge_index shape: {edge_index.shape}")
        print(f"edge_index: {edge_index}")

        edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Primeira camada GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Segunda camada GCN
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Pooling global, se necessário
        if batch is not None:
            x = global_mean_pool(x, batch)

        x = self.lin(x)
        return F.log_softmax(x, dim=1)




# Definição de funções auxiliares primeiro
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        # Extraia os componentes necessários para o forward
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        optimizer.zero_grad()
        out = model(x, edge_index, batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Extraia os componentes necessários para o forward
            x, edge_index = data.x, data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None

            out = model(x, edge_index, batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.num_graphs

    return correct / total

def main():
    # Caminhos dos dados
    predator_csv = 'data/predator_data.csv'
    prey_csv = 'data/prey_data.csv'

    # Carregar e combinar dados
    data_predator = pd.read_csv(predator_csv)
    data_prey = pd.read_csv(prey_csv)
    data_df = pd.concat([data_predator, data_prey], ignore_index=True)

    # Gerar conectomas
    connectomes = generate_connectome_from_data(data_df)

    # Criar objetos de dados
    data_objects, condition_to_label = create_data_objects(connectomes)

    if not data_objects:
        print("Nenhum objeto de dados foi criado. Verifique os conectomas.")
        return

    # Dividir os dados em treinamento e teste
    labels = [data.y.item() for data in data_objects]
    train_data, test_data = train_test_split(data_objects, test_size=0.2, random_state=42, stratify=labels)

    # Configurar DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Determinar dimensões
    num_node_features = data_objects[0].x.shape[1]  # Número de features dos nós
    num_classes = len(condition_to_label)           # Número de classes

    print(f"num_node_features: {num_node_features}, num_classes: {num_classes}")

    # Configurar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=num_node_features, num_classes=num_classes).to(device)

    # Configurar otimizador e critério
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Treinar o modelo
    for epoch in range(1, 101):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        train_acc = test_model(model, train_loader, device)
        test_acc = test_model(model, test_loader, device)
        print(f'Época {epoch}, Loss: {train_loss:.4f}, Acurácia Treino: {train_acc:.4f}, Acurácia Teste: {test_acc:.4f}')

    # Salvar o modelo e o mapeamento
    torch.save(model.state_dict(), 'models/gnn_classifier.pth')
    with open('models/condition_to_label_mapping.json', 'w') as f:
        json.dump(condition_to_label, f)
    print("Modelo e mapeamento salvos.")

if __name__ == '__main__':
    main()
