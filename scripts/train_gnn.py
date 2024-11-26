# scripts/train_gnn.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split

# Importar a função generate_connectome_from_data do seu script generate_connectomes.py
from generate_connectomes import generate_connectome_from_data

def create_data_objects(connectomes):
    """
    Creates data objects for all graphs in the new hierarchical structure of connectomes.

    Args:
        connectomes (dict): A nested dictionary structured as:
            connectomes[session][condition][window] = (graph, cluster_legend)

    Returns:
        list: A list of data objects, each representing a graph in the connectome.
        dict: A mapping from condition labels to numerical labels.
    """
    data_objects = []
    all_conditions = set()

    # Primeiro, coletar todas as condições únicas para criar o mapeamento de rótulos
    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            all_conditions.add(condition)

    condition_to_label = {condition: idx for idx, condition in enumerate(sorted(all_conditions))}

    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            label = condition_to_label[condition]
            for window, (graph, cluster_legend) in windows.items():
                # Ignorar gráficos sem nós
                if graph.number_of_nodes() == 0:
                    continue
                
                # Extrair nós e arestas
                nodes = list(graph.nodes)
                edges = list(graph.edges)
                edge_weights = [graph[u][v]['weight'] for u, v in edges]

                # Mapear nós para índices
                node_to_index = {node: idx for idx, node in enumerate(nodes)}
                indexed_edges = [[node_to_index[u], node_to_index[v]] for u, v in edges]

                # Converter para tensores
                if len(indexed_edges) > 0:
                    edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0,), dtype=torch.float)

                # Criar features dos nós (aqui usando 1.0 como placeholder)
                x = torch.ones((len(nodes), 1), dtype=torch.float)

                # Criar objeto Data
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor(label, dtype=torch.long)
                )

                # Adicionar metadados se necessário
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Tratar grafos sem arestas
        if edge_index.numel() == 0:
            x = self.conv1(x, torch.empty((2, 0), dtype=torch.long).to(x.device))
        else:
            x = self.conv1(x, edge_index)
        x = F.relu(x)

        if edge_index.numel() == 0:
            x = self.conv2(x, torch.empty((2, 0), dtype=torch.long).to(x.device))
        else:
            x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)

        x = self.lin(x)

        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
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
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
            total += data.num_graphs

    return correct / total

def main():
    # Caminhos dos arquivos CSV
    predator_csv = 'data/c5607a01.csv'
    prey_csv = 'data/c5103a01MUA.csv'

    # Carregar os dados do predador
    data_predator = pd.read_csv(predator_csv, delimiter=',')
    data_predator['Species'] = 'Predator'

    # Carregar os dados da presa
    data_prey = pd.read_csv(prey_csv, delimiter=',')
    data_prey['Species'] = 'Prey'

    # Combinar os dados
    data_df = pd.concat([data_predator, data_prey], ignore_index=True)

    # Gerar conectomas
    connectomes = generate_connectome_from_data(data_df)

    # Criar objetos de dados
    data_objects, condition_to_label = create_data_objects(connectomes)

    # Verificar se data_objects não está vazio
    if not data_objects:
        print("Nenhum objeto de dados foi criado. Verifique os conectomas.")
        return

    # Dividir os dados em conjuntos de treinamento e teste
    labels = [data.y.item() for data in data_objects]
    train_data, test_data = train_test_split(data_objects, test_size=0.2, random_state=42, stratify=labels)

    # Criar DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Determinar o número de features e classes
    num_node_features = data_objects[0].x.shape[1]
    num_classes = len(condition_to_label)

    # Inicializar o modelo, otimizador e função de perda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=num_node_features, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # Loop de treinamento
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        train_acc = test_model(model, train_loader, device)
        test_acc = test_model(model, test_loader, device)
        print(f'Época {epoch}, Loss: {loss:.4f}, Acurácia Treino: {train_acc:.4f}, Acurácia Teste: {test_acc:.4f}')

    # Salvar o modelo treinado
    torch.save(model.state_dict(), 'models/gnn_classifier.pth')
    print("Modelo salvo em 'models/gnn_classifier.pth'.")

    # Salvar o mapeamento de condições para rótulos
    label_mapping_file = 'models/condition_to_label_mapping.json'
    import json
    with open(label_mapping_file, 'w') as f:
        json.dump(condition_to_label, f)
    print(f"Mapeamento de rótulos salvo em '{label_mapping_file}'.")

if __name__ == '__main__':
    main()
