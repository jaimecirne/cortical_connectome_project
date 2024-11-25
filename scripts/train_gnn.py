# scripts/train_gnn.py

import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import os
import numpy as np
import networkx as nx
from torch.utils.data import random_split


def create_data_objects(connectomes):
    """
    Creates data objects for all graphs in the new hierarchical structure of connectomes.

    Args:
        connectomes (dict): A nested dictionary structured as:
            connectomes[session][condition][window] = (graph, cluster_legend)

    Returns:
        list: A list of data objects, each representing a graph in the connectome.
    """


    data_objects = []

    # Iterate over the hierarchical structure
    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            for window, (graph, cluster_legend) in windows.items():
                # Extract nodes and edges
                nodes = list(graph.nodes)
                edges = list(graph.edges)
                edge_weights = [graph[u][v]['weight'] for u, v in edges]

                # Map nodes to indices for PyTorch Geometric
                node_to_index = {node: idx for idx, node in enumerate(nodes)}
                indexed_edges = [[node_to_index[u], node_to_index[v]] for u, v in edges]

                # Convert edges and weights to tensors
                edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_weights, dtype=torch.float)

                # Create node features (if any, default to 1 for now)
                x = torch.ones((len(nodes), 1), dtype=torch.float)  # Replace with actual node features if available

                # Add metadata as attributes
                metadata = {
                    "session": session,
                    "condition": condition,
                    "window": window,
                    "cluster_legend": cluster_legend
                }

                # Create a PyTorch Geometric Data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    metadata=metadata
                )

                data_objects.append(data)

    return data_objects


def prepare_dataset(graph_dir):
    dataset = []
    for file_name in os.listdir(graph_dir):
        if file_name.endswith('.graphml'):
            condition = int(file_name.split('_')[-1].split('.')[0])
            graph_file = os.path.join(graph_dir, file_name)
            data = create_data_object(graph_file, condition)
            dataset.append(data)
    return dataset

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_gnn(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=32, num_classes=12).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
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
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Época {epoch+1}, Loss: {avg_loss}')
        model.eval()
        correct = 0
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
        val_acc = correct / len(val_loader.dataset)
        print(f'Acurácia na Validação: {val_acc:.4f}')
    model.eval()
    correct = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)
    print(f'Acurácia no Teste: {test_acc:.4f}')
    torch.save(model.state_dict(), '../models/gnn_classifier.pth')

if __name__ == "__main__":
    dataset = prepare_dataset('../outputs/predator_connectomes')
    train_gnn(dataset)
