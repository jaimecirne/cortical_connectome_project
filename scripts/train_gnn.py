# train_gnn.py

import pandas as pd
import networkx as nx
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, Dataset
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.loader import DataLoader
from generate_connectomes import generate_connectome_from_data
import numpy as np
import os
import json

def pad_features(graphs, max_nodes):
    """
    Pads the feature matrices of graphs to have consistent sizes.

    Args:
        graphs (list of networkx.Graph): List of graph objects.
        max_nodes (int): Maximum number of nodes across all graphs.

    Returns:
        list of torch_geometric.data.Data: List of padded graph data objects.
    """
    padded_graphs = []
    for idx, G in enumerate(graphs):
        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            print(f"Warning: Empty graph found (index {idx}) and will be ignored.")
            continue

        x = torch.zeros((max_nodes, max_nodes), dtype=torch.float)
        x[:num_nodes, :num_nodes] = torch.eye(num_nodes, dtype=torch.float)

        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            print(f"Warning: Graph {idx} has no edges.")
            continue

        edge_index = edge_index[:, (edge_index < max_nodes).all(dim=0)]

        if edge_index.size(1) == 0:
            print(f"Warning: Graph {idx} has no edges after padding.")
            continue

        padded_graphs.append(Data(x=x, edge_index=edge_index))
    return padded_graphs

class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        super(GraphDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        label = self.labels[idx]
        data.y = torch.tensor([label], dtype=torch.long)
        return data

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data, edge_weight=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_max_pool(x, batch)  # Max pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, data, edge_weight=None):
        """
        Extracts the graph embedding before the final classification layer.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_max_pool(x, batch)
        return x

def load_data(predator_csv, prey_csv):
    """
    Loads and combines predator and prey data from CSV files.

    Args:
        predator_csv (str): Path to the predator data CSV file.
        prey_csv (str): Path to the prey data CSV file.

    Returns:
        pandas.DataFrame: Combined data.
    """
    data_predator = pd.read_csv(predator_csv)
    data_prey = pd.read_csv(prey_csv)

    data_predator['especie'] = 'predator'
    data_prey['especie'] = 'prey'

    data = pd.concat([data_predator, data_prey], ignore_index=True)
    return data

def prepare_graph_data(data):
    """
    Generates separate connectomes for prey and predator, and prepares graph data and labels.

    Args:
        data (pandas.DataFrame): Combined data.

    Returns:
        tuple: (graphs, labels, label_map, condition_to_label)
    """
    # Filtra os dados por espécie
    data_prey = data[data['especie'] == 'prey']
    data_predator = data[data['especie'] == 'predator']

    # Gera connectomes para cada espécie separadamente
    connectomes_prey = generate_connectome_from_data(
        data_prey,
        display_directed=False,
        coherence_threshold=0.01,
        top_k=None,
        n_jobs=-1
    )
    connectomes_predator = generate_connectome_from_data(
        data_predator,
        display_directed=False,
        coherence_threshold=0.01,
        top_k=None,
        n_jobs=-1
    )

    # Mapeia rótulos
    label_map = {'predator': 0, 'prey': 1}

    # Inicializa listas para gráficos, rótulos e mapeamento de condições
    graphs = []
    labels = []
    condition_to_label = {}

    # Processa os connectomes para "prey"
    for session, session_data in connectomes_prey.items():
        for condition, condition_data in session_data.items():
            for window, (G, _) in condition_data.items():
                if G is not None:
                    graphs.append(G)
                    labels.append(label_map['prey'])
                    condition_to_label[f"{session}_{condition}_{window}"] = label_map['prey']

    # Processa os connectomes para "predator"
    for session, session_data in connectomes_predator.items():
        for condition, condition_data in session_data.items():
            for window, (G, _) in condition_data.items():
                if G is not None:
                    graphs.append(G)
                    labels.append(label_map['predator'])
                    condition_to_label[f"{session}_{condition}_{window}"] = label_map['predator']

    return graphs, labels, label_map, condition_to_label

def create_dataset(graphs, labels):
    """
    Creates a dataset from graphs and labels.

    Args:
        graphs (list): List of graph objects.
        labels (list): List of labels.

    Returns:
        tuple: (dataset, max_nodes)
    """
    max_nodes = max(graph.number_of_nodes() for graph in graphs)
    padded_graphs = pad_features(graphs, max_nodes)
    dataset = GraphDataset(padded_graphs, labels)
    return dataset, max_nodes

def balance_dataset(dataset, max_nodes):
    """
    Balances the dataset using SMOTE.

    Args:
        dataset (GraphDataset): Original dataset.
        max_nodes (int): Maximum number of nodes.

    Returns:
        GraphDataset: Balanced dataset.
    """
    X = torch.stack([data.x.flatten() for data in dataset]).numpy()
    y = np.array([data.y.item() for data in dataset])
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_graphs = []
    for i in range(len(X_resampled)):
        original_index = i % len(dataset)
        edge_index = dataset[original_index].edge_index
        x_resampled = torch.tensor(X_resampled[i], dtype=torch.float32).reshape(max_nodes, -1)
        resampled_graphs.append(Data(x=x_resampled, edge_index=edge_index))

    resampled_dataset = GraphDataset(resampled_graphs, y_resampled.tolist())
    return resampled_dataset

def split_dataset(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets.

    Args:
        dataset (GraphDataset): The dataset to split.
        train_ratio (float): Ratio of training data.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    from torch.utils.data import random_split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def train_model(model, train_loader, num_epochs=100, learning_rate=0.005):
    """
    Trains the GCN model.

    Args:
        model (torch.nn.Module): The GCN model.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.

    Returns:
        torch.nn.Module: Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    return model

def evaluate_model(model, test_loader, label_map):
    """
    Evaluates the trained model.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for test data.
        label_map (dict): Mapping from labels to class names.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.tolist())
            y_pred.extend(pred.tolist())

    target_names = list(label_map.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

def save_model_and_mapping(model, condition_to_label, model_path, mapping_path, max_nodes, num_classes):
    """
    Saves the trained model and condition-to-label mapping to files.

    Args:
        model (torch.nn.Module): The trained model to save.
        condition_to_label (dict): The mapping from conditions to labels.
        model_path (str): The file path to save the model.
        mapping_path (str): The file path to save the mapping.
        max_nodes (int): Maximum number of nodes (model parameter).
        num_classes (int): Number of classes (model parameter).
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Add model parameters and label_map to the mapping
    condition_to_label['model_params'] = {
        'max_nodes': max_nodes,
        'num_classes': num_classes
    }
    condition_to_label['label_map'] = {'predator': 0, 'prey': 1}

    # Save the condition-to-label mapping
    with open(mapping_path, 'w') as f:
        json.dump(condition_to_label, f)
    print(f"Condition-to-label mapping saved to {mapping_path}")

def train_gnn_model(predator_csv, prey_csv, model_save_path="models/gnn_classifier.pth",
                    mapping_save_path="models/condition_to_label_mapping.json"):
    """
    Main function to train the GCN model.

    Args:
        predator_csv (str): Path to the predator data CSV file.
        prey_csv (str): Path to the prey data CSV file.
        model_save_path (str): Path to save the trained model.
        mapping_save_path (str): Path to save the condition-to-label mapping.

    Returns:
        torch.nn.Module: Trained GCN model.
    """
    data = load_data(predator_csv, prey_csv)
    graphs, labels, label_map, condition_to_label = prepare_graph_data(data)
    dataset, max_nodes = create_dataset(graphs, labels)
    resampled_dataset = balance_dataset(dataset, max_nodes)
    train_dataset, test_dataset = split_dataset(resampled_dataset, train_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    num_classes = len(label_map)
    model = GCN(num_features=max_nodes, num_classes=num_classes)
    model = train_model(model, train_loader, num_epochs=100)
    evaluate_model(model, test_loader, label_map)
    save_model_and_mapping(model, condition_to_label, model_save_path, mapping_save_path, max_nodes, num_classes)
    return model

if __name__ == "__main__":
    # Paths to your CSV files
    predator_csv = 'data/predator_data.csv'
    prey_csv = 'data/prey_data.csv'
    # Paths to save the trained model and mapping
    model_save_path = 'models/gnn_classifier.pth'
    mapping_save_path = 'models/condition_to_label_mapping.json'
    # Train the model
    model = train_gnn_model(predator_csv, prey_csv, model_save_path=model_save_path,
                            mapping_save_path=mapping_save_path)
