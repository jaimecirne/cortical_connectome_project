import pandas as pd
import networkx as nx
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, Dataset, Batch
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.loader import DataLoader
from generate_connectomes import generate_connectome_from_data


def pad_features(graphs, max_nodes):
    padded_graphs = []
    for idx, G in enumerate(graphs):
        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            print(f"Aviso: Grafo vazio encontrado (índice {idx}) e será ignorado.")
            continue

        x = torch.zeros(max_nodes, max_nodes, dtype=torch.float)
        x[:num_nodes, :num_nodes] = torch.eye(num_nodes, dtype=torch.float)

        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_index = edge_index[:, (edge_index < max_nodes).all(dim=0)]

        if edge_index.size(1) == 0:
            print(f"Aviso: Grafo {idx} sem arestas após padronização.")
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = global_max_pool(x, batch)  # Alteração para max pooling
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def classify_prey_predator_gnn(predator_csv, prey_csv):
    data_predator = pd.read_csv(predator_csv)
    data_prey = pd.read_csv(prey_csv)

    data_predator['especie'] = 'predator'
    data_prey['especie'] = 'prey'

    data = pd.concat([data_predator, data_prey], ignore_index=True)

    connectomes = generate_connectome_from_data(data, display_directed=False, use_connection_count=False,
                                                 coherence_threshold=0.2, top_k=10, n_jobs=4)

    session_labels = {session: 'predator' if session in data_predator['Session'].unique() else 'prey'
                      for session in data['Session'].unique()}
    label_map = {'predator': 0, 'prey': 1}

    graphs = []
    labels = []
    for session, session_data in connectomes.items():
        for condition, condition_data in session_data.items():
            for window, (G, _) in condition_data.items():
                if G is not None:
                    graphs.append(G)
                    labels.append(label_map[session_labels[session]])

    max_nodes = max(graph.number_of_nodes() for graph in graphs)
    padded_graphs = pad_features(graphs, max_nodes)

    dataset = GraphDataset(padded_graphs, labels)

    X = torch.stack([data.x.flatten() for data in dataset]).numpy()
    y = torch.tensor([data.y.item() for data in dataset]).numpy()
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_graphs = []
    for i in range(len(X_resampled)):
        original_index = i % len(dataset)
        edge_index = dataset[original_index].edge_index
        x_resampled = torch.tensor(X_resampled[i], dtype=torch.float32).reshape(max_nodes, -1)
        resampled_graphs.append(Data(x=x_resampled, edge_index=edge_index))

    resampled_dataset = GraphDataset(resampled_graphs, y_resampled.tolist())

    from torch.utils.data import random_split
    train_size = int(0.8 * len(resampled_dataset))
    test_size = len(resampled_dataset) - train_size
    train_dataset, test_dataset = random_split(resampled_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = GCN(num_features=max_nodes, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    model.train()
    for epoch in range(100):  # Aumento do número de épocas
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.tolist())
            y_pred.extend(pred.tolist())

    print(classification_report(y_true, y_pred, target_names=label_map.keys()))


if __name__ == "__main__":
    classify_prey_predator_gnn('data/predator_data.csv', 'data/prey_data.csv')
