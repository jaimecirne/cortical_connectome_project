# scripts/visualize_tsne.py

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_hidden_features(model, dataset):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = model.conv1(x, edge_index)
            x = torch.relu(x)
            x = model.conv2(x, edge_index)
            x = global_mean_pool(x, batch)
            features.append(x.cpu().numpy())
            labels.append(data.y.item())
    features = np.vstack(features)
    return features, labels

def visualize_tsne(features, labels, output_file):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Condições")
    plt.title('Visualização t-SNE das Features da Camada Oculta')
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=32, num_classes=12).to(device)
    model.load_state_dict(torch.load('../models/gnn_classifier.pth'))
    dataset = prepare_dataset('../outputs/predator_connectomes')
    features, labels = extract_hidden_features(model, dataset)
    visualize_tsne(features, labels, '../outputs/tsne_visualization.png')
