# scripts/explain_predictions.py

import torch
from captum.attr import IntegratedGradients
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def explain_prediction(model, data):
    model.eval()
    data = data.to(device)
    def model_forward(x, edge_index, batch):
        output = model.forward(Data(x=x, edge_index=edge_index, batch=batch))
        return output
    ig = IntegratedGradients(model_forward)
    attributions = ig.attribute(
        inputs=data.x.unsqueeze(0),
        additional_forward_args=(data.edge_index, data.batch),
        target=data.y.item()
    )
    attributions = attributions.squeeze(0)
    return attributions.detach().cpu().numpy()

def visualize_attributions(G, attributions, output_file):
    pos = nx.spring_layout(G)
    node_color = attributions
    plt.figure(figsize=(8, 6))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.viridis, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=32, num_classes=12).to(device)
    model.load_state_dict(torch.load('../models/gnn_classifier.pth'))
    dataset = prepare_dataset('../outputs/predator_connectomes')
    test_data = dataset[0]
    attributions = explain_prediction(model, test_data)
    graph_file = '../outputs/predator_connectomes/connectome_condition_1.graphml'
    G = nx.read_graphml(graph_file)
    visualize_attributions(G, attributions, '../outputs/attributions_condition_1.png')
