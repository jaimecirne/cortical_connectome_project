# scripts/app.py

import streamlit as st
from pathlib import Path
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import os
import seaborn as sns
import numpy as np
import json
import community as community_louvain  # Certifique-se de que o pacote está instalado

        
# Import necessary components from train_gnn.py
from train_gnn import GCN, pad_features

# Functions imported from corresponding scripts
from generate_connectomes import generate_connectome_from_data
from torch_geometric.utils import to_networkx

# Import Captum methods
from captum.attr import Saliency, IntegratedGradients

# Define a color palette for species
species_palette = {
    'predator': '#FF0000',  # Red for Predator
    'prey': '#0000FF',      # Blue for Prey
}

def get_node_colors(G):
    return ['#1f78b4' for _ in G.nodes]  # Default color for nodes

# Function to load the trained GNN model
def load_model(model_path, num_features, num_classes):
    model = GCN(num_features=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def generate_cached_connectomes(data):
    """
    Generates connectomes from data and caches them.
    """
    return generate_connectome_from_data(
        data,
        display_directed=False,
        coherence_threshold=0.01,
        top_k=None,
        n_jobs=-1
    )
# Configuração dinâmica do Coherence Threshold
if 'coherence_threshold' not in st.session_state:
    st.session_state['coherence_threshold'] = 0.01  # Valor padrão

def visualize_graph(G, title="Graph", edge_masks=None, cluster_view=False, coherence_threshold=st.session_state['coherence_threshold']):
    """
    Visualizes a graph, optionally with edge importance masks and cluster visualization,
    while filtering edges based on a coherence threshold.

    Args:
        G: NetworkX graph to visualize.
        title: Title for the plot.
        edge_masks: Optional dictionary of method names to edge masks.
        cluster_view: If True, visualizes the graph as clusters.
        coherence_threshold: Threshold for edge weight filtering. Edges below this value will be removed.
    """

    if cluster_view:
        # Louvain clustering
        partition = community_louvain.best_partition(G)

        # Group nodes into clusters
        clusters_map = {}
        for node, community in partition.items():
            clusters_map.setdefault(community, []).append(node)

        # Display clusters
        cluster_legenda = "\n".join([f"Cluster {com}: {', '.join(map(str, nos))}" for com, nos in clusters_map.items()])
        st.text(f"Clusters Detected:\n{cluster_legenda}")

        # Create reduced (clustered) graph
        clustered_G = nx.Graph()
        for node, community in partition.items():
            clustered_G.add_node(community)

        for ch1, ch2, data in G.edges(data=True):
            com1 = partition[ch1]
            com2 = partition[ch2]
            weight = data.get('weight', 1)  # Default weight to 1 if not present
            if clustered_G.has_edge(com1, com2):
                clustered_G[com1][com2]['weight'] += weight
            else:
                clustered_G.add_edge(com1, com2, weight=weight)

        # Visualize the reduced graph
        pos = nx.spring_layout(clustered_G, seed=42)  # Adjusted layout for reduced graph
        weights = np.array(list(nx.get_edge_attributes(clustered_G, 'weight').values()))

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        if len(weights) > 0:
            norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
            edge_colors = plt.cm.viridis(norm(weights))
            nx.draw(clustered_G, pos, ax=ax, with_labels=True, node_size=500, node_color='lightblue',
                    font_size=10, edge_color=edge_colors, width=2, edge_cmap=plt.cm.viridis)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Edge Weight (Cluster Level)', rotation=270, labelpad=20)
        else:
            nx.draw(clustered_G, pos, ax=ax, with_labels=True, node_size=500, node_color='lightblue', font_size=10)

        ax.set_title(f"{title} - Clustered View")
        ax.set_axis_off()

        st.pyplot(plt.gcf())
        plt.close()

    else:
        # Standard visualization
        pos = nx.circular_layout(G, scale=10)
        node_sizes = 300  # Fixed size
        node_colors = get_node_colors(G)

        if edge_masks is None:

            # Remove edges with weight below the coherence_threshold
            edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data.get('weight', 0) < coherence_threshold]
            G.remove_edges_from(edges_to_remove)

            # Verifica se o grafo ainda possui arestas
            if len(G.edges) == 0:
                st.warning(f"The graph '{title}' has no edges after applying the coherence threshold ({coherence_threshold}). It will be discarded.")
                return  # Descartar o grafo

            # Adjust node labels if the smallest node is 0
            if min(G.nodes) == 0:
                mapping = {node: node + 1 for node in G.nodes}  # Create mapping of 0->1, 1->2, ...
                G = nx.relabel_nodes(G, mapping)  # Apply the mapping
                st.info("Node labels were adjusted to start from 1.")

            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))

            if len(weights) > 0:
                norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
                edge_colors = plt.cm.viridis(norm(weights))
                nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors,
                        font_size=10, edge_color=edge_colors, width=2, edge_cmap=plt.cm.viridis)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Edge Weight (Normalized Coherence)', rotation=270, labelpad=20)
            else:
                nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=10)

            ax.set_title(title)
            ax.set_axis_off()

            st.pyplot(plt.gcf())
            plt.close()
        else:
            # Visualization with explanations
            for method_name, edge_mask in edge_masks.items():
                fig, ax = plt.subplots(figsize=(10, 8))

                for i, (u, v) in enumerate(G.edges()):
                    G[u][v]['importance'] = edge_mask[i]

                edge_colors = [G[u][v]['importance'] for u, v in G.edges()]
                edge_colors = np.array(edge_colors)
                if edge_colors.max() > 0:
                    edge_colors = (edge_colors - edge_colors.min()) / (edge_colors.max() - edge_colors.min())

                cmap = plt.cm.Reds if method_name == 'Integrated Gradients' else plt.cm.Blues
                nx.draw(
                    G,
                    pos,
                    ax=ax,
                    with_labels=True,
                    node_size=node_sizes,
                    node_color=node_colors,
                    font_size=10,
                    edge_color=edge_colors,
                    edge_cmap=cmap,
                    width=2
                )

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=edge_colors.min(), vmax=edge_colors.max()))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('Edge Importance', rotation=270, labelpad=20)
                ax.set_title(f"{title} {method_name}")
                ax.axis('off')

                st.pyplot(fig)
                plt.close(fig)

def convert_graphs_to_data_list(graphs, max_nodes):
    """
    Converts a list of NetworkX graphs to a list of torch_geometric.data.Data objects.
    """
    padded_graphs = pad_features(graphs, max_nodes)
    data_list = []
    for data in padded_graphs:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)  # Single graph batch
        data_list.append(data)
    return data_list

def explain_predictions(model, data_loader, graphs_info, device, method='saliency'):
    """
    Explains the predictions made by the model on the provided data using Captum.

    Args:
        model: The trained GNN model.
        data_loader: DataLoader containing the graphs to be explained.
        graphs_info: List of dictionaries with metadata of the graphs.
        device: The device to perform computations on (CPU or GPU).
        method: Explanation method, either 'saliency' or 'ig' (IntegratedGradients).
    """
    st.write('Generating explanations for the predictions using Captum...')

    # Define model_forward function that accepts edge_mask
    def model_forward(edge_mask, data):
        data = data.to(device)
        edge_weight = edge_mask.to(device)
        out = model(data, edge_weight=edge_weight)
        return out

    # Initialize explainers for both methods
    explainer_ig = IntegratedGradients(model_forward)
    explainer_saliency = Saliency(model_forward)

    for idx, data in enumerate(data_loader):
        data = data.to(device)
        graph_info = graphs_info[idx]

        with torch.no_grad():
            pred = model(data).argmax(dim=1).item()

        # Prepare the input for explainer
        input_mask = torch.ones(data.edge_index.shape[1], requires_grad=True, device=device)

        # Compute attribution with IG
        mask_ig = explainer_ig.attribute(
            input_mask,
            target=pred,
            additional_forward_args=(data,),
            internal_batch_size=data.edge_index.shape[1]
        )
        edge_mask_ig = mask_ig.detach().cpu().numpy()
        edge_mask_ig = np.abs(edge_mask_ig)
        if edge_mask_ig.max() > 0:
            edge_mask_ig = edge_mask_ig / edge_mask_ig.max()

        # Compute attribution with Saliency
        mask_saliency = explainer_saliency.attribute(
            input_mask,
            target=pred,
            additional_forward_args=(data,)
        )
        edge_mask_saliency = mask_saliency.detach().cpu().numpy()
        edge_mask_saliency = np.abs(edge_mask_saliency)
        if edge_mask_saliency.max() > 0:
            edge_mask_saliency = edge_mask_saliency / edge_mask_saliency.max()

        # Prepare the graph
        G = to_networkx(data, to_undirected=True)

        # Title for the plot
        title = f"Prediction Explanation - Session: {graph_info['session']}, " \
                f"Condition: {graph_info['condition']}, Window: {graph_info['window']}, " \
                f"Prediction: {pred}"

        # Call visualize_graph with edge masks
        visualize_graph(
            G=G,
            title=title,
            edge_masks={'Integrated Gradients': edge_mask_ig, 'Saliency': edge_mask_saliency}            
        )


def main():
    st.title("Comparative Cortical Mesoconnectome Analysis Based on Electrophysiological Data: A Study of different species")
    
    # Banner English buttons
    def read_markdown_file(markdown_file):
        return Path(markdown_file).read_text(encoding="utf-8")

    if st.sidebar.button("En"):
        intro_markdown = read_markdown_file("README.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)
    
    if st.sidebar.button("Pt"):
        intro_markdown = read_markdown_file("README_PT.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)


    st.sidebar.title("Menu")
    uploaded_file = st.sidebar.file_uploader("Upload data file", type=["csv"])

    if 'cluster_view' not in st.session_state:
        st.session_state['cluster_view'] = False


    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.write("### Data Preview")
        st.write(data.head())

        # Generate connectomes
        with st.spinner('Generating connectomes...'):
            connectomes = generate_cached_connectomes(data)
        st.success("Connectomes generated!")
        
        st.sidebar.markdown("### Coherence Threshold")
        st.session_state['coherence_threshold'] = st.sidebar.slider(
            "Adjust the Coherence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['coherence_threshold'],
            step=0.01,
            help="Edges with weights below this threshold will be removed from the graph."
        )

            
        # Main interaction buttons
        if st.sidebar.button("Visualize Graphs"):
            st.write("### Connectome Graphs")
            # Adiciona o checkbox para alternar entre visualizações
            cluster_view = st.session_state['cluster_view']
            for session, condition_dict in connectomes.items():
                for condition, windows in condition_dict.items():
                    for window, (G, cluster_legend) in windows.items():
                        if G.number_of_nodes() == 0:
                            continue
                        title = f"Session: {session}, Condition: {condition}, Window: {window}"
                        visualize_graph(G, title=title, cluster_view=False)

        if st.sidebar.button("Graphs in Cluster Visualization"):
            st.write("### Connectome Graphs")
            # Adiciona o checkbox para alternar entre visualizações
            cluster_view = st.session_state['cluster_view']
            for session, condition_dict in connectomes.items():
                for condition, windows in condition_dict.items():
                    for window, (G, cluster_legend) in windows.items():
                        if G.number_of_nodes() == 0:
                            continue
                        title = f"Session: {session}, Condition: {condition}, Window: {window}"
                        visualize_graph(G, title=title, cluster_view=True)
    

        if st.sidebar.button("Classify Graphs"):
            st.write("### Graph Classification")
            # Load the trained model
            model_path = "models/gnn_classifier.pth"
            mapping_path = "models/condition_to_label_mapping.json"

            # Load the condition-to-label mapping
            with open(mapping_path, 'r') as f:
                condition_to_label = json.load(f)

            # Retrieve model parameters
            model_params = condition_to_label.get('model_params', {})
            max_nodes = model_params.get('max_nodes', None)
            num_classes = model_params.get('num_classes', None)

            if max_nodes is None or num_classes is None:
                st.error("Model parameters are not available in the mapping file.")
                return

            model = load_model(model_path, num_features=max_nodes, num_classes=num_classes)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            graphs = []
            graphs_info = []

            # Extract graphs from connectomes
            for session, condition_dict in connectomes.items():
                for condition, windows in condition_dict.items():
                    for window, (G, cluster_legend) in windows.items():
                        if G.number_of_nodes() == 0:
                            continue
                        # Verifica se o grafo ainda possui arestas
                        if len(G.edges) == 0:
                            continue  # Descartar o grafo
                        graphs.append(G)
                        graphs_info.append({
                            'session': session,
                            'condition': condition,
                            'window': window
                        })

            if len(graphs) == 0:
                st.warning("No valid graphs found for classification.")
                return

            # Convert graphs to Data objects
            data_list = convert_graphs_to_data_list(graphs, max_nodes)

            # Create DataLoader
            from torch_geometric.loader import DataLoader
            data_loader = DataLoader(data_list, batch_size=1, shuffle=False)

            predictions = []
            embeddings = []

            st.write('Making predictions...')
            model.eval()
            with torch.no_grad():
                for data_object in data_loader:
                    data_object = data_object.to(device)
                    out = model(data_object)
                    pred = out.argmax(dim=1).item()
                    predictions.append(pred)
                    # Obtain the embedding before the final layer
                    embedding = model.get_embedding(data_object)
                    embeddings.append(embedding.cpu().numpy())

            # Map predicted labels to species names
            label_map = {v: k for k, v in condition_to_label.get('label_map', {'predator': 0, 'prey': 1}).items()}
            predicted_species = [label_map.get(pred, 'Unknown') for pred in predictions]

            # Create a DataFrame with predictions
            df_predictions = pd.DataFrame(graphs_info)
            df_predictions['Prediction'] = predictions
            df_predictions['Predicted Species'] = predicted_species

            st.write('Prediction Results:')
            st.write(df_predictions)

            # Save embeddings and labels for later use
            st.session_state['embeddings'] = embeddings
            st.session_state['predictions'] = predictions
            st.session_state['predicted_species'] = predicted_species

            # Save variables for explanation
            st.session_state['model'] = model
            st.session_state['data_loader'] = data_loader
            st.session_state['graphs_info'] = graphs_info
            st.session_state['device'] = device

        if st.sidebar.button("Visualize t-SNE"):
            st.write("### t-SNE Visualization")

            # Ensure embeddings and labels are available
            if 'embeddings' in st.session_state and 'predictions' in st.session_state:
                embeddings = st.session_state['embeddings']
                predictions = st.session_state['predictions']
                predicted_species = st.session_state['predicted_species']

                embeddings_array = np.vstack(embeddings)
                labels_array = np.array(predictions)

                df_tsne = visualize_tsne(embeddings_array, predicted_species)
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("Please perform graph classification first.")

        if st.sidebar.button("Explain Predictions"):
            st.write("### Prediction Explanation")

            if 'model' in st.session_state and 'data_loader' in st.session_state and 'graphs_info' in st.session_state and 'device' in st.session_state:
                model = st.session_state['model']
                data_loader = st.session_state['data_loader']
                graphs_info = st.session_state['graphs_info']
                device = st.session_state['device']

                explain_predictions(model, data_loader, graphs_info, device)
            else:
                st.warning("Please perform graph classification first.")

    else:
        st.write('Please upload a CSV file to continue.')

def visualize_tsne(embeddings_array, predicted_species):
    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)

    df_tsne = pd.DataFrame({
        'X': embeddings_2d[:, 0],
        'Y': embeddings_2d[:, 1],
        'Predicted Species': predicted_species
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='X', y='Y', hue='Predicted Species', palette=species_palette)
    plt.title('t-SNE Visualization of Graph Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return df_tsne

if __name__ == '__main__':
    main()
