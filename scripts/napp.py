# scripts/app.py

import streamlit as st
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
        use_connection_count=False,
        coherence_threshold=0.1,
        top_k=None,
        n_jobs=-1
    )

def visualize_graph(G, title="Graph", edge_masks=None):
    """
    Visualizes a graph, optionally with edge importance masks.

    Args:
        G: NetworkX graph to visualize.
        title: Title for the plot.
        edge_masks: Optional dictionary of method names to edge masks.
    """
    pos = nx.circular_layout(G, scale=10)
    node_sizes = 300  # Fixed size, adjust as needed
    node_colors = get_node_colors(G)
    #plt.figure(figsize=(10, 8))
    #ax = plt.gca()

    if edge_masks is None:
        # Original visualization
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Define layout and edge weights
        #pos = nx.circular_layout(G, scale=10)
        weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
        #node_sizes = 300  # Fixed size, adjust as needed
        #node_colors = get_node_colors(G)

        # Draw the graph
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


        # Set the title
        ax.set_title(title)
        ax.set_axis_off()

        # Show the graph
        st.pyplot(plt.gcf())
        plt.close()

    else:
        # Visualization with explanations
        num_methods = len(edge_masks)
        fig, axs = plt.subplots(1, num_methods, figsize=(10 * num_methods, 8))

        if num_methods == 1:
            axs = [axs]

        #pos = nx.spring_layout(G, seed=42)
        #node_sizes = 300  # Fixed size
        #node_colors = get_node_colors(G)

        for ax, (method_name, edge_mask) in zip(axs, edge_masks.items()):
            # Assign importance to edges
            for i, (u, v) in enumerate(G.edges()):
                G[u][v]['importance'] = edge_mask[i]

            # Configure edge colors based on importance
            edge_colors = [G[u][v]['importance'] for u, v in G.edges()]
            edge_colors = np.array(edge_colors)
            if edge_colors.max() > 0:
                edge_colors = (edge_colors - edge_colors.min()) / (edge_colors.max() - edge_colors.min())

            # Draw the graph
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
            ax.set_title(method_name)
            ax.axis('off')

        # Set the overall title
        fig.suptitle(title)

        # Adjust layout
        plt.tight_layout()

        # Display the plot in Streamlit
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
    st.title("Connectome Analysis with GNN")
    st.sidebar.title("Menu")
    uploaded_file = st.sidebar.file_uploader("Upload data file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        st.write("### Data Preview")
        st.write(data.head())

        # Generate connectomes
        with st.spinner('Generating connectomes...'):
            connectomes = generate_cached_connectomes(data)
        st.success("Connectomes generated!")

        # Main interaction buttons
        if st.sidebar.button("Visualize Graphs"):
            st.write("### Connectome Graphs")
            for session, condition_dict in connectomes.items():
                for condition, windows in condition_dict.items():
                    for window, (G, cluster_legend) in windows.items():
                        if G.number_of_nodes() == 0:
                            continue
                        title = f"Session: {session}, Condition: {condition}, Window: {window}"
                        visualize_graph(G, title=title)

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
