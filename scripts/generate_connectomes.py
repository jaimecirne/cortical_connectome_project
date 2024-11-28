# scripts/generate_connectomes.py

import pandas as pd
import numpy as np
import networkx as nx
import os

# Funções importadas dos scripts correspondentes
from utils import preprocess_dataFrame

def generate_connectome_from_data(df, display_directed=True, use_connection_count=False, use_clustering=False, coherence_threshold=None, top_k=None):
    """
    Generates connectomes from the given dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing connectivity data.
        display_directed (bool): Whether to create directed graphs.
        use_connection_count (bool): If True, edge weights are based on connection count; otherwise, coherence.
        use_clustering (bool): If True, apply clustering to graphs.
        coherence_threshold (float): Threshold for coherence values (between 0 and 1).
        top_k (int): Number of top edges to include based on coherence.

    Returns:
        dict: Nested dictionary with connectomes structured as:
              connectomes[session][condition][window] = (graph, cluster_legend)
    """
    # Preprocess the DataFrame
    sessions, conditions, windows, df = preprocess_dataFrame(df)

    # Initialize the connectomes dictionary
    connectomes = {}

    # Generate graphs for each session, condition, and window
    for session in sessions:
        connectomes[session] = {}
        for condition in conditions:
            connectomes[session][condition] = {}
            for window in windows:
                # Filter data for current session and condition
                filtered_data = df[df['Session'] == session]
                filtered_data = df[df['Condition'] == condition]

                # Check if filtered_data is empty
                if filtered_data.empty:
                    continue  # Skip if no data

                # Determine the coherence column based on the window
                coherence_col = None
                if window == 'Win0':
                    coherence_col = 'LoGammaCoherenceSignifWin0Spike'
                elif window == 'Win1':
                    coherence_col = 'LoGammaCoherenceSignifWin1Spike'
                elif window == 'Win2':
                    coherence_col = 'LoGammaCoherenceSignifWin2Spike'

                # Initialize the graph
                G = nx.DiGraph() if display_directed else nx.Graph()
                coherence_sums = {}
                coherence_counts = {}

                # Add edges to the graph
                for _, row in filtered_data.iterrows():
                    ch1, ch2 = row['Ch1'], row['Ch2']
                    coherence = row[coherence_col] if coherence_col else 0

                    if use_connection_count:
                        if G.has_edge(ch1, ch2):
                            G[ch1][ch2]['weight'] += 1
                        else:
                            G.add_edge(ch1, ch2, weight=1)
                    else:
                        coherence_sums[(ch1, ch2)] = coherence_sums.get((ch1, ch2), 0) + coherence
                        coherence_counts[(ch1, ch2)] = coherence_counts.get((ch1, ch2), 0) + 1

                # Add edges with normalized coherence weights
                if not use_connection_count:
                    max_avg_coherence = 0
                    avg_coherence = {}
                    for (ch1, ch2), sum_coherence in coherence_sums.items():
                        avg_coherence[(ch1, ch2)] = sum_coherence / coherence_counts[(ch1, ch2)]
                        max_avg_coherence = max(max_avg_coherence, avg_coherence[(ch1, ch2)])

                    if max_avg_coherence == 0:
                        print(f"max_avg_coherence é zero para Session={session}, Condition={condition}, Window={window}. Pulando esta iteração.")
                        continue  # Pula para a próxima iteração, pois não há dados válidos

                    # Apply coherence threshold
                    if coherence_threshold is not None:
                        avg_coherence = {edge: coherence for edge, coherence in avg_coherence.items() if coherence / max_avg_coherence >= coherence_threshold}

                    # Sort edges by coherence
                    sorted_edges = sorted(avg_coherence.items(), key=lambda x: x[1], reverse=True)

                    # Select top K edges
                    if top_k is not None:
                        sorted_edges = sorted_edges[:top_k]

                    for (ch1, ch2), coherence in sorted_edges:
                        normalized_coherence = coherence / max_avg_coherence if max_avg_coherence > 0 else 0
                        G.add_edge(ch1, ch2, weight=normalized_coherence)

                # Apply clustering if enabled
                cluster_legend = None
                if use_clustering and G.number_of_nodes() > 0:
                    # [Clustering code remains the same]
                    pass  # Omitido para brevidade

                # Store the graph and cluster legend
                connectomes[session][condition][window] = (G, cluster_legend)

    return connectomes

def save_connectomes(connectomes, output_dir):
    """
    Saves connectomes in GraphML and adjacency matrix formats and computes key graph metrics.

    Args:
        connectomes (dict): Nested dictionary of connectomes (session -> condition -> window).
        output_dir (str): Directory where the files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            for window, (G, cluster_legend) in windows.items():
                # Define filenames
                session_condition_window = f"session_{session}_condition_{condition}_window_{window}"
                graph_file = os.path.join(output_dir, f'{session_condition_window}.graphml')
                adj_file = os.path.join(output_dir, f'{session_condition_window}_adjacency.npy')
                centrality_file = os.path.join(output_dir, f'{session_condition_window}_centrality.csv')
                cluster_file = os.path.join(output_dir, f'{session_condition_window}_clusters.txt')

                # Save GraphML
                nx.write_graphml(G, graph_file)

                # Save adjacency matrix
                adj_matrix = nx.to_numpy_array(G)
                np.save(adj_file, adj_matrix)

                # Compute and save metrics
                avg_degree = sum(dict(G.degree()).values()) / float(len(G)) if len(G) > 0 else 0
                density = nx.density(G)
                print(f'{session_condition_window}: Avg Degree = {avg_degree:.2f}, Density = {density:.4f}')

                centrality = nx.degree_centrality(G)
                pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality']).to_csv(centrality_file)

                # Save cluster legend if available
                if cluster_legend:
                    with open(cluster_file, 'w') as f:
                        f.write(cluster_legend)

if __name__ == "__main__":
    # Exemplo de uso:
    # Carregar o arquivo CSV
    csv_file = 'data/predator_data.csv'  
    data = pd.read_csv(csv_file,  delimiter=',')
    
    connectomes = generate_connectome_from_data(data, use_clustering=False)
    
    # Salvar conectomas
    output_dir = 'outputs/predator_connectomes'  
    save_connectomes(connectomes, output_dir)

    csv_file = 'data/prey_data.csv'  

    # Gerar conectomas
    connectomes = generate_connectome_from_data(data, use_clustering=False)
    
    # Salvar conectomas
    output_dir = 'outputs/prey_connectomes'  
    save_connectomes(connectomes, output_dir)