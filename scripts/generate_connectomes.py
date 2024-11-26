# scripts/generate_connectomes.py

import pandas as pd
import numpy as np
import networkx as nx
import os

def generate_connectome_from_data(df, display_directed=False, use_connection_count=False, use_clustering=False):
    """
    Generates connectomes from the given dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing connectivity data.
        display_directed (bool): Whether to create directed graphs.
        use_connection_count (bool): If True, edge weights are based on connection count; otherwise, coherence.
        use_clustering (bool): If True, apply clustering to graphs.

    Returns:
        dict: Nested dictionary with connectomes structured as:
              connectomes[session][condition][window] = (graph, cluster_legend)
    """
    # Preprocess the DataFrame
    df = df.replace(',', '.', regex=True)
    df = df.replace('.', np.nan)
    numeric_columns = [
        'LoGammaCoherenceWin0Spike', 'LoGammaCoherenceSignifWin0Spike',
        'LoGammaCoherenceWin1Spike', 'LoGammaCoherenceSignifWin1Spike',
        'LoGammaCoherenceWin2Spike', 'LoGammaCoherenceSignifWin2Spike'
    ]
    df[numeric_columns] = df[numeric_columns].astype(float).fillna(df[numeric_columns].mean())

    # Unique sessions, conditions, and windows
    sessions = list(df['Session'].unique()) + ['all']
    conditions = [str(x) for x in df['Condition'].unique()] + ['all']
    windows = ['Win0', 'Win1', 'Win2', 'all']

    # Initialize the connectomes dictionary
    connectomes = {}

    # Generate graphs for each session, condition, and window
    for session in sessions:
        connectomes[session] = {}
        for condition in conditions:
            connectomes[session][condition] = {}
            for window in windows:
                # Filter data for current session and condition
                filtered_data = df if session == 'all' else df[df['Session'] == session]
                filtered_data = filtered_data if condition == 'all' else filtered_data[filtered_data['Condition'] == condition]

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

                    for (ch1, ch2), coherence in avg_coherence.items():
                        normalized_coherence = coherence / max_avg_coherence if max_avg_coherence > 0 else 0
                        G.add_edge(ch1, ch2, weight=normalized_coherence)

                # Apply clustering if enabled
                cluster_legend = None
                if use_clustering:
                    partition = community_louvain.best_partition(G)
                    clusters_map = {}
                    for node, community in partition.items():
                        clusters_map.setdefault(community, []).append(node)

                    cluster_legend = "\n".join([f"Cluster {c}: {', '.join(map(str, nodes))}" for c, nodes in clusters_map.items()])
                    clustered_G = nx.Graph()
                    for node, community in partition.items():
                        clustered_G.add_node(community)

                    for ch1, ch2, data in G.edges(data=True):
                        community1 = partition[ch1]
                        community2 = partition[ch2]
                        weight = data['weight']
                        if clustered_G.has_edge(community1, community2):
                            clustered_G[community1][community2]['weight'] += weight
                        else:
                            clustered_G.add_edge(community1, community2, weight=weight)

                    G = clustered_G

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
    csv_file = 'data/c5607a01.csv'  
    data = pd.read_csv(csv_file,  delimiter=',')
    
    # Substituir vírgulas por pontos nos valores numéricos
    data = data.replace(',', '.', regex=True)

    # Remover valores inválidos (substituindo por NaN)
    data = data.replace('.', np.nan)

    # Verificando valores não numéricos e substituindo por NaN
    cols_to_convert = data.columns.difference(['Record', 'Session', 'Condition'])

    for col in cols_to_convert:
        # Substituir '.' por NaN
        data[col] = data[col].replace('.', np.nan)
        # Converter para float, substituindo qualquer valor não conversível por NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remover colunas completamente vazias após a conversão
    data.dropna(axis=1, how='all', inplace=True)

    # Remover linhas completamente vazias após a conversão
    data.dropna(axis=0, how='all', inplace=True)

    # Verificar se o DataFrame ainda tem dados suficientes
    if data.empty:
        raise ValueError("O DataFrame está vazio após a limpeza. Verifique os dados de entrada.")

    # Divisão em variáveis preditoras (X) e variável alvo (y)
    if 'Condition' not in data.columns:
        raise ValueError("A coluna 'Condition' não está presente no DataFrame. Verifique o arquivo de entrada.")

    connectomes = generate_connectome_from_data(data, use_clustering=False)
    
    # Salvar conectomas
    output_dir = 'outputs/predator_connectomes'  
    save_connectomes(connectomes, output_dir)

    csv_file = 'data/c5103a01MUA.csv'  

    data = pd.read_csv(csv_file,  delimiter=',')
    
    # Substituir vírgulas por pontos nos valores numéricos
    data = data.replace(',', '.', regex=True)

    # Remover valores inválidos (substituindo por NaN)
    data = data.replace('.', np.nan)

    # Verificando valores não numéricos e substituindo por NaN
    cols_to_convert = data.columns.difference(['Record', 'Session', 'Condition'])

    for col in cols_to_convert:
        # Substituir '.' por NaN
        data[col] = data[col].replace('.', np.nan)
        # Converter para float, substituindo qualquer valor não conversível por NaN
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Remover colunas completamente vazias após a conversão
    data.dropna(axis=1, how='all', inplace=True)

    # Remover linhas completamente vazias após a conversão
    data.dropna(axis=0, how='all', inplace=True)

    # Verificar se o DataFrame ainda tem dados suficientes
    if data.empty:
        raise ValueError("O DataFrame está vazio após a limpeza. Verifique os dados de entrada.")

    # Divisão em variáveis preditoras (X) e variável alvo (y)
    if 'Condition' not in data.columns:
        raise ValueError("A coluna 'Condition' não está presente no DataFrame. Verifique o arquivo de entrada.")

    # Gerar conectomas
    connectomes = generate_connectome_from_data(data, use_clustering=False)
    
    # Salvar conectomas
    output_dir = 'outputs/prey_connectomes'  
    save_connectomes(connectomes, output_dir)