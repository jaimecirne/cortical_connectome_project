# scripts/generate_connectomes.py

import pandas as pd
import numpy as np
import networkx as nx
import os
import streamlit as st

def process_condition(data, condition_label, usar_clustering=False):
    # Filtrar os dados para a condição especificada
    condition_data = data[data['Condition'] == condition_label]
    
    if condition_data.empty:
        st.error(f"Não há dados para a condição '{condition_label}'.")
        return None, None

    # Selecionar apenas as colunas numéricas relevantes
    numeric_columns = condition_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.error("Não foram encontradas colunas numéricas nos dados.")
        return None, None

    channel_data = condition_data[numeric_columns]
    
    # Preencher valores NaN com a média da coluna
    channel_data = channel_data.apply(pd.to_numeric, errors='coerce')
    channel_data = channel_data.fillna(channel_data.mean())
    
    # Calcular a matriz de correlação
    adj_matrix = channel_data.corr().values
    
    # Verificar se a matriz de adjacência é válida
    if np.isnan(adj_matrix).any() or np.isinf(adj_matrix).any():
        st.error("A matriz de adjacência contém valores inválidos.")
        return None, None
    
    # Criar o grafo
    G = nx.from_numpy_array(adj_matrix)
    G = nx.relabel_nodes(G, dict(zip(range(len(numeric_columns)), numeric_columns)))
    
    return G, adj_matrix

def generate_connectomes(csv_file, output_dir):
    # Gera conectomas para cada condição no arquivo CSV fornecido.
    data = pd.read_csv(csv_file)
    if 'Condition' not in data.columns:
        raise ValueError("A coluna 'Condition' não foi encontrada nos dados.")
    conditions = data['Condition'].unique()
    os.makedirs(output_dir, exist_ok=True)
    for condition in conditions:
        G, adj_matrix = process_condition(data, condition)
        graph_file = os.path.join(output_dir, f'connectome_condition_{condition}.graphml')
        nx.write_graphml(G, graph_file)
        adj_file = os.path.join(output_dir, f'adjacency_condition_{condition}.npy')
        np.save(adj_file, adj_matrix)
        avg_degree = sum(dict(G.degree()).values()) / float(len(G))
        density = nx.density(G)
        print(f'Condição {condition}: Grau Médio = {avg_degree}, Densidade = {density}')
        centrality = nx.degree_centrality(G)
        centrality_file = os.path.join(output_dir, f'centrality_condition_{condition}.csv')
        pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality']).to_csv(centrality_file)

if __name__ == "__main__":
    # Geração de conectomas para os dados do predador
    generate_connectomes('../data/c5607a01.csv', '../outputs/predator_connectomes')
    # Geração de conectomas para os dados da presa
    generate_connectomes('../data/c5103a01MUAPairs_AllConditions.csv', '../outputs/prey_connectomes')
