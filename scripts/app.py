# scripts/app.py

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import numpy as np
import os

# Funções importadas dos scripts correspondentes
from generate_connectomes import generate_connectome_from_data
from train_gnn import GNNClassifier, create_data_objects
from explain_predictions import explain_prediction, visualize_attributions

# def generate_connectome_from_data(df, display_directed=False, use_connection_count=False, use_clustering=False):
#     """
#     Generates connectomes from the given dataset.

#     Args:
#         df (pd.DataFrame): Input DataFrame containing connectivity data.
#         display_directed (bool): Whether to create directed graphs.
#         use_connection_count (bool): If True, edge weights are based on connection count; otherwise, coherence.
#         use_clustering (bool): If True, apply clustering to graphs.

#     Returns:
#         dict: Nested dictionary with connectomes structured as:
#               connectomes[session][condition][window] = (graph, cluster_legend)
#     """
#     # Preprocess the DataFrame
#     df = df.replace(',', '.', regex=True)
#     df = df.replace('.', np.nan)
#     numeric_columns = [
#         'LoGammaCoherenceWin0Spike', 'LoGammaCoherenceSignifWin0Spike',
#         'LoGammaCoherenceWin1Spike', 'LoGammaCoherenceSignifWin1Spike',
#         'LoGammaCoherenceWin2Spike', 'LoGammaCoherenceSignifWin2Spike'
#     ]
#     df[numeric_columns] = df[numeric_columns].astype(float).fillna(df[numeric_columns].mean())

#     # Unique sessions, conditions, and windows
#     sessions = list(df['Session'].unique()) + ['all']
#     conditions = [str(x) for x in df['Condition'].unique()] + ['all']
#     windows = ['Win0', 'Win1', 'Win2', 'all']

#     # Initialize the connectomes dictionary
#     connectomes = {}

#     # Generate graphs for each session, condition, and window
#     for session in sessions:
#         connectomes[session] = {}
#         for condition in conditions:
#             connectomes[session][condition] = {}
#             for window in windows:
#                 # Filter data for current session and condition
#                 filtered_data = df if session == 'all' else df[df['Session'] == session]
#                 filtered_data = filtered_data if condition == 'all' else filtered_data[filtered_data['Condition'] == condition]

#                 # Determine the coherence column based on the window
#                 coherence_col = None
#                 if window == 'Win0':
#                     coherence_col = 'LoGammaCoherenceSignifWin0Spike'
#                 elif window == 'Win1':
#                     coherence_col = 'LoGammaCoherenceSignifWin1Spike'
#                 elif window == 'Win2':
#                     coherence_col = 'LoGammaCoherenceSignifWin2Spike'

#                 # Initialize the graph
#                 G = nx.DiGraph() if display_directed else nx.Graph()
#                 coherence_sums = {}
#                 coherence_counts = {}

#                 # Add edges to the graph
#                 for _, row in filtered_data.iterrows():
#                     ch1, ch2 = row['Ch1'], row['Ch2']
#                     coherence = row[coherence_col] if coherence_col else 0

#                     if use_connection_count:
#                         if G.has_edge(ch1, ch2):
#                             G[ch1][ch2]['weight'] += 1
#                         else:
#                             G.add_edge(ch1, ch2, weight=1)
#                     else:
#                         coherence_sums[(ch1, ch2)] = coherence_sums.get((ch1, ch2), 0) + coherence
#                         coherence_counts[(ch1, ch2)] = coherence_counts.get((ch1, ch2), 0) + 1

#                 # Add edges with normalized coherence weights
#                 if not use_connection_count:
#                     max_avg_coherence = 0
#                     avg_coherence = {}
#                     for (ch1, ch2), sum_coherence in coherence_sums.items():
#                         avg_coherence[(ch1, ch2)] = sum_coherence / coherence_counts[(ch1, ch2)]
#                         max_avg_coherence = max(max_avg_coherence, avg_coherence[(ch1, ch2)])

#                     for (ch1, ch2), coherence in avg_coherence.items():
#                         normalized_coherence = coherence / max_avg_coherence if max_avg_coherence > 0 else 0
#                         G.add_edge(ch1, ch2, weight=normalized_coherence)

#                 # Apply clustering if enabled
#                 cluster_legend = None
#                 if use_clustering:
#                     partition = community_louvain.best_partition(G)
#                     clusters_map = {}
#                     for node, community in partition.items():
#                         clusters_map.setdefault(community, []).append(node)

#                     cluster_legend = "\n".join([f"Cluster {c}: {', '.join(map(str, nodes))}" for c, nodes in clusters_map.items()])
#                     clustered_G = nx.Graph()
#                     for node, community in partition.items():
#                         clustered_G.add_node(community)

#                     for ch1, ch2, data in G.edges(data=True):
#                         community1 = partition[ch1]
#                         community2 = partition[ch2]
#                         weight = data['weight']
#                         if clustered_G.has_edge(community1, community2):
#                             clustered_G[community1][community2]['weight'] += weight
#                         else:
#                             clustered_G.add_edge(community1, community2, weight=weight)

#                     G = clustered_G

#                 # Store the graph and cluster legend
#                 connectomes[session][condition][window] = (G, cluster_legend)

#     return connectomes

def make_prediction(model, data_object, device):
    model.eval()
    data_object = data_object.to(device)
    with torch.no_grad():
        out = model(data_object)
        pred = out.argmax(dim=1).item()
        return pred, out.cpu().numpy()  # Retorna pred e embedding


def visualize_all_graphs(connectomes, save_to_file=False, output_dir="graphs"):
    """
    Visualizes all graphs in the `connectomes` structure and optionally saves them.

    Args:
        connectomes (dict): Nested dictionary of graphs: connectomes[session][condition][window].
        save_to_file (bool): If True, saves each graph as an image in the output directory.
        output_dir (str): Directory to save graph images if `save_to_file` is True.
    """
    
    if save_to_file and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define node colors (example palette)
    color_palette = [
        '#FF8C00', '#FF0000', '#00BFFF', '#4169E1', '#32CD32', '#FFD700', '#8A2BE2', '#00FF7F',
        '#FF69B4', '#8B4513', '#DC143C', '#808000', '#5F9EA0', '#BA55D3', '#FF4500', '#3CB371',
        '#D9FF00', '#FF9900', '#0026FF', '#7941E1', '#32CD8F', '#8EFF00', '#E22BCC', '#00E6FF',
        '#FF7869', '#898B13', '#DC6414', '#338000', '#5F77A0', '#D355A0', '#FFDE00', '#3CAEB3',
        '#D9FF00', '#FF9900', '#0026FF', '#7941E1', '#32CD8F', '#8EFF00', '#E22BCC', '#00E6FF',
        '#FF7869', '#898B13', '#DC6414', '#338000', '#5F77A0', '#D355A0', '#FFDE00', '#3CAEB3',
        '#D9FF00', '#FF9900', '#0026FF', '#7941E1', '#32CD8F', '#8EFF00', '#E22BCC', '#00E6FF',
        '#FF7869', '#898B13', '#DC6414', '#338000', '#5F77A0', '#D355A0', '#FFDE00', '#3CAEB3'

    ]
    color_node_ref = {str(i): color for i, color in enumerate(color_palette, start=1)}

    def get_node_colors(G):
        return [color_node_ref.get(str(node), '#0000FF') for node in G.nodes]

    graph_count = 1  # To keep track of graph numbering

    for session, conditions in connectomes.items():
        for condition, windows in conditions.items():
            for window, (G, cluster_legend) in windows.items():
                if G.number_of_nodes() == 0:
                    continue
                # Prepare the figure
                plt.figure(figsize=(10, 8))
                ax = plt.gca()

                # Define layout and edge weights
                pos = nx.circular_layout(G, scale=10)
                weights = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
                node_sizes = 300  # Fixed size, adjust as needed
                node_colors = get_node_colors(G)

                # Draw the graph
                if len(weights) > 0:
                    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
                    edge_colors = plt.cm.Blues(norm(weights))
                    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors,
                            font_size=10, edge_color=edge_colors, width=2, edge_cmap=plt.cm.Blues)
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax)
                    cbar.set_label('Edge Weight (Normalized Coherence)', rotation=270, labelpad=20)
                else:
                    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors, font_size=10)

                # Add cluster legend if available
                if cluster_legend:
                    ax.text(0, -0.1, f"Clusters:\n{cluster_legend}", fontsize=10, va='top', ha='center', transform=ax.transAxes)

                # Set title
                ax.set_title(f"Connectome: Session={session}, Condition={condition}, Window={window}")
                ax.set_axis_off()

                # Save to file if required
                if save_to_file:
                    file_name = f"{output_dir}/connectome_{graph_count:03d}_session_{session}_condition_{condition}_window_{window}.png"
                    plt.savefig(file_name, bbox_inches="tight")
                    print(f"Graph saved to {file_name}")

                # Show the graph
                st.pyplot(plt.gcf())
                plt.close()
                graph_count += 1

    #st.pyplot(plt)
    #plt.close()

st.title('Análise de Conectomas e Classificação com GNN')

uploaded_file = st.file_uploader('Faça o upload dos dados CSV', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, delimiter=',')

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


    st.write('Pré-visualização dos Dados:')
    st.write(data.head())
    st.write('Gerando conectoma...')
    connectomes = generate_connectome_from_data(data)
    st.write('Grafo do Conectoma:')
    visualize_all_graphs(connectomes)
    data_objects, condition_to_label = create_data_objects(connectomes)
    label_to_condition = {v: k for k, v in condition_to_label.items()}
    # Determinar o número de features e classes
    num_node_features = data_objects[0].x.shape[1]
    num_classes = len(condition_to_label)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=num_node_features, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('models/gnn_classifier.pth'))
    
    predictions = []
    for data_object in data_objects:
        pred = make_prediction(model, data_object, device)
        predictions.append(pred)

    # # Criar um DataFrame com as predições
    # df = pd.DataFrame({
    #     'Grafo': range(len(predictions)),
    #     'Predição': predictions,
    #     'Condição Predita': [label_to_condition.get(pred, 'Desconhecida') for pred in predictions]
    # })

    # st.write(df)

    # prediction = make_prediction(model, data_objects, device)
    # st.write(f'Condição Predita: {prediction}')
    # st.write('Explicando a Predição:')
    attributions_list = []
    predictions = []

    for idx, data_object in enumerate(data_objects):
        edge_importance, pred = explain_prediction(model, data_object)
        attributions_list.append(edge_importance)
        predictions.append(pred)
        predicted_condition = label_to_condition.get(pred, 'Desconhecida')
        print(f"Grafo {idx}: Condição predita: {predicted_condition}")
        #visualize_attributions(connectomes, attributions_list, f'outputs/temp_attribution{predicted_condition}.png')

    st.write('Realizando predições...')
    predictions = []
    embeddings = []

    for data_object in data_objects:
        pred, embedding = make_prediction(model, data_object, device)
        predictions.append(pred)
        embeddings.append(embedding)

    # Convert embeddings to 2D array
    embeddings = np.vstack(embeddings)

    # Map predictions to condition names
    predicted_conditions = [label_to_condition.get(pred, 'Desconhecida') for pred in predictions]

    # Create a DataFrame with predictions
    df_predictions = pd.DataFrame({
        'Grafo': range(len(predictions)),
        'Predição': predictions,
        'Condição Predita': predicted_conditions
    })

    st.write('Resultados das Predições:')
    st.write(df_predictions)

    st.write('Visualizando as Atribuições:')
    visualize_attributions(data_object, edge_importance, condition_to_label)
    st.pyplot(plt.gcf())  # Exibe a figura atual no Streamlit
    plt.close()

    st.write('Visualização t-SNE:')
    # Apply t-SNE to embeddings
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a DataFrame for t-SNE results
    df_tsne = pd.DataFrame({
        'X': embeddings_2d[:, 0],
        'Y': embeddings_2d[:, 1],
        'Condição': predicted_conditions
    })

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x='X', y='Y', hue='Condição', palette='viridis')
    plt.title('t-SNE Visualization of Graph Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    
    st.pyplot(plt.gcf())
    plt.close()    

#    attributions = explain_prediction(model, data_objects)
 #   visualize_attributions(connectomes, attributions, 'outputs/temp_attribution.png')
  #  st.image('outputs/temp_attribution.png')
  #  st.write('Visualização t-SNE:')
  #  st.image('outputs/tsne_visualization.png')
else:
    st.write('Por favor, faça o upload de um arquivo CSV para continuar.')
