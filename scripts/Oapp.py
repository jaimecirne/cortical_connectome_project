# scripts/app.py

import streamlit as st
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from train_gnn import GNNClassifier  # Importa o modelo treinado
from utils import preprocess_dataFrame  # Utilidade para preprocessamento
import os
import seaborn as sns
import numpy as np
from captum.attr import IntegratedGradients


# Funções importadas dos scripts correspondentes
from generate_connectomes import generate_connectome_from_data

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

# Função para carregar o modelo GNN treinado
def load_model(model_path, num_node_features, num_classes):
    model = GNNClassifier(num_node_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

@st.cache_resource
def generate_cached_connectomes(data):
    """
    Gera conectomas a partir dos dados e armazena em cache.
    
    Args:
        data (DataFrame): Dados carregados do arquivo.
    
    Returns:
        dict: Estrutura de conectomas.
    """
    return generate_connectome_from_data(data, display_directed=True, use_connection_count=False,
                                   coherence_threshold=0.1, top_k=None, n_jobs=-1)

def visualize_graph(G, session, condition,window):
    
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
        edge_colors = plt.cm.viridis(norm(weights))
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors,
                font_size=10, edge_color=edge_colors, width=2, edge_cmap=plt.cm.viridis)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
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

    # Show the graph
    st.pyplot(plt.gcf())
    plt.close()

def visualize_all_graphs(connectomes, save_to_file=True, edge_weight_threshold=0.5, output_dir="graphs"):
    """
    Visualizes all graphs in the `connectomes` structure and optionally saves them.

    Args:
        connectomes (dict): Nested dictionary of graphs: connectomes[session][condition][window].
        edge_weight_threshold (float): Minimum edge weight to display.
        save_to_file (bool): If True, saves each graph as an image in the output directory.
        output_dir (str): Directory to save graph images if `save_to_file` is True.
    """
    
    if save_to_file and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                    edge_colors = plt.cm.viridis(norm(weights))
                    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color=node_colors,
                            font_size=10, edge_color=edge_colors, width=2, edge_cmap=plt.cm.viridis)
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
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


# def explain_predictions(model, data, condition_labels):
#     """
#     Gera explicações para as predições do modelo GNN usando Captum.

#     Args:
#         model (nn.Module): O modelo GNN treinado.
#         data (torch_geometric.data.Data): O grafo a ser explicado.
#         condition_labels (list): Lista de rótulos das condições.

#     Returns:
#         None: Exibe os gráficos no Streamlit.
#     """
#     model.eval()

#     # Valida e ajusta edge_index
#     data = validate_and_adjust_edge_index(data)

#     # Move o modelo e os dados para o dispositivo
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     data = data.to(device)

#     # Inicializa o método de explicação
#     ig = IntegratedGradients(lambda x, edge_index: model(x, edge_index, data.batch))

#     # Calcula as atribuições de importância
#     target = data.y.item()
#     attributions, delta = ig.attribute(
#         inputs=(data.x, data.edge_index),
#         target=target,  # Classe real
#         return_convergence_delta=True
#     )

#     # Exibir resultados
#     importances = attributions[0].detach().cpu().numpy()
#     st.write("### Importância das Features dos Nós")
#     st.bar_chart(importances.mean(axis=0))  # Importância média por feature

def explain_predictions(model, data, condition_labels):
    """
    Generates explanations for the GNN model's predictions using Captum.

    Args:
        model (nn.Module): The trained GNN model.
        data (torch_geometric.data.Data): The graph to be explained.
        condition_labels (list): List of condition labels.

    Returns:
        None: Displays charts in Streamlit.
    """
    model.eval()

    # Validate and adjust edge_index
    data = validate_and_adjust_edge_index(data)

    # Move the model and data to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Create a baseline (e.g., zeros)
    baseline = torch.zeros_like(data.x).to(device)

    # Initialize the explanation method
    ig = IntegratedGradients(lambda x, edge_index, batch: model(x, edge_index, batch))

    # Compute the importance attributions
    target = data.y.item()
    attributions, delta = ig.attribute(
        inputs=data.x,
        baselines=baseline,
        target=target,  # Real class label
        additional_forward_args=(data.edge_index, data.batch),
        return_convergence_delta=True
    )

    # Display results
    importances = attributions.detach().cpu().numpy()
    st.write("### Node Feature Importances")
    st.bar_chart(importances.mean(axis=0))  # Average importance per feature


def adjust_tensor_sizes(edge_index, loop_index, num_nodes):
    """
    Ajusta dinamicamente as dimensões de edge_index e loop_index para serem compatíveis.

    Args:
        edge_index (torch.Tensor): Tensor contendo as arestas do grafo.
        loop_index (torch.Tensor): Tensor contendo os laços próprios dos nós.
        num_nodes (int): Número total de nós no grafo.

    Returns:
        torch.Tensor: edge_index ajustado.
    """
    # Certifique-se de que loop_index tenha o número correto de laços próprios
    expected_loops = num_nodes
    if loop_index.size(1) != expected_loops:
        loop_index = torch.arange(0, num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)

    # Verifique a compatibilidade das dimensões
    if edge_index.size(0) != loop_index.size(0):
        raise ValueError(f"Dimensão incompatível: edge_index {edge_index.size(0)}, loop_index {loop_index.size(0)}")

    # Ajuste o tamanho das dimensões, se necessário
    if edge_index.size(1) == 0:  # Caso edge_index esteja vazio
        edge_index = loop_index
    else:
        # Concatene edge_index e loop_index de forma segura
        edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index

# def validate_and_adjust_edge_index(data):
#     """
#     Valida e ajusta o edge_index para garantir consistência, com logs detalhados.

#     Args:
#         data (torch_geometric.data.Data): Objeto de grafo.

#     Returns:
#         data (torch_geometric.data.Data): Objeto de grafo com edge_index ajustado.
#     """
#     num_nodes = data.x.size(0)

#     # Debugging prints
#     print(f"Number of nodes (num_nodes): {num_nodes}")
#     print(f"Original edge_index shape: {data.edge_index.shape if data.edge_index is not None else 'None'}")
#     if data.edge_index is not None:
#         print(f"Original edge_index: {data.edge_index}")

#     # Garante que edge_index não seja None
#     if data.edge_index is None:
#         data.edge_index = torch.empty((2, 0), dtype=torch.long)
#         print("edge_index was None; initialized to empty tensor.")

#     # Verifica se edge_index está vazio ou inconsistente
#     if data.edge_index.size(0) != 2 or data.edge_index.size(1) == 0:
#         print("edge_index is empty or inconsistent. Replacing with self-loops.")
#         data.edge_index = torch.arange(0, num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)

#     # Debugging print
#     print(f"Adjusted edge_index shape: {data.edge_index.shape}")
#     print(f"Adjusted edge_index: {data.edge_index}")

#     # Gera o loop_index
#     loop_index = torch.arange(0, num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)

#     # Debugging print
#     print(f"Generated loop_index shape: {loop_index.shape}")
#     print(f"Generated loop_index: {loop_index}")

#     # Ajusta edge_index com laços próprios
#     try:
#         data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
#             data.edge_index, num_nodes=data.x.size(0)
#         )
#     except RuntimeError as e:
#         print(f"RuntimeError while adding self-loops: {e}")
#         data.edge_index = torch.arange(0, num_nodes, dtype=torch.long).view(1, -1).repeat(2, 1)

#     # Debugging print
#     print(f"Final edge_index shape: {data.edge_index.shape}")
#     print(f"Final edge_index: {data.edge_index}")

#     return data

def validate_and_adjust_edge_index(data):
    """
    Validates and adjusts edge_index to ensure consistency.

    Args:
        data (torch_geometric.data.Data): Graph data object.

    Returns:
        data (torch_geometric.data.Data): Graph data object with adjusted edge_index.
    """
    num_nodes = data.x.size(0)

    # Ensure edge_index is not None
    if data.edge_index is None:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        print("edge_index was None; initialized to empty tensor.")

    # Ensure edge_index is of correct data type and on the correct device
    data.edge_index = data.edge_index.to(torch.long).to(data.x.device)

    # Ensure edge_index has correct shape
    if data.edge_index.size(0) != 2:
        raise ValueError(f"edge_index should have shape [2, num_edges], but got {data.edge_index.shape}")

    # Add self-loops if necessary
    data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
        data.edge_index, num_nodes=num_nodes
    )

    return data


def convert_graph_to_data(graph, label=None, num_features=32):
    """
    Converte um grafo NetworkX para um objeto `torch_geometric.data.Data`.

    Args:
        graph (networkx.Graph): Grafo a ser convertido.
        label (int, optional): Rótulo do grafo.
        num_features (int): Número de features dos nós (padrão: 32).
    
    Returns:
        torch_geometric.data.Data: Objeto de grafo convertido.
    """
    # Verificar se o grafo tem nós
    if graph.number_of_nodes() == 0:
        return None

    # Extrair nós e arestas
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    edge_weights = [graph[u][v].get('weight', 1.0) for u, v in edges]  # Peso padrão é 1.0

    # Mapear nós para índices
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    indexed_edges = [[node_to_index[u], node_to_index[v]] for u, v in edges]

    # Criar tensores de arestas e atributos
    if len(indexed_edges) > 0:
        edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)

    # Criar features dos nós (placeholder com `num_features` features por nó)
    x = torch.ones((len(nodes), num_features), dtype=torch.float)

    # Criar objeto Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long) if label is not None else None
    )

    return data



# Função para visualização com t-SNE
def tsne_visualization(hidden_features, labels, condition_labels):
    from sklearn.manifold import TSNE
    import altair as alt

    # Certifique-se de que o número de amostras é suficiente para a perplexidade
    n_samples = hidden_features.shape[0]
    perplexity = min(30, n_samples // 2)  # Define perplexidade adaptativa

    if n_samples < 2:
        st.error("Não há amostras suficientes para calcular o t-SNE.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings = tsne.fit_transform(hidden_features)

    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'label': [condition_labels[label] for label in labels]
    })

    st.write("Visualização t-SNE")
    st.altair_chart(
        alt.Chart(df).mark_circle(size=60).encode(
            x='x',
            y='y',
            color='label:N',
            tooltip=['label']
        ).interactive()
    )

# Configurações do Streamlit
st.title("Análise de Conectomas com GNN")
st.sidebar.title("Menu")
uploaded_file = st.sidebar.file_uploader("Carregar arquivo de dados", type=["csv", "pickle"])

connectomes = {}

# Modelos e Dados
if uploaded_file:
    st.sidebar.write("Dados carregados!")
    st.write("### Pré-visualização dos dados")
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        with st.spinner('Criando conectomas...'):
            connectomes = generate_cached_connectomes(data)  # Armazena no estado
        st.success("Conectomas gerados!")
    elif uploaded_file.name.endswith('.pickle'):
        data = pd.read_pickle(uploaded_file)
        st.write(data)
    
    # Verifica se o conectoma foi carregado antes de prosseguir
    if connectomes is None:
        st.warning("Por favor, carregue um arquivo de dados antes de continuar.")
    else:
        # Botões para interações principais
        if st.sidebar.button("Gerar Conectoma"):
            st.write("### Conectoma Gerado")
            visualize_all_graphs(connectomes)
            
        if st.sidebar.button("Classificar Grafos"):
            st.write("### Classificação dos Grafos")
            model = load_model("models/gnn_classifier.pth", num_node_features=32, num_classes=12)

            for session, conditions in connectomes.items():
                st.write(f"### Sessão: {session}")
                for condition, windows in conditions.items():
                    st.write(f"#### Condição: {condition}")
                    for window, (graph, cluster_legend) in windows.items():
                        st.write(f"##### Janela: {window}")

                        # Converte o grafo para o formato PyTorch Geometric
                        graph_data = convert_graph_to_data(graph, num_features=32)
                        graph_data = validate_and_adjust_edge_index(graph_data)

                        if graph_data is None or graph_data.x.size(0) < 2:
                            st.warning(f"Grafo da janela {window} é inválido ou tem menos de 2 nós. Ignorando...")
                            continue

                        # Adiciona batch para o pooling global
                        batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

                        # Predição do modelo
                        prediction = model(graph_data.x, graph_data.edge_index, batch)
                        predicted_label = prediction.argmax(dim=1).item()

                        # Exibe a classe predita
                        st.write(f"Classe Predita: {predicted_label}")

                        # Exibe o grafo
                        visualize_graph(graph, session, condition, window)




        # Botão para explicar predições
        if st.sidebar.button("Explicar Predições"):
            st.write("### Explicações das Predições")

            # Carrega o modelo treinado
            model = load_model("models/gnn_classifier.pth", num_node_features=32, num_classes=12)

            # Coleta todas as condições para criar o mapeamento de rótulos
            all_conditions = set()
            for session, conditions in connectomes.items():
                for condition in conditions.keys():
                    all_conditions.add(condition)
            condition_to_label = {condition: idx for idx, condition in enumerate(sorted(all_conditions))}

            # Itera sobre sessões, condições e janelas
            for session, conditions in connectomes.items():
                st.write(f"### Sessão: {session}")
                for condition, windows in conditions.items():
                    st.write(f"#### Condição: {condition}")
                    for window, (graph, cluster_legend) in windows.items():
                        st.write(f"##### Janela: {window}")
                        # Adiciona o rótulo ao grafo
                        label = condition_to_label[condition]
                        
                        # Converte o grafo para o formato PyTorch Geometric
                        graph_data = convert_graph_to_data(graph, label=label, num_features=32)

                        # Adiciona features padrão, se necessário
                        if not hasattr(graph_data, 'x') or graph_data.x is None:
                            num_nodes = graph.number_of_nodes()
                            graph_data.x = torch.ones((num_nodes, 32))  # Placeholder com 32 features por nó

                        # Verifica se as dimensões das features correspondem ao modelo
                        if graph_data.x.shape[1] != 32:
                            st.error(f"As dimensões das features do grafo (janela {window}) não correspondem ao modelo.")
                            continue

                        # Valida e ajusta edge_index
                        graph_data = validate_and_adjust_edge_index(graph_data)


                        # Verifica se o grafo tem nós suficientes
                        if graph_data.x.size(0) < 2:
                            st.warning(f"Grafo da janela {window} tem menos de 2 nós. Ignorando...")
                            continue

                        
                        graph_data.y = torch.tensor([label], dtype=torch.long)

                        # Gera explicações para o grafo atual
                        st.write(f"**Grafo da janela {window} em explicação**")
                        explain_predictions(
                            model, graph_data,
                            condition_labels=[f"Cond{i}" for i in range(1, 13)]  # Ajuste conforme necessário
                        )

        if st.sidebar.button("Visualizar t-SNE"):
            st.write("### Visualização com t-SNE")

            # Carrega o modelo treinado
            model = load_model("models/gnn_classifier.pth", num_node_features=32, num_classes=12)

            hidden_features_list = []
            labels_list = []
            all_conditions = set()

            # Coleta todas as condições para criar o mapeamento de rótulos
            for session, conditions in connectomes.items():
                for condition, windows in conditions.items():
                    all_conditions.add(condition)

            condition_to_label = {condition: idx for idx, condition in enumerate(sorted(all_conditions))}

            # Itera sobre sessões, condições e janelas
            for session, conditions in connectomes.items():
                for condition, windows in conditions.items():
                    for window, (graph, cluster_legend) in windows.items():
                        num_nodes = graph.number_of_nodes()
                        if num_nodes < 2:
                            continue

                        nodes = list(graph.nodes)
                        edges = list(graph.edges)
                        edge_weights = [graph[u][v]['weight'] for u, v in edges]

                        node_to_index = {node: idx for idx, node in enumerate(nodes)}
                        indexed_edges = [[node_to_index[u], node_to_index[v]] for u, v in edges]

                        if len(indexed_edges) > 0:
                            edge_index = torch.tensor(indexed_edges, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
                        else:
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty((0,), dtype=torch.float)

                        x = torch.ones((num_nodes, 32), dtype=torch.float)  # Placeholder com 32 features

                        graph_data = Data(
                            x=x,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                        )

                        # Adiciona o rótulo ao grafo
                        label = condition_to_label[condition]
                        graph_data.y = torch.tensor([label], dtype=torch.long)

                        # Predição do modelo para obter as features ocultas
                        model.eval()
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = model.to(device)
                        graph_data = graph_data.to(device)

                        with torch.no_grad():
                            hidden_features = model.conv1(graph_data.x, graph_data.edge_index)
                            hidden_features_list.append(hidden_features.mean(dim=0).cpu().numpy())
                            labels_list.append(label)

            # Garantir que os tamanhos sejam consistentes
            if len(hidden_features_list) != len(labels_list):
                st.error("Erro: O número de embeddings não corresponde ao número de rótulos.")
            elif len(hidden_features_list) == 0:
                st.error("Nenhum dado disponível para visualização do t-SNE.")
            else:
                hidden_features_combined = np.array(hidden_features_list)
                labels_combined = np.array(labels_list)

                # Visualiza o t-SNE
                condition_labels = [f"Cond{i}" for i in range(1, len(condition_to_label) + 1)]
                tsne_visualization(hidden_features_combined, labels_combined, condition_labels)

