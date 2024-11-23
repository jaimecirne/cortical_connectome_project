# scripts/app.py

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch

# Funções importadas dos scripts correspondentes
from generate_connectomes import process_condition
from train_gnn import GNNClassifier, create_data_object
from explain_predictions import explain_prediction, visualize_attributions

def generate_connectome_from_data(data):
    G, adj_matrix = process_condition(data, condition_label='Uploaded')
    return G

def make_prediction(model, data_object):
    model.eval()
    data_object = data_object.to(device)
    out = model(data_object)
    pred = out.argmax(dim=1).item()
    return pred

def visualize_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='lightblue')
    st.pyplot(plt)
    plt.close()

st.title('Análise de Conectomas e Classificação com GNN')

uploaded_file = st.file_uploader('Faça o upload dos dados CSV', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Pré-visualização dos Dados:')
    st.write(data.head())
    st.write('Gerando conectoma...')
    G = generate_connectome_from_data(data)
    st.write('Grafo do Conectoma:')
    visualize_graph(G)
    data_object = create_data_object(G, label=0)  # Label fictício
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(num_node_features=32, num_classes=12).to(device)
    model.load_state_dict(torch.load('../models/gnn_classifier.pth'))
    prediction = make_prediction(model, data_object)
    st.write(f'Condição Predita: {prediction}')
    st.write('Explicando a Predição:')
    attributions = explain_prediction(model, data_object)
    visualize_attributions(G, attributions, '../outputs/temp_attribution.png')
    st.image('../outputs/temp_attribution.png')
    st.write('Visualização t-SNE:')
    st.image('../outputs/tsne_visualization.png')
else:
    st.write('Por favor, faça o upload de um arquivo CSV para continuar.')
