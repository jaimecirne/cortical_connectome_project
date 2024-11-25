# Projeto de Análise de Conectomas Corticais com GNN

Este projeto implementa um pipeline completo para a análise de conectomas corticais a partir de dados eletrofisiológicos, incluindo:

- Geração de conectomas a partir de dados CSV.
- Implementação de uma Graph Neural Network (GNN) para classificação.
- Explicação das predições usando a biblioteca Captum.
- Visualização dos pesos dos neurônios com t-SNE.
- Interface interativa com Streamlit.

## **Estrutura do Projeto**

- **data/**: Dados de entrada em formato CSV.
- **outputs/**: Resultados gerados pelos scripts.
- **models/**: Modelos treinados.
- **scripts/**: Scripts Python para cada etapa do pipeline.
- **README.md**: Documentação do projeto.

## **Requisitos**

- Python 3.7 ou superior
- Bibliotecas listadas no `requirements.txt` (deve ser criado com base nas bibliotecas usadas)

## **Instruções de Uso**

1. **Clone o repositório e navegue até o diretório do projeto:**

   ```bash
   git clone https://github.com/jaimecirne/cortical_connectome_project.git
   cd cortical_connectome_project

   conda env create -f environment.yml

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+gpu.html

   pip install captum

   set KMP_DUPLICATE_LIB_OK=TRUE

   streamlit run .\scripts\app.py