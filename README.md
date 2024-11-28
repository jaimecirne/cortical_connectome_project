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
   ````

   Gere os conectomas:

   ```bash
   python scripts/generate_connectomes.py
   ````

   Treine a GNN:

   ```bash
   python scripts/train_gnn.py
   ```

   Explique as predições:

   ```bash
   python scripts/explain_predictions.py
   ```
   Visualize os pesos com t-SNE:

   ```bash
   python scripts/visualize_tsne.py
   ```
   Execute a interface Streamlit:

   ```bash
   streamlit run scripts/app.py
   ```
   

### Pontos:
1. **Formato das Variáveis:**
   - `x shape`: representa os atributos dos nós no grafo, com dimensões `(número de nós, características por nó)`. Exemplo: `[1024, 32]` indica 1024 nós com 32 atributos cada.
   - `edge_index shape`: define as conexões (arestas) no grafo com formato `[2, número de arestas]`. Cada coluna do tensor `edge_index` representa uma aresta entre dois nós.

2. **Edge Index Tensor:**
   - Este tensor conecta os nós no grafo e define como os dados fluem entre os nós na arquitetura da rede.

3. **Métricas de Treinamento:**
   - **Loss (Perda):** Representa o erro durante o treinamento. Neste caso, a perda permanece em torno de `2.485-2.486`, o que pode indicar dificuldades no aprendizado ou a necessidade de ajustes no modelo.
   - **Acurácia:** As taxas de acurácia de treino (≈8.49%) e teste (≈7.69%) são baixas, sugerindo que o modelo pode estar subajustado ou que os dados possuem um alto grau de complexidade para a arquitetura atual.

4. **Alterações nas Conexões:**
   - As mudanças em `edge_index` entre as épocas podem estar relacionadas a diferentes amostras ou estratégias como dropout estrutural nos grafos, variando a topologia para melhorar a generalização.

5. **Dimensão Reduzida:**
   - Em algumas etapas, a dimensão de `x` e `edge_index` é menor (e.g., `[96, 32]` e `[2, 1488]`). Pode ser uma subamostra ou uma parte específica do grafo processada de forma independente.

### Possíveis Melhorias:
- **Ajuste de Hiperparâmetros:**
  - Revisar a taxa de aprendizado e regularização.
  - Testar arquiteturas diferentes, como aumento de camadas ou dimensões no espaço latente.

- **Pré-processamento de Dados:**
  - Normalização ou balanceamento de dados.
  - Ajuste na definição de conexões para refletir melhor as relações entre os nós.

- **Análise do Modelo:**
  - Verificar se o modelo está aprendendo corretamente (análise de gradientes, checagem de overfitting/underfitting).
  - Avaliar a adequação do modelo à tarefa em termos de arquitetura e loss function.

Se precisar de uma análise mais detalhada ou ajuda para interpretar os resultados, por favor, forneça mais contexto sobre o modelo e o problema abordado.