# Projeto de Análise de Conectomas Corticais com GNN

# **Conectoma GNN Analyser**

Este projeto é uma aplicação interativa desenvolvida com [Streamlit](https://streamlit.io/) para analisar conectomas utilizando redes neurais gráficas (GNNs). Ele permite a visualização, classificação e explicação de grafos derivados de conectomas, tornando o fluxo de trabalho intuitivo para pesquisadores e cientistas de dados.

---

## **Objetivo**
O objetivo principal desta aplicação é processar conectomas em formato de grafo, classificá-los com um modelo de rede neural gráfica (GNN) treinado e oferecer explicações interpretáveis das predições usando a biblioteca [Captum](https://captum.ai/).

---

## **Características Principais**

1. **Upload de Arquivos**:
   - Suporte para arquivos `.csv` e `.pickle`.
   - Visualização dos dados carregados diretamente na interface.

2. **Geração de Conectomas**:
   - Processa os dados carregados e converte-os em grafos conectoma.
   - Gera representações gráficas dos conectomas.

3. **Classificação de Grafos**:
   - Utiliza um modelo GNN pré-treinado para classificar os grafos.
   - Exibe o rótulo predito e o grafo correspondente.

4. **Explicação das Predições**:
   - Explica as predições do modelo GNN utilizando técnicas de atribuição de importância (Integrated Gradients).
   - Apresenta gráficos interpretáveis para as contribuições das features.

5. **Visualização de Embeddings**:
   - Projeta as representações ocultas (features aprendidas) em um espaço 2D utilizando [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

---

## **Instalação**

### **1. Pré-requisitos**
Certifique-se de que você tem as seguintes ferramentas instaladas:
- [Python 3.8+](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)

### **2. Clone o Repositório**
```bash
git clone https://github.com/seu-usuario/conectoma-gnn-analyser.git
cd conectoma-gnn-analyser
```

### **3. Instale as Dependências**
Recomenda-se o uso de um ambiente virtual:
```bash
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## **Como Usar**

1. **Inicie a Aplicação**
   ```bash
   streamlit run scripts/app.py
   ```

2. **Carregue os Dados**
   - Clique em **"Carregar arquivo de dados"** no menu lateral e selecione um arquivo `.csv` ou `.pickle`.

3. **Gere os Conectomas**
   - Clique no botão **"Gerar Conectoma"** para visualizar os grafos processados.

4. **Classifique os Grafos**
   - Clique em **"Classificar Grafos"** para prever os rótulos dos conectomas com o modelo GNN pré-treinado.

5. **Explique as Predições**
   - Clique em **"Explicar Predições"** para entender quais features influenciam as predições.

6. **Visualize com t-SNE**
   - Clique em **"Visualizar t-SNE"** para explorar as representações ocultas das features aprendidas pelo modelo.

---

## **Fluxo de Trabalho**

### **1. Processamento dos Dados**
Os dados carregados são transformados em conectomas utilizando a função `generate_connectome_from_data`. Cada conectoma é representado como um grafo contendo:
- **Nós**: Representando regiões cerebrais.
- **Arestas**: Representando conexões ponderadas entre regiões.

### **2. Geração de Conectomas**
Os conectomas são exibidos graficamente com informações adicionais, como pesos das arestas e clusters.

### **3. Classificação**
O modelo GNN analisa os grafos e prediz um rótulo correspondente à classe do conectoma.

### **4. Explicações**
Utilizamos o método **Integrated Gradients** da biblioteca Captum para calcular a importância das features dos nós na predição, tornando as decisões do modelo mais interpretáveis.

---

## **Explicação Técnica**
### **GNN Classifier**
A arquitetura do modelo inclui:
- **GCNConv**: Camadas convolucionais gráficas para capturar dependências locais.
- **Global Mean Pooling**: Reduz as representações dos nós para representar o grafo inteiro.
- **Fully Connected Layer**: Para gerar a predição final.

### **Captum**
Utilizamos o Captum para gerar explicações com:
- **Integrated Gradients**: Mede a contribuição de cada feature para a predição.

### **t-SNE**
Reduz as dimensões dos embeddings aprendidos para 2D, permitindo visualizar as relações entre os grafos.

---

## **Exemplo de Uso**

1. **Carregue um arquivo de dados:**
   - Um arquivo `.csv` contendo métricas de conectividade entre regiões cerebrais.

2. **Visualize os conectomas gerados:**
   - Grafos coloridos que mostram conexões ponderadas.

3. **Execute a classificação e visualize os rótulos:**
   - O modelo GNN prediz os estados ou condições cerebrais.

4. **Explore as explicações:**
   - Gráficos de barras que mostram a importância relativa de cada feature na predição.

5. **Visualize as relações dos conectomas:**
   - Use t-SNE para identificar padrões e clusters nos conectomas.

---

## **Contribuições**

1. **Adicione novos modelos**
   - Treine ou implemente novos modelos GNN com diferentes arquiteturas.

2. **Integre novos dados**
   - Adapte a função de geração de conectomas para diferentes formatos ou tipos de conectividade.

3. **Melhore as visualizações**
   - Personalize as representações gráficas para adicionar mais informações.

---

## **Licença**
Este projeto é distribuído sob a licença MIT.

---

Com esta explicação didática, a documentação do projeto está mais clara e organizada para uso e contribuição!
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