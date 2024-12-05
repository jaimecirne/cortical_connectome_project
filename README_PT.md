### **Autores**  
Jaime B. Cirne*, Dardo N. Ferreiro++, Sergio A. Conde-Ocazionez***, João H.N. Patriota***, S. Neuenschwander** e Kerstin E. Schmidt*  
(*Universidade Federal do Rio Grande do Norte, Brasil; **Vislab, UFRN; ***Universidade de Amsterdã; ++Ludwig Maximilian Universität, Alemanha)  

---

### **Resumo**  

Nas últimas duas décadas, dados neurofisiológicos de alta resolução temporal e espacial provenientes de centenas de canais intracorticais registrados simultaneamente tornaram-se cada vez mais disponíveis, graças a avanços técnicos, como maior densidade de empacotamento de microeletrodos, miniaturização de cabeçotes e amplificadores, frequências de amostragem mais altas e melhor relação sinal-ruído de eletrodos individuais (e.g., Chen et al., 2022). No entanto, as técnicas clássicas de análise frequentemente permanecem focadas em métricas seriais baseadas em taxas de disparo de unidades únicas (e.g., Conde-Ocazionez et al., 2017a) ou em medidas pareadas, como coerência entre canais, ignorando amplamente a alta dimensionalidade das interações mútuas em conjuntos de dados de múltiplos canais paralelos registrados sob as mesmas condições experimentais.  
Essa questão foi parcialmente abordada através do cálculo de estatísticas de spikes baseadas em conjuntos neuronais (e.g., Conde-Ocazionez et al., 2017b), onde a atividade de múltiplos canais dentro de uma mesma janela de tempo é agregada em eventos populacionais, sacrificando parcialmente a informação espacial sobre a distribuição de eletrodos nas matrizes de registro com design espacial.  
Uma solução desenvolvida para construir mapas de conectividade do cérebro humano a partir de imagens de ressonância magnética funcional ou por tensor de difusão com dados de alta resolução espacial, mas baixa resolução temporal, é a análise baseada em grafos do conectoma derivado de todas as conexões funcionais ou anatômicas entre nós definidos, como áreas corticais (para revisão, Sporns, 2015). Contudo, essa técnica tem sido raramente aplicada a dados mesoconectômicos obtidos de registros extracelulares de eletrodos corticais distribuídos espacialmente (e.g., Dann et al., 2016).  
A interpretação correta dos resultados e a modelagem precisa dos circuitos subjacentes também dependem fortemente de uma visualização de dados eficaz. Para lidar com isso, desenvolvemos uma ferramenta computacional para modelar e visualizar interativamente mesoconectomas cerebrais derivados de dados eletrofisiológicos pré-processados, como medidas de coerência spike-spike ou spike-field entre eletrodos (plataforma NES de SN e toolbox Chronux de Bokil et al., 2010). A aplicação utiliza teoria dos grafos e técnicas de aprendizado de máquina para caracterizar conectividade neural e identificar diferenças funcionais e estruturais entre conectomas corticais visuais de espécies com características evolutivas e comportamentais distintas.  
Usamos metadados de dois modelos animais: o gato doméstico (*Felis catus*), um predador com mapas corticais periódicos semelhantes aos de primatas, e a cutia (*Dasyprocta leporina*), um herbívoro diurno e roedor altamente visual com um tamanho de córtex comparável ao do gato, mas apresentando um mapa típico de roedores em "sal e pimenta". Ambas as espécies foram estudadas sob condições experimentais idênticas (Ferreiro et al., 2021).  
A interface processa dados tabulares de conectividade neural derivados de múltiplos registros eletrofisiológicos paralelos para modelar conectomas que representam a atividade cerebral em diferentes contextos funcionais. Padrões distintos foram observados, confirmando maior agrupamento de conexões no córtex visual do predador e um padrão de conectividade mais disperso no herbívoro. A ferramenta também facilita a navegação em grandes conjuntos de dados, permitindo visualização interativa de redes, análise exploratória e interpretação mais profunda dos dados neurofisiológicos.  

---

### **Objetivo**  
O objetivo principal desta aplicação é processar conectomas em formato de grafo, classificá-los usando um modelo de Redes Neurais em Grafos (GNN) treinado e fornecer explicações para as predições do modelo GNN por meio de técnicas de atribuição de características (*Integrated Gradients*).  

---

### **Metodologia**  
- **Espécies Estudadas**: Gato (*Felis catus*) e Cutia (*Dasyprocta leporina*).  
- **Entrada**: Dados de conectividade dos animais (e.g., Coerência Spike-Spike).  
- **Saída**: Grafos interativos com agrupamentos representativos.  
- **Implementação**:  
  - Pipeline desenvolvido em **Python**.  
  - Classificação usando **PyTorch Geometric**.  
  - Análise interpretável via **Captum**.  
- **Técnicas Usadas**:  
  - Análise de grafos com **NetworkX**.  
  - Modelagem baseada em **Redes Neurais em Grafos (GNN)**.  
  - Visualização interativa usando **Streamlit**.  
  - Predições interpretáveis usando a biblioteca [Captum](https://captum.ai/).  
  - Visualização de embeddings em 2D usando [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).  
- **Conjunto de Dados**: Registros eletrofisiológicos organizados sob condições experimentais idênticas.  

---

### **Resultados**  
#### Visualização do Conectoma  
![Visualização do Conectoma](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/conectoma_visual.png?raw=true)  

#### Clusterização Louvain  
![Clusterização Louvain](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/clusters.png?raw=true)  

#### Desempenho do Modelo GNN  

| Categoria  | Precisão | Recall | F1-Score | Suporte |  
|------------|----------|--------|----------|---------|  
| **Gato**   | 0.95     | 0.95   | 0.95     | 60      |  
| **Cutia**  | 0.95     | 0.95   | 0.95     | 56      |  
| **Acurácia** | -        | -      | 0.95     | 116     |  

#### Classificador GNN  
A arquitetura do modelo inclui:  
- **GCNConv**: Camadas de convolução em grafos para capturar dependências locais.  
- **Global Mean Pooling**: Reduz representações de nós para representar todo o grafo.  
- **Camada Totalmente Conectada**: Produz a predição final.  

#### Exemplo de Classificação  
| | Sessão | Condição | Janela | Predição | Espécie Prevista |  
|---|--------|----------|--------|----------|-----------------|  
| 0 | S1     | 1        | Win0   | 0        | predador        |  
| 1 | S1     | 1        | Win1   | 0        | predador        |  

...  

#### Captum  
Usamos Captum para gerar explicações com:  
- **Integrated Gradients**: Mede a contribuição de cada característica para a predição.  

#### t-SNE  
Reduz as dimensões dos embeddings aprendidos para 2D, permitindo a visualização das relações entre grafos.  

---

### **Contribuições**  
1. **Visualização Avançada**: Conectomas interativos para navegação em grandes conjuntos de dados.  
2. **Detecção de Padrões**: Diferenças estruturais significativas nas espécies estudadas.  
3. **Ferramenta Educacional**: Potencial para treinamento em neurociência.  

---

### **Referências**  
1. Chen et al., (2022) *Nature Genetics*. DOI: [10.1038/s41597-022-01180-1](https://doi.org/10.1038/s41597-022-01180-1)
2. Ferreiro et al., (2021) *iScience*. DOI: [10.1016/j.isci.2020.101882](https://doi.org/10.1016/j.isci.2020.101882)
3. Bokil et al. (2010) *Journal of Neuroscience Methods*. DOI: [10.1016/j.jneumeth.2010.06.020](https://doi.org/10.1016/j.jneumeth.2010.06.020)
4. Conde-Ocazionez et al., (2018a),  DOI: [10.3389/fnsys.2018.00011](https://doi.org/10.3389/fnsys.2018.00011)
5. Conde-Ocazionez et al., (2018b),  DOI: [10.1111/ejn.13786](https://doi.org/10.1111/ejn.13786)
---  

### **Contato**  
[Instituto do Cérebro - UFRN](https://www.neuro.ufrn.br)  
[Biome - UFRN](https://bioinfo.imd.ufrn.br/)  

---

### **Instalação**  

#### **1. Pré-requisitos**  
Certifique-se de ter as seguintes ferramentas instaladas:  
- [Python 3.9+](https://www.python.org/)  
- [pip](https://pip.pypa.io/en/stable/)  
- [miniconda](https://docs.anaconda.com/miniconda/install/)  

#### **2. Clone o Repositório**  
```bash  
git clone https://github.com/jaimecirne/cortical_connectome_project/  
cd cortical_connectome_project  
```  

#### **3. Instale as Dependências**  
Recomenda-se usar um ambiente virtual:  
```bash  
conda env create -f

 environment.yml  
```  

---

### **Como Usar**  
1. **Inicie a Aplicação**  
```bash  
streamlit run scripts/app.py  
```  

---

### **Licença**  
Este projeto é distribuído sob a licença MIT.  

---  

**Dicas extras:**  
```bash  
set KMP_DUPLICATE_LIB_OK=TRUE  
```