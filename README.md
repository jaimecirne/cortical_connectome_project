---

### **Authors**
Jaime B. Cirne*, Dardo N. Ferreiro++, Sergio A. Conde-Ocazionez***, João H.N. Patriota***, S. Neuenschwander** e Kerstin E. Schmidt*  
(*Universidade Federal do Rio Grande do Norte, Brasil; **Vislab, UFRN; ***Universidade de Amsterdã; ++Ludwig Maximilian Universität, Alemanha)

---

### **Abstract**

In the last two decades, neurophysiological data of both high temporal and spatial resolution from up to hundreds of simultaneously recorded intracortical channels have become increasingly available, thanks to technical advancements such as the higher packing density of microelectrodes, miniaturization of headstages and amplifiers, higher sampling frequencies, and improved signal-to-noise ratios of individual electrodes (e.g., Chen et al., 2022). However, classical analysis techniques often remain focused on serial, single-unit firing rate-based metrics (e.g., Conde-Ocazionez et al., 2017a) or paired measures such as coherence between channels, thereby largely ignoring the high dimensionality of mutual interactions in datasets from multiple parallel channels recorded under the same experimental conditions.
This issue has been partially addressed by computing spike statistics based on neuronal assemblies (e.g., Conde-Ocazionez et al., 2017b), where activity across multiple channels within the same time window is aggregated into population events, partially sacrificing spatial information about electrode distribution within recording matrices with spatial design.
A solution developed to build human brain connectivity maps from diffusion tensor or functional magnetic resonance imaging with datasets offering high spatial but poor temporal resolution, is graph-based analysis of the connectome derived from all functional or anatomical links between defined nodes such as cortical areas (for review, Sporns, 2015). This technique, however, has been sparsely applied to mesoconnectomic data obtained from extracellular recordings of spatially distributed cortical electrodes (e.g., Dann et al., 2016).
The correct interpretation of the results and accurate modeling of underlying circuits also heavily rely on effective data visualization. To address this, we developed a computational tool to model and interactively visualize brain mesoconnectomes derived from preprocessed electrophysiological data, such as interelectrode spike-spike or spike-field coherence measures (NES platform by SN and chronux toolbox by Bokil et al., 2010). The application utilizes graph theory and machine learning techniques to characterize neural connectivity and the identification of functional and structural differences between visual cortical connectomes of species with distinct evolutionary and behavioral traits.
We used metadata from two animal models: the domestic cat (Felis catus), a predator with periodic cortical maps similar to those of primates, and the agouti (Dasyprocta leporina), a diurnal herbivore and highly visual rodent with a cortex size comparable to the cat but featuring a rodent-typical salt-and-pepper map. Both species were studied under identical experimental conditions (Ferreiro et al., 2021).
The interface processes tabular neural connectivity data derived from multiple parallel electrophysiological recordings to model connectomes representing brain activity under different functional contexts. Distinct patterns were observed, confirming higher clustering of connections in the predator's visual cortex and a more dispersed connectivity pattern in the herbivore. The tool also facilitates navigation in big datasets because it enables interactive network visualization, exploratory analysis and deeper interpretation of the neurophysiological data.

---

### **Objective**
The main goal of this application is to process connectomes in graph format, classify them using a trained Graph Neural Network (GNN) model, and provide explanations for the GNN model's predictions through feature attribution techniques (Integrated Gradients).

---

### **Methodology**
- **Studied Species**: Cat (Felis catus) and Agouti (Dasyprocta leporina).
- **Input**: Connectivity data from animals (e.g., Spike-Spike Coherence).
- **Output**:  Interactive graphs with representative clusters.
- **Implementation**:
  - Pipeline developed in **Python**.
  - Classification using **PyTorch Geometric**.
  - Interpretable analysis via **Captum**.
- **Techniques Used**:
  - Graph analysis with **NetworkX**.
  - Modeling based on **Graph Neural Networks (GNN)**.
  - Interactive visualization using **Streamlit**.
  - Interpretable predictions using the [Captum](https://captum.ai/) library
  - Embedding visualization projects learned hidden representations (features) in 2D space using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
- **Dataset**: Electrophysiological records organized under identical experimental conditions.

---

### **Results**
#### Connectome Visualization
   ![Visualização do Conectoma](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/conectoma_visual.png?raw=true)

#### Louvain Clustering
   ![Clusterização Louvain](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/clusters.png?raw=true)

#### GNN Model Performance  

| Category  | Precision | Recall | F1-Score | Support |
|------------|----------|--------|----------|---------|
| **Cat** | 0.95     | 0.95   | 0.95     | 60      |
| **Agouti**   | 0.95     | 0.95   | 0.95     | 56      |
| **Accuracy** | -        | -      | 0.95     | 116     |

#### GNN Classifier
The model architecture includes:
- **GCNConv**: Graph convolutional layers to capture local dependencies.
- **Global Mean Pooling**: Reduces node representations to represent the entire graph.
- **Fully Connected Layer**: Produces the final prediction.

#### Classifier Example  
| |session|condition|window|Prediction|Predicted Species|
|------|-------|---------|------|----------|-----------------|
|0     |S1     |1        |Win0  |0         |predator         |
|1     |S1     |1        |Win1  |0         |predator         |
|2     |S1     |1        |Win2  |0         |predator         |
|3     |S1     |2        |Win0  |0         |predator         |
|4     |S1     |2        |Win1  |0         |predator         |
|5     |S1     |2        |Win2  |0         |predator         |

...

#### Captum
We used Captum to generate explanations with:
- **Integrated Gradients**: Measures the contribution of each feature to the prediction.
 ![Integrated Gradients](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/Prediction1_ig.png?raw=true)


#### t-SNE
Reduces the dimensions of the learned embeddings to 2D, allowing visualization of the relationships between graphs.

 ![Integrated Gradients](https://github.com/jaimecirne/cortical_connectome_project/blob/main/img/t-SNEs_small.png?raw=true)
---
  
### **Contributions**
1. **Advanced Visualization**: Interactive connectomes for navigating large datasets.
2. **Pattern Detection**: Significant structural differences in the studied species.
3. **Educational Tool**: Potential for neuroscience training.

---

### **References**
1. Chen et al., (2022) *Nature Genetics*. DOI: [10.1038/s41597-022-01180-1](https://doi.org/10.1038/s41597-022-01180-1)
2. Ferreiro et al., (2021) *iScience*. DOI: [10.1016/j.isci.2020.101882](https://doi.org/10.1016/j.isci.2020.101882)
3. Bokil et al. (2010) *Journal of Neuroscience Methods*. DOI: [10.1016/j.jneumeth.2010.06.020](https://doi.org/10.1016/j.jneumeth.2010.06.020)
4. Conde-Ocazionez et al., (2018a),  DOI: [10.3389/fnsys.2018.00011](https://doi.org/10.3389/fnsys.2018.00011)
5. Conde-Ocazionez et al., (2018b),  DOI: [10.1111/ejn.13786](https://doi.org/10.1111/ejn.13786)

---
### **Contact**

[Instituto do Cérebro - UFRN](https://www.neuro.ufrn.br)  

[Biome - UFRN](https://bioinfo.imd.ufrn.br/)  

---

## **Installation**

### **1. Prerequisites**
Make sure you have the following tools installed:
- [Python 3.9+](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)
- [miniconda](https://docs.anaconda.com/miniconda/install/)

### **2. Clone the Repository**
```bash
git clone https://github.com/jaimecirne/cortical_connectome_project/
cd cortical_connectome_project
```

### **3. Install Dependencies**
It is recommended to use a virtual environment:
```bash
conda env create -f environment.yml
```

---

## **How to Use**

1. **Start the Application**
   ```bash
   streamlit run scripts/app.py
   ```

---

## **License**
Este projeto é distribuído sob a licença MIT.

---
**Extra tips:**
   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE
   ````