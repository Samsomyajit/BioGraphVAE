# BioGraphVAE: Bio-Inspired Graph Autoencoder for Structured Data Reconstruction

This repository contains the code and setup to reproduce the results from the paper **"BioGraphVAE: A Bio-inspired Autoencoder for Structured Data Reconstruction."**
BioGraphVAE is a lightweight extension of GraphVAE with a biologically inspired **funnel attention mechanism** that re-weights latent representations to achieve better reconstruction fidelity, smoothness, and diversity.

---

## 📁 Project Structure

```
.
├── data/
│   ├── dataloader.py          # Data loading and preprocessing utilities
│   ├── extract_qm9.py         # Preprocess QM9 dataset
│   ├── extract_zinc.py        # Preprocess ZINC dataset
├── main/
│   ├── experiment_qm9.py      # Training script for QM9
│   ├── experiment_zinc.py     # Training script for ZINC
│   ├── extract_qm9.py         # Preprocessing for QM9 dataset (extracted features)
│   ├── extract_zinc.py        # Preprocessing for ZINC dataset (extracted features)
│   └── visualise.py           # Visualization script for results (PCA/UMAP/TSNE)
├── LICENSE                    # License information
└── README.md                  # This README file
```

---

## ✨ Features

* **Funnel Attention:** A learnable funnel vector applied to latent representations, inspired by photosynthetic energy funnels.
* **Deterministic Noise Schedule:** Linear noise annealing to avoid posterior collapse.
* **GraphVAE baseline comparison** for molecular graph reconstruction.
* **Performance Metrics:** MSE (↓), Diversity (↑), Smoothness (↓), Kolmogorov-Smirnov Statistic (↓).
* **Visualization Support:** PCA, t-SNE, and UMAP embeddings for the latent space structure.

---

## 🚀 Installation

### 1) Create an environment

```bash
conda create -n biographvae python=3.10 -y
conda activate biographvae
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt`:**

```
torch
torch-geometric
numpy
pandas
tqdm
pyyaml
matplotlib
plotly
seaborn
umap-learn
scikit-learn
rdkit
```

---

## 📊 Datasets

You need to preprocess the datasets (QM9 and ZINC). The preprocessing steps are handled in the **`main/extract_qm9.py`** and **`main/extract_zinc.py`** scripts. These scripts will download and preprocess the data, save it as required for model training.

Run the following commands:

```bash
python main/extract_qm9.py   # Downloads and processes QM9
python main/extract_zinc.py  # Downloads and processes ZINC
```

---

## 🎓 Training

You can train BioGraphVAE (or its baselines) on the QM9 and ZINC datasets using the following scripts.

```bash
python main/experiment_qm9.py  # Train BioGraphVAE on QM9
python main/experiment_zinc.py # Train BioGraphVAE on ZINC
```

The `experiment_qm9.py` and `experiment_zinc.py` scripts run the training pipeline for the **BioGraphVAE** model on their respective datasets.

### Training options:

* You can modify the hyperparameters (e.g., batch size, epochs) in the script to fit your setup.
* The script assumes the processed data is stored in the `data/` folder.

---

## 📊 Evaluation & Visualization

After training, you can visualize the performance of the model using the **`visualise.py`** script. This script generates the PCA, t-SNE, and UMAP visualizations.

```bash
python main/visualise.py  # Visualize the results and embeddings
```

This script generates:

* **PCA plots**: Visualizes the overall structure of the latent space.
* **t-SNE/UMAP**: Non-linear manifold visualization of the latent space.

---



## 🔑 Evaluation Metrics

* **MSE (↓)**: Mean-Squared Error, lower values indicate better reconstruction fidelity.
* **Diversity (↑)**: Standard deviation of generated node features, higher values indicate more diversity.
* **Smoothness (↓)**: Mean squared finite difference along canonical node indices, lower values are better.
* **Kolmogorov-Smirnov (KS) (↓)**: Statistical test for distributional mismatch, lower values show better distributional alignment.

---

## 📄 Citing

If you use **BioGraphVAE** in your work, please cite the paper:

```bibtex
@inproceedings{chakraborty2025biographvae,
  title     = {BioGraphVAE: A Bio-inspired Autoencoder for Structured Data Reconstruction},
  author    = {Chakraborty, Somyajit and Chakraborty, Debashis and Mondal, Khokan and Jana, Angshuman and Gayen, Avijit},
  booktitle = {Proceedings of ...},  % update when available
  year      = {2025}
}
```

---

## 📑 License

This code is released under the Apache 2.0 License. See `LICENSE` for details.

---

## 🤝 Contact

If you have any questions or issues, feel free to open an issue or contact us directly.

---

## 🛠 Known Issues & Troubleshooting

### 1) **Data not downloading**

Ensure you're using the correct Python version and all dependencies are installed.

### 2) **Model not training properly**

* Double-check the data preprocessing steps (check if data is correctly split).
* Ensure CUDA drivers are installed if you're using GPU training.

---
