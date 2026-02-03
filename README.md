# GNN-CNN Based Disease Detection in Agriculture

**Graph Neural Networks and Convolutional Neural Networks for Disease Detection and Propagation Prediction**

![Project Status](https://img.shields.io/badge/status-in--development-yellow)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

## 📋 Project Overview

This project addresses the challenge of disease detection and propagation prediction in precision agriculture using machine learning approaches. We implement and compare two complementary methods:

- **CNN (Convolutional Neural Network)**: Image-based disease detection from drone imagery
- **GNN (Graph Neural Network)**: Spatio-temporal disease propagation modeling

Both approaches are integrated with **Confidence-Guided Path Planning (CGP)** to optimize drone-based monitoring strategies using the **WaterberryFarms** simulator.

### Key Features

- Multi-modal disease detection using CNNs and GNNs
- Spatio-temporal modeling of disease spread patterns
- Integration with WaterberryFarms simulator for realistic data generation
- Adaptive drone path planning based on model uncertainty
- Comparative evaluation framework for both approaches

## 🎯 Objectives

1. Develop CNN models for real-time disease detection from aerial imagery
2. Implement GNN architectures to model disease propagation across spatial networks
3. Integrate both models into a CGP framework for optimized drone monitoring
4. Compare performance in terms of accuracy, robustness, and computational efficiency
5. Provide actionable insights for precision agriculture applications

## 🏗️ Project Structure

```
n7_PL-GNN-CNN-based-disease-detection/
├── data/                          # Dataset handling
│   ├── raw/                       # Raw simulation outputs from WaterberryFarms
│   ├── processed/                 # Preprocessed data for training
│   └── simulations/               # Simulation configuration files
│
├── gnn/                           # Graph Neural Network
│   ├── models/                    # GNN architectures (Python modules)
│   ├── training_logs/             # TensorBoard logs
│   └── checkpoints/               # Model checkpoints (.pth)
│
├── cnn/                           # Convolutional Neural Network
│   ├── models/                    # CNN architectures (Python modules)
│   ├── training_logs/             # TensorBoard logs
│   └── checkpoints/               # Model checkpoints (.pth)
│
├── results/                       # Experimental results
│   ├── figures/                   # Visualizations (PNG, PDF)
│   ├── metrics/                   # Performance metrics (JSON, CSV)
│   └── comparisons/               # Model comparisons
│
├── configs/                       # Configuration YAML files
│
├── utils/                         # Utility modules
│   ├── cgp/                      # Confidence-Guided Path Planning
│   ├── data_loader.py            # Data loading
│   ├── graph_builder.py          # Graph construction for GNN
│   ├── metrics.py                # Evaluation metrics
│   └── visualization.py          # Plotting tools
│
├── scripts/                       # Executable Python scripts
│   ├── generate_data.py          # Generate data from WaterberryFarms
│   ├── train_gnn.py              # Train GNN
│   ├── train_cnn.py              # Train CNN
│   ├── evaluate.py               # Evaluation
│   └── run_cgp.py                # CGP integration
│
├── test/                          # 📓 Notebooks for testing/demos (not versioned)
│
├── .gitignore
├── requirements.txt
├── JOURNAL.md                     # 📝 Team development log
└── README.md
```

**💡 Design:** Code in Python scripts (`scripts/`, `utils/`), NOT notebooks. Notebooks only in `test/` for exploration (not versioned).

## 🚀 Getting Started

### Installation

**1. Créer l'environnement conda**
```bash
conda create -n N7_PL python=3.10 -y
conda activate N7_PL
```

**2. Installer PyTorch** (choisir selon votre configuration)

```bash
# GPU NVIDIA (Linux/Windows)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Apple Silicon (M1/M2/M3)
conda install pytorch torchvision -c pytorch -y

# CPU only
conda install pytorch torchvision cpuonly -c pytorch -y
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**Commande complète (une seule ligne) :**
```bash
# GPU NVIDIA
conda create -n N7_PL python=3.10 -y && conda activate N7_PL && conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y && pip install -r requirements.txt

# Apple Silicon
conda create -n N7_PL python=3.10 -y && conda activate N7_PL && conda install pytorch torchvision -c pytorch -y && pip install -r requirements.txt

# CPU only
conda create -n N7_PL python=3.10 -y && conda activate N7_PL && conda install pytorch torchvision cpuonly -c pytorch -y && pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Activer l'environnement
conda activate N7_PL

# 2. Générer les données
python scripts/generate_data.py --config configs/simulation_config.yaml

# 3. Entraîner les modèles
python scripts/train_gnn.py --config configs/gnn_config.yaml
python scripts/train_cnn.py --config configs/cnn_config.yaml

# 4. Évaluer
python scripts/evaluate.py --model gnn
```

## 📊 Methodology

### GNN Approach

The GNN models disease propagation as a graph where:
- **Nodes**: Grid cells in the agricultural field
- **Edges**: Spatial connectivity (k-NN, distance-weighted, crop-type based)
- **Features**: Positional encoding, crop type, observation mask, disease state, temporal history

The model predicts continuous infection probability: ŷᵢ(t+k) ∈ [0,1]

**Key Advantages:**
- Captures spatial transmission patterns
- Models non-local disease spread (wind, insects)
- Incorporates domain knowledge (crop types)
- Provides uncertainty estimates

### CNN Approach

The CNN processes grid-based field representations as images:
- **Input**: Multi-channel images (disease state, crop type, observation mask)
- **Architecture**: Encoder-decoder or U-Net style networks
- **Output**: Pixel-wise disease probability maps

**Key Advantages:**
- Fast inference for real-time monitoring
- Effective for local spatial patterns
- Well-established architectures
- Good performance on dense observations

### CGP Integration

Both models integrate with Confidence-Guided Path Planning:
1. Models predict mean infection µ̂ᵢ and uncertainty σ̂²ᵢ
2. Relevance score: rᵢ = α·µ̂ᵢ + β·σ̂ᵢ
3. Drone navigates to maximize rᵢ (balance exploration/exploitation)

## 📈 Timeline (5 Weeks)

| Week | Objectives | Team 1 (CNN) | Team 2 (GNN) |
|------|-----------|--------------|--------------|
| **Week 1** | Data Preparation | Generate image datasets | Extract graph structure |
| **Week 2** | Baseline Models | Implement CNN architecture | Design GNN graph + baseline |
| **Week 3** | Optimization & CGP | Optimize CNN + real-time inference | Enhance GNN + CGP integration |
| **Week 4** | Evaluation | Design test cases + benchmark CNN | Benchmark GNN + dynamic tests |
| **Week 5** | Finalization | Finalize CNN + discussion | Finalize GNN + visualizations |

## 👥 Team

- **Myriam ROBBANA**
- **Mihai COSTIN**
- **Simon LECUYER**
- **Assala ASSELALOU**
- **Yassin MOUKAN**

**École Nationale Supérieure d'Électrotechnique, d'Électronique, d'Informatique, d'Hydraulique et des Télécommunications (N7)**

## 📚 References

1. Matloob et al. (2023) - *Grid Limited Randomness Path Planning*
2. Matloob et al. (2023) - *Waterberry Farms Benchmark*
3. Bölöni & Matloob (2024) - *WaterberryFarms Framework*
4. Matloob et al. (2025) - *Bounomodes Algorithm*
5. Turgut et al. (2023) - *Confidence-Guided Path Planning*
6. Jahin et al. (2025) - *Hybrid CNN-GNN for Soybean Disease Detection*

## 📝 License

This project is part of academic research at ENSEEIHT (N7). Please cite appropriately if using this work.

## 📊 Team Contributions

See [JOURNAL.md](JOURNAL.md) for development log.

This is an academic project. For questions or collaboration inquiries, please contact the team members.

---

**Last Updated:** February 2026
