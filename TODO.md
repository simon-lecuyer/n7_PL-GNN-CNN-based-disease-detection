# 📝 Fichiers à Créer - TODO List

## Scripts à Implémenter (`scripts/`)

### Semaine 1 : Data Generation
- [ ] `scripts/generate_data.py`
  - Générer des données de simulation WaterberryFarms
  - Paramètres configurables via YAML
  - Sauvegarder dans `data/raw/`
  
### Semaine 2 : Training Scripts
- [ ] `scripts/train_gnn.py`
  - Entraînement du modèle GNN
  - Logging TensorBoard
  - Sauvegarde checkpoints
  
- [ ] `scripts/train_cnn.py`
  - Entraînement du modèle CNN
  - Logging TensorBoard
  - Sauvegarde checkpoints

### Semaine 3-4 : Evaluation & CGP
- [ ] `scripts/evaluate.py`
  - Évaluation des modèles entraînés
  - Génération métriques
  - Comparaison GNN vs CNN
  
- [ ] `scripts/run_cgp.py`
  - Exécution CGP avec estimateurs GNN/CNN
  - Simulation de drone path planning
  
### Utilitaires
- [ ] `scripts/preprocess_data.py`
  - Prétraitement des données brutes
  - Normalisation, augmentation
  
- [ ] `scripts/visualize_results.py`
  - Génération de figures pour rapport
  - Plots comparatifs

---

## Modules Utilitaires (`utils/`)

### Data Handling
- [ ] `utils/data_loader.py`
  - DataLoader PyTorch pour GNN
  - DataLoader PyTorch pour CNN
  - Fonctions de preprocessing
  
- [ ] `utils/graph_builder.py`
  - Construction du graphe spatial
  - k-NN, distance-weighted edges
  - Crop-type connections
  - Temporal edges

### Models
- [ ] `gnn/models/spatio_temporal_gnn.py`
  - Architecture GNN principale
  - Message passing layers
  - Temporal aggregation
  
- [ ] `cnn/models/disease_detector_cnn.py`
  - Architecture CNN (U-Net style)
  - Encoder-decoder
  
### Evaluation
- [ ] `utils/metrics.py`
  - MSE, MAE, RMSE
  - Accuracy, Precision, Recall, F1
  - IoU pour segmentation
  - Uncertainty metrics
  
### Visualization
- [ ] `utils/visualization.py`
  - Plot training curves
  - Visualize predictions vs ground truth
  - Heatmaps de propagation
  - Drone trajectories

### CGP Integration
- [ ] `utils/cgp/gnn_estimator.py`
  - GNN-based uncertainty estimator
  - Predict mean + variance
  
- [ ] `utils/cgp/cnn_estimator.py`
  - CNN-based uncertainty estimator
  
- [ ] `utils/cgp/path_planner.py`
  - Relevance scoring (α·µ + β·σ)
  - Drone navigation logic
  - Integration avec WBF simulator

---

## Configuration Files (`configs/`)

- [ ] `configs/gnn_config.yaml`
  ```yaml
  model:
    hidden_dim: 64
    num_layers: 3
    k_neighbors: 16
  
  training:
    epochs: 100
    batch_size: 32
    lr: 0.001
    optimizer: adam
  ```

- [ ] `configs/cnn_config.yaml`
  ```yaml
  model:
    architecture: unet
    channels: [32, 64, 128, 256]
  
  training:
    epochs: 100
    batch_size: 16
    lr: 0.0001
  ```

- [ ] `configs/simulation_config.yaml`
  ```yaml
  waterberryfarms:
    field_size: [100, 100]
    num_drones: 1
    disease_model: epidemic_spread
    duration: 100
  ```

- [ ] `configs/cgp_config.yaml`
  ```yaml
  cgp:
    alpha: 0.6  # Weight for exploitation
    beta: 0.4   # Weight for exploration
    estimator: gnn  # or cnn
  ```

---

## Priorités par Semaine

### Semaine 1 (03-09/02)
**Priority 1:**
1. `scripts/generate_data.py`
2. `utils/data_loader.py`
3. `configs/simulation_config.yaml`

**Priority 2:**
4. `utils/visualization.py` (basic plots)

### Semaine 2 (10-16/02)
**Priority 1:**
1. `gnn/models/spatio_temporal_gnn.py`
2. `cnn/models/disease_detector_cnn.py`
3. `utils/graph_builder.py`
4. `scripts/train_gnn.py`
5. `scripts/train_cnn.py`

### Semaine 3 (17-23/02)
**Priority 1:**
1. `utils/metrics.py`
2. `scripts/evaluate.py`
3. `utils/cgp/` (tous les fichiers)
4. `scripts/run_cgp.py`

### Semaine 4 (24/02-02/03)
**Priority 1:**
1. Optimisation des modèles
2. Tests comparatifs
3. `scripts/visualize_results.py`

### Semaine 5 (03-09/03)
**Priority 1:**
1. Finalisation rapport
2. Préparation présentation
3. Documentation finale

---

## Template de Fichier Python

```python
"""
Description du module.

Author: Prénom NOM
Date: JJ/MM/AAAA
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

# TODO: Implémenter


def main():
    """Point d'entrée principal."""
    pass


if __name__ == "__main__":
    main()
```

---

## Checklist de Développement

Pour chaque fichier créé :
- [ ] Docstring complète
- [ ] Type hints
- [ ] Tests unitaires (si applicable)
- [ ] Entrée dans JOURNAL.md
- [ ] Commit Git avec message clair
- [ ] Code review si possible

---

**Mise à jour :** 03/02/2026
