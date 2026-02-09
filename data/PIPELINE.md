# 📊 Pipeline de Données - Guide Complet

## Vue d'ensemble

Cette pipeline transforme les simulations brutes WaterberryFarms en datasets PyTorch prêts pour l'entraînement CNN et GNN.

```
Simulations     Prétraitement      Datasets        Entraînement
  (brutes)         (images)        PyTorch           (modèles)
     │                │                │                  │
     ▼                ▼                ▼                  ▼
generate_data → preprocess_data → create_datasets → train_*.py
```

## 🔄 Étapes de la Pipeline

### 1️⃣ Génération de Simulations

**Script:** `scripts/generate_data.py`

Génère des simulations de propagation de maladies avec WaterberryFarms.

```bash
python scripts/generate_data.py \
    --num_simulations 50 \
    --grid_size 100 \
    --timesteps 200 \
    --model_type epidemic \
    --p_transmission 0.2 \
    --seed 42
```

**Sortie:** `data/simulations/generation_TIMESTAMP/`
- Images brutes (PNG)
- Graphes bruts (NPY)
- Métadonnées complètes

---

### 2️⃣ Prétraitement des Données

**Script:** `scripts/preprocess_data.py`

Transforme les données brutes en formats optimisés pour CNN et GNN.

#### Usage de base

```bash
python scripts/preprocess_data.py \
    --input data/simulations/generation_20260204_173051 \
    --output data/processed \
    --target_size 64 \
    --normalize \
    --crop
```

#### Paramètres importants

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--input` | - | Dossier de génération (requis) |
| `--output` | `data/processed` | Dossier de sortie |
| `--target_size` | 64 | Taille des images CNN (64x64) |
| `--crop` | False | Crop automatique sur zones infectées |
| `--crop_margin` | 5 | Marge autour du crop |
| `--normalize` | False | Normaliser en [0, 1] |
| `--standardize` | False | Standardiser (mean=0, std=1) |
| `--add_spatial_features` | False | Ajouter coordonnées aux features GNN |
| `--min_infection` | 0.0 | Seuil minimum pour garder timestep |
| `--formats` | `cnn gnn` | Formats à générer |

#### Optimisations CNN

**Crop automatique:**
```bash
--crop --crop_margin 10
```
→ Se concentre sur les zones avec infection, réduit le bruit

**Normalisation:**
```bash
--normalize  # [0, 1]
# ou
--standardize  # mean=0, std=1
```

#### Optimisations GNN

**Features spatiales:**
```bash
--add_spatial_features
```
→ Ajoute (x, y) normalisées aux features des nœuds

**Distance personnalisée:**
```bash
--edge_threshold 2.0
```
→ Connecte nœuds à distance ≤ 2.0 (défaut: 4-voisinage)

#### Exemples

**CNN optimisé:**
```bash
python scripts/preprocess_data.py \
    --input data/simulations/generation_XXX \
    --formats cnn \
    --target_size 64 \
    --crop \
    --normalize
```

**GNN avec features enrichies:**
```bash
python scripts/preprocess_data.py \
    --input data/simulations/generation_XXX \
    --formats gnn \
    --add_spatial_features \
    --normalize
```

**Les deux formats (comparaison):**
```bash
python scripts/preprocess_data.py \
    --input data/simulations/generation_XXX \
    --formats cnn gnn \
    --target_size 64 \
    --crop \
    --normalize \
    --add_spatial_features
```

**Sortie:** `data/processed/processed_TIMESTAMP/`
```
processed_20260209_165358/
├── preprocessing_metadata.json
├── cnn/
│   ├── sim_0000/
│   │   ├── t_0000.npy  # {'data': array(64,64), 'crop_bbox': ...}
│   │   └── ...
│   └── sim_0001/
└── gnn/
    ├── sim_0000/
    │   ├── t_0000.npy  # {'nodes': ..., 'edges': ..., 'node_features': ...}
    │   └── ...
    └── sim_0001/
```

---

### 3️⃣ Création des Datasets PyTorch

**Script:** `scripts/create_datasets.py`

Crée les splits train/val/test avec stratification.

#### Usage de base

```bash
python scripts/create_datasets.py \
    --input data/processed/processed_20260209_165358 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --stratify_by simulation \
    --seed 42
```

#### Paramètres de split

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--train_ratio` | 0.7 | Proportion train (70%) |
| `--val_ratio` | 0.15 | Proportion validation (15%) |
| `--test_ratio` | 0.15 | Proportion test (15%) |
| `--stratify_by` | `simulation` | Stratification (simulation/infection_level/timestep/none) |
| `--seed` | 42 | Seed pour reproductibilité |

#### Séquences temporelles

Pour modèles temporels (LSTM, Transformer):

```bash
python scripts/create_datasets.py \
    --input data/processed/processed_XXX \
    --create_sequences \
    --sequence_length 5 \
    --sequence_stride 1
```

**Sortie:** `data/processed/processed_XXX/datasets/`
```
datasets/
├── dataset_metadata.json
├── cnn/
│   ├── train.json      # Liste des fichiers train
│   ├── train.pkl       # Version pickle (rapide)
│   ├── val.json
│   ├── val.pkl
│   ├── test.json
│   └── test.pkl
└── gnn/
    ├── train.json
    ├── val.json
    └── test.json
```

---

### 4️⃣ Utilisation dans PyTorch

**Module:** `utils/datasets.py`

#### Dataset simple

```python
from utils.datasets import DiseaseDetectionDataset, get_dataloader
from torchvision import transforms

# CNN
train_loader = get_dataloader(
    "data/processed/processed_XXX/datasets/cnn/train.json",
    format="cnn",
    batch_size=32,
    shuffle=True,
    transform=transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
)

for images in train_loader:
    # images: torch.Tensor [batch, 1, 64, 64]
    pass

# GNN
train_loader = get_dataloader(
    "data/processed/processed_XXX/datasets/gnn/train.json",
    format="gnn",
    batch_size=16,
    shuffle=True
)

for graphs in train_loader:
    # graphs: list of dicts avec 'nodes', 'edges', 'node_features'
    pass
```

#### Dataset temporel

```python
from utils.datasets import TemporalDiseaseDataset

dataset = TemporalDiseaseDataset(
    "data/processed/processed_XXX/datasets/cnn/train_seq5.json",
    format="cnn"
)

for sequence, target, metadata in dataset:
    # sequence: list de 5 tensors [1, 64, 64]
    # target: tensor [1, 64, 64]
    pass
```

---

## 📋 Pipeline Complète - Exemple

### Générer 100 simulations pour production

```bash
# 1. Génération
python scripts/generate_data.py \
    --num_simulations 100 \
    --grid_size 100 \
    --timesteps 200 \
    --model_type epidemic \
    --p_transmission 0.25 \
    --infection_duration 7 \
    --seed 42

# 2. Prétraitement (optimisé pour comparaison CNN/GNN)
python scripts/preprocess_data.py \
    --input data/simulations/generation_TIMESTAMP \
    --output data/processed \
    --name disease_dataset_v1 \
    --target_size 128 \
    --crop \
    --crop_margin 10 \
    --normalize \
    --add_spatial_features \
    --formats cnn gnn

# 3. Création datasets
python scripts/create_datasets.py \
    --input data/processed/disease_dataset_v1 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --stratify_by simulation \
    --seed 42

# 4. Entraînement (TODO)
python scripts/train_cnn.py --dataset data/processed/disease_dataset_v1/datasets/cnn
python scripts/train_gnn.py --dataset data/processed/disease_dataset_v1/datasets/gnn
```

---

## 🎯 Stratégies de Comparaison CNN vs GNN

### Garantir la comparabilité

Pour comparer équitablement CNN et GNN:

1. **Même source de données**
   ```bash
   --formats cnn gnn  # Générer les deux simultanément
   ```

2. **Même split train/val/test**
   ```bash
   --stratify_by simulation --seed 42  # Split identique
   ```

3. **Normalisation cohérente**
   ```bash
   --normalize  # Même normalisation pour CNN et GNN
   ```

4. **Vérifier les statistiques**
   ```json
   // preprocessing_metadata.json
   "statistics": {
     "cnn": {"mean": 0.758, "std": 0.339},
     "gnn": {"avg_nodes": 900, "feature_dim": 3}
   }
   ```

### Métriques à comparer

- Précision de détection
- Temps d'inférence
- Robustesse au bruit
- Généralisation (test set)
- Efficacité mémoire

---

## 🔍 Vérification et Debug

### Vérifier le prétraitement

```python
import numpy as np
import matplotlib.pyplot as plt

# Charger un échantillon CNN
data = np.load("data/processed/processed_XXX/cnn/sim_0000/t_0010.npy", 
               allow_pickle=True).item()
plt.imshow(data['data'], cmap='viridis')
plt.title(f"Timestep {data['timestep']}")
plt.show()

# Charger un échantillon GNN
data = np.load("data/processed/processed_XXX/gnn/sim_0000/t_0010.npy",
               allow_pickle=True).item()
print(f"Nodes: {data['nodes'].shape}")
print(f"Edges: {data['edges'].shape}")
print(f"Features: {data['node_features'].shape}")
```

### Vérifier les splits

```python
import json

with open("data/processed/processed_XXX/datasets/cnn/train.json") as f:
    train = json.load(f)

print(f"Train samples: {train['num_samples']}")
print(f"Simulations: {set(s['sim_id'] for s in train['samples'])}")
print(f"Timestep range: [{min(s['timestep'] for s in train['samples'])}, "
      f"{max(s['timestep'] for s in train['samples'])}]")
```

---

## 📁 Structure Finale

```
data/
├── simulations/                    # Données brutes
│   └── generation_20260204_173051/
│       ├── generation_metadata.json
│       └── sim_XXXX/
│           ├── metadata.json
│           ├── images/
│           └── graphs/
│
└── processed/                      # Données preprocessées
    └── processed_20260209_165358/
        ├── preprocessing_metadata.json
        ├── cnn/
        │   └── sim_XXXX/
        │       └── t_XXXX.npy      # {'data': array, 'crop_bbox': ...}
        ├── gnn/
        │   └── sim_XXXX/
        │       └── t_XXXX.npy      # {'nodes': ..., 'edges': ..., ...}
        └── datasets/
            ├── dataset_metadata.json
            ├── cnn/
            │   ├── train.json      # Splits pour CNN
            │   ├── val.json
            │   └── test.json
            └── gnn/
                ├── train.json      # Splits pour GNN
                ├── val.json
                └── test.json
```

---

## ⚡ Commandes Rapides

```bash
# Pipeline complète en 3 commandes
python scripts/generate_data.py --num_simulations 50 --grid_size 100 --timesteps 200
python scripts/preprocess_data.py --input data/simulations/generation_XXX --crop --normalize --add_spatial_features
python scripts/create_datasets.py --input data/processed/processed_XXX --stratify_by simulation

# Vérifier les résultats
ls data/processed/processed_XXX/datasets/cnn/
ls data/processed/processed_XXX/datasets/gnn/
```

---

## 📚 Ressources

- **Scripts:** [scripts/](../scripts/)
- **Utils:** [utils/datasets.py](../utils/datasets.py)
- **Exemples:** [test/](../test/)
