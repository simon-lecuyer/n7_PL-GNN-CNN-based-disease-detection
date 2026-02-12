#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  DEMO - GNN-CNN Based Disease Detection in Agriculture
#  Projet Long N7 - Février 2026
#  Équipe : Myriam ROBBANA, Mihai COSTIN, Simon LECUYER,
#           Assala ASSELALOU, Yassin MOUKAN
# ═══════════════════════════════════════════════════════════════════════════
#
# Ce script démontre les fonctionnalités implémentées à ce jour :
#   1. Génération de données de simulation (WaterberryFarms)
#   2. Prétraitement des données (crop, normalisation, redimensionnement)
#   3. Création de datasets PyTorch (split train/val/test stratifié)
#   4. Visualisation des datasets (échantillons, évolution, stats, comparaison)
#   5. Chargement PyTorch Dataset (vérification de l'intégration)
#
# Usage:
#   chmod +x demo.sh
#   ./demo.sh              # Démo complète
#   ./demo.sh --skip-gen   # Sauter la génération (réutiliser données existantes)
#   ./demo.sh --help       # Afficher l'aide
#
# ═══════════════════════════════════════════════════════════════════════════

set -e

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_SIM=2
GRID_SIZE=30
TIMESTEPS=10
TARGET_SIZE=64
SEED=42
DEMO_OUTPUT_DIR="results/figures/demo"
SKIP_GEN=false

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# ──────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} ${BOLD}$1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

step() {
    echo ""
    echo -e "  ${BLUE}▶${NC} ${BOLD}$1${NC}"
    echo -e "  ${DIM}$2${NC}"
    echo ""
}

success() {
    echo -e "  ${GREEN}✔${NC} $1"
}

info() {
    echo -e "  ${DIM}ℹ${NC} $1"
}

warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "  ${RED}✖${NC} $1"
}

separator() {
    echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"
}

# ──────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --skip-gen)
            SKIP_GEN=true
            ;;
        --help|-h)
            echo "Usage: ./demo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-gen   Sauter la génération de données (réutiliser existantes)"
            echo "  --help       Afficher cette aide"
            echo ""
            echo "Cette démo exécute la pipeline complète avec des paramètres réduits"
            echo "pour une exécution rapide (~1-2 minutes)."
            exit 0
            ;;
    esac
done

# ══════════════════════════════════════════════════════════════
# INTRODUCTION
# ══════════════════════════════════════════════════════════════
clear 2>/dev/null || true

echo ""
echo -e "${CYAN}  ╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}  ║                                                            ║${NC}"
echo -e "${CYAN}  ║${NC}   ${BOLD}GNN-CNN Based Disease Detection in Agriculture${NC}         ${CYAN}║${NC}"
echo -e "${CYAN}  ║${NC}   ${DIM}Projet Long N7 - Démonstration${NC}                         ${CYAN}║${NC}"
echo -e "${CYAN}  ║                                                            ║${NC}"
echo -e "${CYAN}  ╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${DIM}Ce projet compare deux approches de détection de maladies${NC}"
echo -e "  ${DIM}végétales pour l'agriculture de précision :${NC}"
echo ""
echo -e "  ${BOLD}CNN${NC} - Détection par images (drone)   ${DIM}→ Classification visuelle${NC}"
echo -e "  ${BOLD}GNN${NC} - Modélisation spatio-temporelle ${DIM}→ Graphe de propagation${NC}"
echo ""
separator
echo ""
echo -e "  ${BOLD}Composants implémentés :${NC}"
echo -e "  ${GREEN}✔${NC} Génération de données      ${DIM}(scripts/generate_data.py)${NC}"
echo -e "  ${GREEN}✔${NC} Prétraitement              ${DIM}(scripts/preprocess_data.py)${NC}"
echo -e "  ${GREEN}✔${NC} Création de datasets        ${DIM}(scripts/create_datasets.py)${NC}"
echo -e "  ${GREEN}✔${NC} Visualisation               ${DIM}(scripts/visualize_dataset.py)${NC}"
echo -e "  ${GREEN}✔${NC} PyTorch Datasets classes     ${DIM}(utils/datasets.py)${NC}"
echo ""
echo -e "  ${BOLD}À venir :${NC}"
echo -e "  ${YELLOW}○${NC} Modèle CNN                  ${DIM}(cnn/models/)${NC}"
echo -e "  ${YELLOW}○${NC} Modèle GNN                  ${DIM}(gnn/models/)${NC}"
echo -e "  ${YELLOW}○${NC} Script d'entraînement       ${DIM}(scripts/train_cnn.py, train_gnn.py)${NC}"
echo -e "  ${YELLOW}○${NC} Évaluation comparative      ${DIM}(scripts/evaluate.py)${NC}"
echo -e "  ${YELLOW}○${NC} Intégration CGP             ${DIM}(utils/cgp/, scripts/run_cgp.py)${NC}"
echo ""
separator
echo -e "  ${DIM}Paramètres de démo : ${NUM_SIM} simulations, grille ${GRID_SIZE}×${GRID_SIZE}, ${TIMESTEPS} timesteps${NC}"
echo ""
echo -e "  ${BOLD}Appuyez sur Entrée pour démarrer...${NC}"
read -r

# ══════════════════════════════════════════════════════════════
# VÉRIFICATIONS PRÉALABLES
# ══════════════════════════════════════════════════════════════
banner "0/5  Vérifications préalables"

step "Vérification de Python et des dépendances" \
     "python3, numpy, matplotlib, scikit-learn, torch, PIL..."

# Vérifier Python
if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    success "Python trouvé : $PY_VERSION"
else
    error "Python3 non trouvé. Installez Python >= 3.8"
    exit 1
fi

# Vérifier les imports critiques
python3 -c "
import sys
modules = {
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'tqdm': 'tqdm',
}
missing = []
for mod, pkg in modules.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'MISSING:{\";\".join(missing)}')
    sys.exit(1)
else:
    print('OK')
" 2>/dev/null && success "Dépendances Python OK" || {
    error "Dépendances manquantes. Lancez: pip install -r requirements.txt"
    exit 1
}

# Vérifier PyTorch (optionnel pour la démo pipeline)
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null \
    && success "PyTorch disponible : $(python3 -c 'import torch; print(torch.__version__)')" \
    || warning "PyTorch non installé (optionnel pour la démo pipeline)"

# Vérifier WaterberryFarms
WBF_PATH="$SCRIPT_DIR/../WaterberryFarms"
if [ -d "$WBF_PATH" ]; then
    success "WaterberryFarms trouvé : $(realpath "$WBF_PATH")"
else
    if [ "$SKIP_GEN" = true ]; then
        warning "WaterberryFarms introuvable (OK car --skip-gen)"
    else
        error "WaterberryFarms introuvable à : $WBF_PATH"
        error "Clonez-le ou utilisez --skip-gen avec des données existantes"
        exit 1
    fi
fi

# ══════════════════════════════════════════════════════════════
# ÉTAPE 1 : GÉNÉRATION DE DONNÉES
# ══════════════════════════════════════════════════════════════
banner "1/5  Génération de données de simulation"

if [ "$SKIP_GEN" = true ]; then
    warning "Génération sautée (--skip-gen)"
    
    # Trouver la dernière génération existante
    LATEST_GEN=$(ls -t data/simulations/ 2>/dev/null | grep "generation_" | head -1)
    if [ -z "$LATEST_GEN" ]; then
        error "Aucune génération trouvée dans data/simulations/"
        error "Relancez sans --skip-gen pour générer des données"
        exit 1
    fi
    success "Utilisation de la génération existante : $LATEST_GEN"
else
    step "Exécution de generate_data.py" \
         "Simule la propagation de maladies végétales via WaterberryFarms (modèle SIR)"

    info "Commande :"
    echo -e "  ${DIM}python3 scripts/generate_data.py \\${NC}"
    echo -e "  ${DIM}    --num_simulations $NUM_SIM --grid_size $GRID_SIZE \\${NC}"
    echo -e "  ${DIM}    --timesteps $TIMESTEPS --seed $SEED \\${NC}"
    echo -e "  ${DIM}    --model_type epidemic --p_transmission 0.2${NC}"
    echo ""

    python3 scripts/generate_data.py \
        --num_simulations $NUM_SIM \
        --grid_size $GRID_SIZE \
        --timesteps $TIMESTEPS \
        --seed $SEED \
        --model_type epidemic \
        --p_transmission 0.2 \
        --output_dir data/simulations

    LATEST_GEN=$(ls -t data/simulations/ | grep "generation_" | head -1)
    success "Génération terminée : $LATEST_GEN"
fi

GEN_DIR="data/simulations/$LATEST_GEN"

# Afficher la structure générée
echo ""
info "Structure des données générées :"
echo -e "  ${DIM}$GEN_DIR/${NC}"
for sim_dir in $(ls -d $GEN_DIR/sim_* 2>/dev/null | head -3); do
    sim_name=$(basename "$sim_dir")
    n_graphs=$(ls "$sim_dir/graphs/"*.npy 2>/dev/null | wc -l | tr -d ' ')
    n_images=$(ls "$sim_dir/images/"*.png 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  ${DIM}├── $sim_name/  →  $n_graphs graphes, $n_images images${NC}"
done

# Afficher les métadonnées de génération
if [ -f "$GEN_DIR/generation_metadata.json" ]; then
    echo ""
    info "Métadonnées de génération :"
    python3 -c "
import json
with open('$GEN_DIR/generation_metadata.json') as f:
    m = json.load(f)
print(f'  Modèle        : {m.get(\"model_type\", \"N/A\")}')
print(f'  Grille         : {m.get(\"grid_size\", \"N/A\")}x{m.get(\"grid_size\", \"N/A\")}')
print(f'  Simulations    : {m.get(\"num_simulations\", \"N/A\")}')
print(f'  Timesteps      : {m.get(\"timesteps\", \"N/A\")}')
print(f'  Seed           : {m.get(\"seed\", \"N/A\")}')
print(f'  Transmission   : {m.get(\"p_transmission\", m.get(\"parameters\", {}).get(\"p_transmission\", \"N/A\"))}')
"
fi

# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 : PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════
banner "2/5  Prétraitement des données"

step "Exécution de preprocess_data.py" \
     "Crop sur zones infectées, resize ${TARGET_SIZE}×${TARGET_SIZE}, normalisation [0,1], features spatiales GNN"

info "Commande :"
echo -e "  ${DIM}python3 scripts/preprocess_data.py \\${NC}"
echo -e "  ${DIM}    --input $GEN_DIR \\${NC}"
echo -e "  ${DIM}    --target_size $TARGET_SIZE --normalize --crop \\${NC}"
echo -e "  ${DIM}    --add_spatial_features --formats cnn gnn${NC}"
echo ""

python3 scripts/preprocess_data.py \
    --input "$GEN_DIR" \
    --output data/processed \
    --target_size $TARGET_SIZE \
    --normalize \
    --crop \
    --add_spatial_features \
    --formats cnn gnn

LATEST_PROC=$(ls -t data/processed/ | grep "processed_" | head -1)
PROC_DIR="data/processed/$LATEST_PROC"
success "Prétraitement terminé : $LATEST_PROC"

# Afficher les statistiques
echo ""
info "Données preprocessées :"
n_cnn=$(find "$PROC_DIR/cnn" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ')
n_gnn=$(find "$PROC_DIR/gnn" -name "*.npy" 2>/dev/null | wc -l | tr -d ' ')
echo -e "  CNN : ${BOLD}$n_cnn${NC} échantillons (images ${TARGET_SIZE}×${TARGET_SIZE})"
echo -e "  GNN : ${BOLD}$n_gnn${NC} échantillons (graphes avec features spatiales)"

# Afficher un aperçu des données preprocessées
python3 -c "
import numpy as np, json
from pathlib import Path

proc_dir = Path('$PROC_DIR')

# CNN sample
cnn_files = sorted(proc_dir.glob('cnn/sim_*/t_*.npy'))
if cnn_files:
    d = np.load(cnn_files[0], allow_pickle=True).item()
    print(f'  CNN sample shape : {d[\"data\"].shape}')
    print(f'  CNN value range  : [{d[\"data\"].min():.3f}, {d[\"data\"].max():.3f}]')

# GNN sample
gnn_files = sorted(proc_dir.glob('gnn/sim_*/t_*.npy'))
if gnn_files:
    d = np.load(gnn_files[0], allow_pickle=True).item()
    nf = d['node_features']
    print(f'  GNN nodes        : {len(d[\"nodes\"])}')
    print(f'  GNN edges        : {len(d[\"edges\"])}')
    print(f'  GNN features/node: {nf.shape[1] if nf.ndim == 2 else 1}')
" 2>/dev/null || true

# ══════════════════════════════════════════════════════════════
# ÉTAPE 3 : CRÉATION DES DATASETS
# ══════════════════════════════════════════════════════════════
banner "3/5  Création des datasets PyTorch"

step "Exécution de create_datasets.py" \
     "Split train/val/test stratifié par simulation (70/15/15)"

info "Commande :"
echo -e "  ${DIM}python3 scripts/create_datasets.py \\${NC}"
echo -e "  ${DIM}    --input $PROC_DIR \\${NC}"
echo -e "  ${DIM}    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \\${NC}"
echo -e "  ${DIM}    --stratify_by simulation --seed $SEED${NC}"
echo ""

python3 scripts/create_datasets.py \
    --input "$PROC_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --stratify_by simulation \
    --seed $SEED

success "Datasets créés"

# Afficher les stats des splits
echo ""
info "Splits générés :"
for fmt in cnn gnn; do
    for split in train val test; do
        split_file="$PROC_DIR/datasets/$fmt/$split.json"
        if [ -f "$split_file" ]; then
            n=$(python3 -c "import json; d=json.load(open('$split_file')); print(len(d['samples']))")
            fmt_upper=$(echo "$fmt" | tr '[:lower:]' '[:upper:]')
            echo -e "  ${fmt_upper} ${split}  : ${BOLD}$n${NC} échantillons"
        fi
    done
    separator
done

# ══════════════════════════════════════════════════════════════
# ÉTAPE 4 : VISUALISATION
# ══════════════════════════════════════════════════════════════
banner "4/5  Visualisation des données"

mkdir -p "$DEMO_OUTPUT_DIR"

# 4a. Échantillons CNN
step "4a. Échantillons CNN aléatoires" \
     "Affiche des images de propagation de maladie preprocessées"

TRAIN_CNN="$PROC_DIR/datasets/cnn/train.json"
if [ -f "$TRAIN_CNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_CNN" \
        --format cnn \
        --mode samples \
        --num_samples 6 \
        --save "$DEMO_OUTPUT_DIR/demo_cnn_samples.png" \
        --no_show \
        --title "Démo - Échantillons CNN (train)" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_cnn_samples.png"
fi

# 4b. Évolution temporelle CNN
step "4b. Évolution temporelle (CNN)" \
     "Propagation de la maladie au cours du temps pour une simulation"

if [ -f "$TRAIN_CNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_CNN" \
        --format cnn \
        --mode temporal \
        --sim_id 0 \
        --save "$DEMO_OUTPUT_DIR/demo_cnn_temporal.png" \
        --no_show \
        --title "Démo - Propagation temporelle (sim 0)" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_cnn_temporal.png"
fi

# 4c. Échantillons GNN
step "4c. Échantillons GNN (graphes)" \
     "Mêmes données représentées sous forme de graphes spatiaux"

TRAIN_GNN="$PROC_DIR/datasets/gnn/train.json"
if [ -f "$TRAIN_GNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_GNN" \
        --format gnn \
        --mode samples \
        --num_samples 4 \
        --save "$DEMO_OUTPUT_DIR/demo_gnn_samples.png" \
        --no_show \
        --title "Démo - Échantillons GNN (train)" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_gnn_samples.png"
fi

# 4d. Statistiques du dataset
step "4d. Statistiques du dataset" \
     "Distributions des niveaux d'infection, timesteps, simulations"

if [ -f "$TRAIN_CNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_CNN" \
        --format cnn \
        --mode stats \
        --save "$DEMO_OUTPUT_DIR/demo_stats.png" \
        --no_show \
        --title "Démo - Statistiques du dataset" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_stats.png"
fi

# 4e. Comparaison CNN vs GNN
step "4e. Comparaison CNN vs GNN côte à côte" \
     "Même échantillon vu par les deux approches"

if [ -f "$TRAIN_CNN" ] && [ -f "$TRAIN_GNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_CNN" \
        --dataset_gnn "$TRAIN_GNN" \
        --mode compare \
        --sim_id 0 \
        --num_samples 3 \
        --save "$DEMO_OUTPUT_DIR/demo_cnn_vs_gnn.png" \
        --no_show \
        --title "Démo - CNN vs GNN (même donnée)" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_cnn_vs_gnn.png"
fi

# 4f. Grille simulation × timestep
step "4f. Grille simulation × timestep (CNN)" \
     "Vue matricielle de toutes les données"

if [ -f "$TRAIN_CNN" ]; then
    python3 scripts/visualize_dataset.py \
        --dataset "$TRAIN_CNN" \
        --format cnn \
        --mode grid \
        --save "$DEMO_OUTPUT_DIR/demo_grid.png" \
        --no_show \
        --title "Démo - Grille complète" 2>/dev/null
    success "Sauvegardé → $DEMO_OUTPUT_DIR/demo_grid.png"
fi

# ══════════════════════════════════════════════════════════════
# ÉTAPE 5 : VÉRIFICATION PYTORCH DATASET
# ══════════════════════════════════════════════════════════════
banner "5/5  Vérification PyTorch Dataset"

step "Chargement avec utils.datasets.DiseaseDetectionDataset" \
     "Vérifie que les datasets sont utilisables directement avec PyTorch"

python3 << PYEOF
import sys
sys.path.insert(0, '.')

try:
    import torch
    from utils.datasets import DiseaseDetectionDataset, TemporalDiseaseDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  ⚠  PyTorch non installé - vérification structurelle uniquement")

import json, numpy as np
from pathlib import Path

proc_dir = Path("$PROC_DIR")
print()

for fmt in ["cnn", "gnn"]:
    train_file = proc_dir / "datasets" / fmt / "train.json"
    if not train_file.exists():
        continue
    
    with open(train_file) as f:
        split_data = json.load(f)
    
    n_samples = len(split_data["samples"])
    
    if HAS_TORCH:
        # Charger via PyTorch Dataset
        dataset = DiseaseDetectionDataset(str(train_file), format=fmt, return_metadata=True)
        print(f"  {fmt.upper()} Dataset:")
        print(f"    Taille         : {len(dataset)} échantillons")
        
        # Charger le premier échantillon
        sample = dataset[0]
        if fmt == "cnn":
            data, meta = sample
            print(f"    Tensor shape   : {data.shape}  (C, H, W)")
            print(f"    dtype          : {data.dtype}")
            print(f"    Valeur range   : [{data.min():.3f}, {data.max():.3f}]")
        else:
            graph, meta = sample
            print(f"    Nodes          : {graph['nodes'].shape}")
            print(f"    Edges          : {graph['edges'].shape}")
            print(f"    Node features  : {graph['node_features'].shape}")
        
        print(f"    Metadata       : sim_id={meta['sim_id']}, t={meta['timestep']}, "
              f"infection={meta['infection_level']:.4f}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        if fmt == "cnn":
            loader = DataLoader(dataset, batch_size=min(4, n_samples), shuffle=True)
            batch = next(iter(loader))
            data_batch, meta_batch = batch
            print(f"    Batch test     : ✔ shape={data_batch.shape}")
        
        print(f"    Status         : ✔ Prêt pour l'entraînement")
    else:
        # Vérification sans PyTorch
        sample = split_data["samples"][0]
        data = np.load(sample["file"], allow_pickle=True).item()
        print(f"  {fmt.upper()} Dataset:")
        print(f"    Échantillons   : {n_samples}")
        if fmt == "cnn":
            print(f"    Data shape     : {data['data'].shape}")
        else:
            print(f"    Nodes          : {len(data['nodes'])}")
            print(f"    Edges          : {len(data['edges'])}")
        print(f"    Status         : ✔ Données valides (installer PyTorch pour DataLoader)")
    
    print()

PYEOF

success "Vérification PyTorch terminée"

# ══════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}  ╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}  ║${NC}                    ${BOLD}DÉMO TERMINÉE ✔${NC}                          ${CYAN}║${NC}"
echo -e "${CYAN}  ╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Pipeline de données${NC} : ${GREEN}Fonctionnelle${NC}"
echo ""
echo -e "  ${BOLD}Données générées :${NC}"
echo -e "    Simulations     → ${DIM}$GEN_DIR/${NC}"
echo -e "    Preprocessées   → ${DIM}$PROC_DIR/${NC}"
echo -e "    Datasets        → ${DIM}$PROC_DIR/datasets/{cnn,gnn}/{train,val,test}.json${NC}"
echo ""
echo -e "  ${BOLD}Figures de démo :${NC}"
for fig in "$DEMO_OUTPUT_DIR"/demo_*.png; do
    [ -f "$fig" ] && echo -e "    ${GREEN}✔${NC} $(basename "$fig")"
done
echo -e "    ${DIM}→ Dossier : $DEMO_OUTPUT_DIR/${NC}"
echo ""
echo -e "  ${BOLD}Prochaines étapes :${NC}"
echo -e "    ${YELLOW}1.${NC} Implémenter le modèle CNN     ${DIM}(cnn/models/disease_detector_cnn.py)${NC}"
echo -e "    ${YELLOW}2.${NC} Implémenter le modèle GNN     ${DIM}(gnn/models/spatio_temporal_gnn.py)${NC}"
echo -e "    ${YELLOW}3.${NC} Créer scripts d'entraînement  ${DIM}(scripts/train_cnn.py, train_gnn.py)${NC}"
echo -e "    ${YELLOW}4.${NC} Évaluation comparative        ${DIM}(scripts/evaluate.py)${NC}"
echo -e "    ${YELLOW}5.${NC} Intégration CGP               ${DIM}(utils/cgp/, scripts/run_cgp.py)${NC}"
echo ""
separator
echo -e "  ${DIM}Pour régénérer avec plus de données :${NC}"
echo -e "  ${DIM}  python3 scripts/generate_data.py --num_simulations 50 --grid_size 100 --timesteps 200${NC}"
echo ""
