#!/bin/bash

# Script de test de la pipeline complete
# Execute une demonstration complete: generation -> preprocessing -> datasets

set -e  # Arreter si erreur

echo "======================================================================"
echo "  TEST PIPELINE - Demo Complete"
echo "======================================================================"

# Configuration
NUM_SIM=2
GRID_SIZE=30
TIMESTEPS=10
TARGET_SIZE=64

echo ""
echo "[1/3] Generation de simulations..."
python scripts/generate_data.py \
    --num_simulations $NUM_SIM \
    --grid_size $GRID_SIZE \
    --timesteps $TIMESTEPS \
    --seed 42 \
    --output_dir data/simulations

# Trouver la derniere generation
LATEST_GEN=$(ls -t data/simulations/ | grep "generation_" | head -1)
echo "Generation creee: $LATEST_GEN"

echo ""
echo "[2/3] Preprocessing des donnees..."
python scripts/preprocess_data.py \
    --input data/simulations/$LATEST_GEN \
    --output data/processed \
    --target_size $TARGET_SIZE \
    --normalize \
    --crop \
    --add_spatial_features

# Trouver le dernier processed
LATEST_PROC=$(ls -t data/processed/ | grep "processed_" | head -1)
echo "Preprocessing cree: $LATEST_PROC"

echo ""
echo "[3/3] Creation des datasets PyTorch..."
python scripts/create_datasets.py \
    --input data/processed/$LATEST_PROC \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --stratify_by simulation \
    --seed 42

echo ""
echo "======================================================================"
echo "  TEST TERMINE"
echo "======================================================================"
echo ""
echo "Structure generee:"
echo "  data/simulations/$LATEST_GEN/"
echo "  data/processed/$LATEST_PROC/"
echo "    ├── cnn/"
echo "    ├── gnn/"
echo "    └── datasets/"
echo "        ├── cnn/ (train.json, val.json, test.json)"
echo "        └── gnn/ (train.json, val.json, test.json)"
echo ""
echo "Pipeline prete pour entrainement!"
echo "======================================================================"
