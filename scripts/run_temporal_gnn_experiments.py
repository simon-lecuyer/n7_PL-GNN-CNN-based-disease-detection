#!/usr/bin/env python3
"""
Script pour générer les séquences temporelles et entraîner les GNN temporels.
Pour chaque expérience (slow, medium, fast, etc.) et chaque sequence_length (3, 5, 10).
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from utils.datasets import get_dataloader
from gnn.models.temporal_disease_gnn import TemporalDiseaseGNN

# Configuration des expériences
EXPERIMENTS = {
    "slow_spread_short_infection": "data/processed/slow_spread_short_infection_exp",
    "medium_spread_medium_infection": "data/processed/medium_spread_medium_infection_exp/processed_20260306_171228",
    "fast_spread_long_infection": "data/processed/fast_spread_long_infection_exp/processed_20260306_153410",
    "very_slow_isolated": "data/processed/very_slow_isolated_exp/processed_20260306_142046",
    "fast_multi_source": "data/processed/processed_20260309_203115"
}

SEQUENCE_LENGTHS = [3, 5, 10]


def run_command(cmd, description):
    """Exécute une commande shell."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    
    if result.returncode != 0:
        print(f"❌ ERREUR lors de: {description}")
        print(f"   Exit code: {result.returncode}")
        return False
    else:
        print(f"✅ Succès: {description}")
        return True


def check_sequence_exists(data_path, seq_length):
    """Vérifie si les fichiers de séquence existent déjà."""
    gnn_dir = Path(data_path) / "datasets" / "gnn"
    train_seq = gnn_dir / f"train_seq{seq_length}.json"
    val_seq = gnn_dir / f"val_seq{seq_length}.json"
    test_seq = gnn_dir / f"test_seq{seq_length}.json"
    
    return train_seq.exists() and val_seq.exists() and test_seq.exists()


def create_gnn_config(exp_name, data_path, seq_length):
    """Crée un fichier de configuration YAML pour un entraînement GNN."""
    config = {
        "experiment_name": f"gnn_{exp_name}_seq{seq_length}_ep10",
        "data": {
            "train_path": f"{data_path}/datasets/gnn/train_seq{seq_length}.json",
            "val_path": f"{data_path}/datasets/gnn/val_seq{seq_length}.json",
            "test_path": f"{data_path}/datasets/gnn/test_seq{seq_length}.json",
            "checkpoint_path": "gnn/checkpoints",
            "log_path": "gnn/training_logs"
        },
        "model": {
            "in_channels": 3,  # Pour temporal
            "hidden_dim": 64,
            "num_layers": 2,
            "out_channels": 1,
            "dropout": 0.2,
            "type": "temporal"
        },
        "training": {
            "epochs": 10,
            "batch_size": 16,
            "lr": 0.001,
            "optimizer": "adam",
            "device": "auto",
            "seed": 42
        }
    }
    
    config_path = f"configs/gnn_{exp_name}_seq{seq_length}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


def main():
    print("="*80)
    print("GÉNÉRATION DES SÉQUENCES TEMPORELLES ET ENTRAÎNEMENT DES GNN TEMPORELS")
    print("="*80)
    print(f"\nExpériences: {list(EXPERIMENTS.keys())}")
    print(f"Sequence lengths: {SEQUENCE_LENGTHS}")
    print(f"Total runs: {len(EXPERIMENTS) * len(SEQUENCE_LENGTHS)} = {len(EXPERIMENTS)} × {len(SEQUENCE_LENGTHS)}")
    
    # Activation du venv
    venv_activate = "source venv/bin/activate"
    
    results = []
    
    for exp_name, data_path in EXPERIMENTS.items():
        for seq_length in SEQUENCE_LENGTHS:
            print(f"\n\n{'#'*80}")
            print(f"# {exp_name.upper()} - SEQUENCE LENGTH {seq_length}")
            print(f"{'#'*80}")
            
            # 1. Vérifier si les séquences existent déjà
            if check_sequence_exists(data_path, seq_length):
                print(f"✓ Séquences seq{seq_length} existent déjà pour {exp_name}")
            else:
                # Générer les séquences
                cmd_generate = (
                    f"{venv_activate} && "
                    f"python scripts/create_datasets.py "
                    f"--input {data_path} "
                    f"--formats gnn "
                    f"--create_sequences "
                    f"--sequence_length {seq_length} "
                    f"--sequence_stride 1"
                )
                
                success = run_command(
                    cmd_generate,
                    f"[1/3] Génération séquences seq{seq_length} pour {exp_name}"
                )
                
                if not success:
                    print(f"⚠️  Échec génération pour {exp_name} seq{seq_length}, passage au suivant...")
                    results.append((exp_name, seq_length, "FAILED_GENERATION"))
                    continue
            
            # 2. Créer le fichier de configuration
            config_path = create_gnn_config(exp_name, data_path, seq_length)
            print(f"✓ Configuration créée: {config_path}")
            
            # 3. Entraîner le modèle
            cmd_train = (
                f"{venv_activate} && "
                f"python scripts/train_gnn.py --cfg {config_path}"
            )
            
            success = run_command(
                cmd_train,
                f"[2/3] Entraînement GNN temporel {exp_name} seq{seq_length}"
            )
            
            if success:
                results.append((exp_name, seq_length, "SUCCESS"))
            else:
                results.append((exp_name, seq_length, "FAILED_TRAINING"))
    
    # Résumé final
    print("\n\n" + "="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    success_count = sum(1 for _, _, status in results if status == "SUCCESS")
    total_count = len(results)
    
    for exp_name, seq_length, status in results:
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"{emoji} {exp_name:40s} seq{seq_length:2d} : {status}")
    
    print(f"\n{success_count}/{total_count} entraînements réussis")
    print("="*80)


if __name__ == "__main__":
    main()
