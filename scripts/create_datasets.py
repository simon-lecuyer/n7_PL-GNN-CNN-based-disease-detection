#!/usr/bin/env python3
"""
Script de création de datasets PyTorch avec split train/val/test.

Ce script prend les données preprocessées et crée des datasets PyTorch
prêts pour l'entraînement avec split stratifié et reproductible.

Usage:
    python scripts/create_datasets.py \\
        --input data/processed/processed_20260204_173051 \\
        --output data/processed/processed_20260204_173051/datasets \\
        --train_ratio 0.7 \\
        --val_ratio 0.15 \\
        --test_ratio 0.15
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Création de datasets PyTorch avec split train/val/test"
    )
    
    # Paramètres d'entrée/sortie
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Dossier de données preprocessées (ex: data/processed/processed_XXX)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Dossier de sortie (défaut: <input>/datasets)"
    )
    
    # Paramètres de split
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio train (défaut: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Ratio validation (défaut: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Ratio test (défaut: 0.15)"
    )
    
    # Paramètres de stratification
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="simulation",
        choices=["simulation", "infection_level", "timestep", "none"],
        help="Stratification (défaut: simulation)"
    )
    parser.add_argument(
        "--min_samples_per_split",
        type=int,
        default=10,
        help="Minimum d'échantillons par split (défaut: 10)"
    )
    
    # Séquences temporelles
    parser.add_argument(
        "--create_sequences",
        action="store_true",
        help="Créer des séquences temporelles"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=5,
        help="Longueur des séquences (défaut: 5)"
    )
    parser.add_argument(
        "--sequence_stride",
        type=int,
        default=1,
        help="Stride entre séquences (défaut: 1)"
    )
    
    # Options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed aléatoire (défaut: 42)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=None,
        help="Formats à traiter (défaut: tous disponibles)"
    )
    
    return parser.parse_args()


def load_preprocessing_metadata(input_dir):
    """Charge les métadonnées de prétraitement."""
    meta_file = Path(input_dir) / "preprocessing_metadata.json"
    with open(meta_file, "r") as f:
        return json.load(f)


def collect_samples_unified(input_dir):
    """
    Collecte les paires (sim_id, timestep) disponibles (format-agnostic).
    
    Returns:
        List de tuples (sim_id, timestep, infection_level)
    """
    # Utiliser CNN comme référence pour identifier les paires disponibles
    cnn_dir = Path(input_dir) / "cnn"
    
    if not cnn_dir.exists():
        return []
    
    pairs = []
    
    # Parcourir toutes les simulations
    for sim_dir in sorted(cnn_dir.glob("sim_*")):
        sim_id = int(sim_dir.name.split("_")[1])
        
        # Parcourir tous les timesteps
        for sample_file in sorted(sim_dir.glob("t_*.npy")):
            timestep = int(sample_file.stem.split("_")[1])
            
            # Charger pour obtenir infection level (après inversion)
            data = np.load(sample_file, allow_pickle=True).item()
            infection_level = float(data["data"].mean())
            
            pairs.append((sim_id, timestep, infection_level))
    
    return pairs


def collect_samples(input_dir, format_type):
    """
    Collecte tous les échantillons pour un format donné.
    
    Returns:
        List de dicts avec 'file', 'sim_id', 'timestep', 'infection_level'
    """
    format_dir = Path(input_dir) / format_type
    
    if not format_dir.exists():
        return []
    
    samples = []
    
    # Parcourir toutes les simulations
    for sim_dir in sorted(format_dir.glob("sim_*")):
        sim_id = int(sim_dir.name.split("_")[1])
        
        # Parcourir tous les timesteps
        for sample_file in sorted(sim_dir.glob("t_*.npy")):
            timestep = int(sample_file.stem.split("_")[1])
            
            # Charger pour obtenir infection level (après inversion sémantique)
            data = np.load(sample_file, allow_pickle=True).item()
            
            if format_type == "cnn":
                infection_level = float(data["data"].mean())
            else:  # gnn
                infection_level = float(data["node_features"][:, 0].mean())
            
            samples.append({
                "file": str(sample_file),
                "sim_id": sim_id,
                "timestep": timestep,
                "infection_level": infection_level
            })
    
    return samples


def stratify_samples(samples, stratify_by):
    """
    Crée des labels de stratification.
    
    Returns:
        Array de labels pour stratification
    """
    if stratify_by == "simulation":
        return np.array([s["sim_id"] for s in samples])
    
    elif stratify_by == "infection_level":
        # Binning des niveaux d'infection
        levels = np.array([s["infection_level"] for s in samples])
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
        return np.digitize(levels, bins)
    
    elif stratify_by == "timestep":
        # Binning des timesteps (début, milieu, fin)
        timesteps = np.array([s["timestep"] for s in samples])
        max_t = timesteps.max()
        bins = [0, max_t // 3, 2 * max_t // 3, max_t]
        return np.digitize(timesteps, bins)
    
    else:  # none
        return None


def create_temporal_sequences(samples, sequence_length, stride):
    """
    Crée des séquences temporelles à partir des échantillons.
    
    Returns:
        List de séquences, chaque séquence = list d'échantillons
    """
    # Grouper par simulation
    sim_samples = {}
    for sample in samples:
        sim_id = sample["sim_id"]
        if sim_id not in sim_samples:
            sim_samples[sim_id] = []
        sim_samples[sim_id].append(sample)
    
    # Trier par timestep dans chaque simulation
    for sim_id in sim_samples:
        sim_samples[sim_id] = sorted(sim_samples[sim_id], key=lambda x: x["timestep"])
    
    # Créer les séquences
    sequences = []
    for sim_id, samples_list in sim_samples.items():
        for i in range(0, len(samples_list) - sequence_length, stride):
            seq = samples_list[i:i + sequence_length]
            target = samples_list[i + sequence_length] if i + sequence_length < len(samples_list) else None
            
            if target:
                sequences.append({
                    "sequence": seq,
                    "target": target,
                    "sim_id": sim_id,
                    "start_timestep": seq[0]["timestep"]
                })
    
    return sequences


def split_data(samples, train_ratio, val_ratio, test_ratio, stratify_labels, seed, min_samples):
    """
    Split les données en train/val/test.
    
    Returns:
        train_samples, val_samples, test_samples
    """
    # Vérifier ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios doivent sommer à 1.0"
    
    n_samples = len(samples)
    
    # Vérifier minimum
    if n_samples < min_samples * 3:
        print(f"⚠️  Nombre d'échantillons insuffisant ({n_samples}), split non-optimal")
    
    # Premier split: train vs (val + test)
    if stratify_labels is not None:
        train_idx, temp_idx = train_test_split(
            np.arange(n_samples),
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=stratify_labels
        )
        
        # Second split: val vs test
        temp_labels = stratify_labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
            stratify=temp_labels
        )
    else:
        train_idx, temp_idx = train_test_split(
            np.arange(n_samples),
            test_size=(val_ratio + test_ratio),
            random_state=seed
        )
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed
        )
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    return train_samples, val_samples, test_samples


def save_split(samples, output_file):
    """Sauvegarde un split dans un fichier."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder en JSON (chemins) + pickle (optionnel)
    split_data = {
        "samples": samples,
        "num_samples": len(samples)
    }
    
    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=2)
    
    # Aussi sauver en pickle pour chargement rapide
    pickle_file = output_file.with_suffix(".pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump(split_data, f)


def compute_split_statistics(train, val, test):
    """Calcule des statistiques sur les splits."""
    stats = {
        "train": {
            "num_samples": len(train),
            "num_simulations": len(set(s["sim_id"] for s in train)),
            "avg_infection": np.mean([s["infection_level"] for s in train]),
            "timestep_range": [
                min(s["timestep"] for s in train),
                max(s["timestep"] for s in train)
            ]
        },
        "val": {
            "num_samples": len(val),
            "num_simulations": len(set(s["sim_id"] for s in val)),
            "avg_infection": np.mean([s["infection_level"] for s in val]),
            "timestep_range": [
                min(s["timestep"] for s in val),
                max(s["timestep"] for s in val)
            ]
        },
        "test": {
            "num_samples": len(test),
            "num_simulations": len(set(s["sim_id"] for s in test)),
            "avg_infection": np.mean([s["infection_level"] for s in test]),
            "timestep_range": [
                min(s["timestep"] for s in test),
                max(s["timestep"] for s in test)
            ]
        }
    }
    
    return stats


def main():
    """Fonction principale."""
    args = parse_args()
    
    input_dir = Path(args.input)
    
    if args.output is None:
        output_dir = input_dir / "datasets"
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger métadonnées
    try:
        preprocess_metadata = load_preprocessing_metadata(input_dir)
    except Exception as e:
        print(f"❌ Impossible de charger les métadonnées: {e}")
        return
    
    # Déterminer formats disponibles
    if args.formats is None:
        formats = []
        if (input_dir / "cnn").exists():
            formats.append("cnn")
        if (input_dir / "gnn").exists():
            formats.append("gnn")
    else:
        formats = args.formats
    
    print(f"{'='*70}")
    print(f"  CRÉATION DES DATASETS PYTORCH")
    print(f"{'='*70}")
    print(f"  Entree: {input_dir}")
    print(f"  Sortie: {output_dir}")
    print(f"  Formats: {', '.join(formats)}")
    print(f"  Split: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}")
    print(f"  Stratification: {args.stratify_by}")
    print(f"  Seed: {args.seed}")
    if args.create_sequences:
        print(f"  Sequences: longueur={args.sequence_length}, stride={args.sequence_stride}")
    print(f"{'='*70}\n")
    
    # Métadonnées du dataset
    dataset_metadata = {
        "source_preprocessing": str(input_dir),
        "split_params": vars(args),
        "formats": {}
    }
    
    # Faire le split UNE SEULE FOIS sur les paires (sim_id, timestep)
    print("\nCollecte des paires (sim_id, timestep) unifiées...")
    pairs = collect_samples_unified(input_dir)
    
    if not pairs:
        print("Erreur: Aucune paire trouvée!")
        return
    
    print(f"  {len(pairs)} paires identifiées")
    
    # Créer les indices pour le split
    all_indices = np.arange(len(pairs))
    pairs_array = np.array(pairs, dtype=object)
    
    # Stratification
    if args.stratify_by == "simulation":
        stratify_labels = pairs_array[:, 0].astype(int)  # sim_id
    elif args.stratify_by == "infection_level":
        levels = pairs_array[:, 2].astype(float)
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
        stratify_labels = np.digitize(levels, bins)
    else:
        stratify_labels = None
    
    # Split une seule fois
    print("Split unifié des données...")
    if stratify_labels is not None:
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=(args.val_ratio + args.test_ratio),
            random_state=args.seed,
            stratify=stratify_labels
        )
        temp_labels = stratify_labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
            random_state=args.seed,
            stratify=temp_labels
        )
    else:
        train_idx, temp_idx = train_test_split(
            all_indices,
            test_size=(args.val_ratio + args.test_ratio),
            random_state=args.seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
            random_state=args.seed
        )
    
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    
    print(f"  Train: {len(train_pairs)} paires")
    print(f"  Val: {len(val_pairs)} paires")
    print(f"  Test: {len(test_pairs)} paires")
    
    # Traiter chaque format en appliquant le même split
    for fmt in formats:
        print(f"\n{'='*50}")
        print(f"  Format: {fmt.upper()}")
        print(f"{'='*50}")
        
        # Collecter tous les échantillons
        print("Collecte des échantillons...")
        all_samples = collect_samples(input_dir, fmt)
        
        if not all_samples:
            print(f"Attention: Aucun échantillon trouvé pour {fmt}")
            continue
        
        # Créer un mapping (sim_id, timestep) -> sample
        sample_map = {(s["sim_id"], s["timestep"]): s for s in all_samples}
        
        # Appliquer le split unifié
        train = [sample_map[(sim_id, timestep)] for sim_id, timestep, _ in train_pairs if (sim_id, timestep) in sample_map]
        val = [sample_map[(sim_id, timestep)] for sim_id, timestep, _ in val_pairs if (sim_id, timestep) in sample_map]
        test = [sample_map[(sim_id, timestep)] for sim_id, timestep, _ in test_pairs if (sim_id, timestep) in sample_map]
        
        print(f"  Train: {len(train)} échantillons")
        print(f"  Val: {len(val)} échantillons")
        print(f"  Test: {len(test)} échantillons")
        
        # Sauvegarder splits
        print("Sauvegarde des splits...")
        fmt_dir = output_dir / fmt
        
        if args.create_sequences:
            suffix = f"_seq{args.sequence_length}"
        else:
            suffix = ""
        
        save_split(train, fmt_dir / f"train{suffix}.json")
        save_split(val, fmt_dir / f"val{suffix}.json")
        save_split(test, fmt_dir / f"test{suffix}.json")
        
        print(f"  Splits sauvegardes dans {fmt_dir}")
        
        # Statistiques
        if not args.create_sequences:
            stats = compute_split_statistics(train, val, test)
            dataset_metadata["formats"][fmt] = stats
            
            print("\nStatistiques:")
            for split_name, split_stats in stats.items():
                print(f"  {split_name}:")
                for key, val in split_stats.items():
                    print(f"    {key}: {val}")
    
    # Sauvegarder métadonnées globales
    meta_file = output_dir / "dataset_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  DATASETS CRÉÉS")
    print(f"{'='*70}")
    print(f"  Métadonnées: {meta_file}")
    print(f"  Splits disponibles:")
    for fmt in formats:
        print(f"    - {fmt}/train.json, val.json, test.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
