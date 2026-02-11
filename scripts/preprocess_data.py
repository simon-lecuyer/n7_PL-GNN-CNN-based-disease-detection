#!/usr/bin/env python3
"""
Script de prétraitement des données de simulation pour CNN et GNN.

Ce script transforme les données brutes de simulation en données preprocessées
prêtes pour l'entraînement, en garantissant la comparabilité entre CNN et GNN.

Usage:
    python scripts/preprocess_data.py \\
        --input data/simulations/generation_20260204_173051 \\
        --output data/processed \\
        --target_size 64 \\
        --normalize
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prétraitement des données de simulation pour CNN et GNN"
    )
    
    # Paramètres d'entrée/sortie
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Dossier de génération en entrée (ex: data/simulations/generation_XXX)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Dossier de sortie (défaut: data/processed)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nom du dataset (défaut: auto depuis timestamp)"
    )
    
    # Paramètres de prétraitement CNN
    parser.add_argument(
        "--target_size",
        type=int,
        default=64,
        help="Taille cible pour redimensionnement (défaut: 64)"
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Activer le crop automatique des zones non-infectées"
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=5,
        help="Marge autour de la zone croppée (défaut: 5)"
    )
    
    # Paramètres de normalisation
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normaliser les valeurs en [0, 1]"
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardiser (mean=0, std=1)"
    )
    
    # Paramètres GNN
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=None,
        help="Distance seuil pour les arêtes GNN (défaut: 4-voisinage)"
    )
    parser.add_argument(
        "--add_spatial_features",
        action="store_true",
        help="Ajouter les coordonnées normalisées comme features"
    )
    
    # Filtrage
    parser.add_argument(
        "--min_infection",
        type=float,
        default=0.0,
        help="Seuil minimum d'infection pour garder un timestep (défaut: 0.0)"
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=None,
        help="Nombre max de timesteps par simulation (défaut: tous)"
    )
    
    # Options
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["cnn", "gnn"],
        choices=["cnn", "gnn"],
        help="Formats à générer (défaut: cnn gnn)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux"
    )
    
    return parser.parse_args()


def load_metadata(generation_dir):
    """Charge les métadonnées de génération."""
    meta_file = Path(generation_dir) / "generation_metadata.json"
    with open(meta_file, "r") as f:
        return json.load(f)


def crop_to_infected_region(data, margin=5):
    """
    Crop l'image pour se concentrer sur la région infectée.
    
    Args:
        data: Array 2D de la grille
        margin: Marge autour de la région détectée
        
    Returns:
        data_cropped, (x_min, y_min, x_max, y_max)
    """
    # Trouver les zones avec infection
    infected = data > 0.1  # Seuil minimal d'infection
    
    if not infected.any():
        # Pas d'infection, retourner l'original
        return data, (0, 0, data.shape[0], data.shape[1])
    
    # Trouver les coordonnées min/max
    rows = np.any(infected, axis=1)
    cols = np.any(infected, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Ajouter marge
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(data.shape[0], y_max + margin + 1)
    x_max = min(data.shape[1], x_max + margin + 1)
    
    return data[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


def preprocess_cnn(graph_file, args):
    """
    Prétraite les données pour CNN.
    
    Returns:
        dict avec 'data', 'original_shape', 'crop_bbox' si applicable
    """
    # Charger le graphe brut
    graph_data = np.load(graph_file, allow_pickle=True).item()
    height, width = graph_data["shape"]
    
    # Reconstruire la grille
    grid = graph_data["node_features"].reshape((height, width))
    
    # Crop si demandé
    crop_bbox = None
    if args.crop:
        grid, crop_bbox = crop_to_infected_region(grid, args.crop_margin)
    
    original_shape = grid.shape
    
    # Redimensionner
    if grid.shape != (args.target_size, args.target_size):
        # Convertir en image PIL pour resize de qualité
        img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
        img = img.resize((args.target_size, args.target_size), Image.BILINEAR)
        grid = np.array(img) / 255.0
    
    # Normalisation
    if args.normalize:
        # Déjà en [0, 1] depuis WaterberryFarms
        pass
    elif args.standardize:
        mean = grid.mean()
        std = grid.std()
        if std > 0:
            grid = (grid - mean) / std
    
    return {
        "data": grid.astype(np.float32),
        "original_shape": original_shape,
        "crop_bbox": crop_bbox,
        "timestep": graph_data["timestep"]
    }


def preprocess_gnn(graph_file, args):
    """
    Prétraite les données pour GNN.
    
    Returns:
        dict avec 'nodes', 'edges', 'node_features', 'edge_features' optionnels
    """
    # Charger le graphe brut
    graph_data = np.load(graph_file, allow_pickle=True).item()
    
    nodes = graph_data["nodes"]  # (N, 2) coordonnées
    features = graph_data["node_features"]  # (N,)
    edges = graph_data["edges"]  # (E, 2)
    shape = graph_data["shape"]
    
    # Normaliser les coordonnées si demandé
    if args.add_spatial_features:
        # Coordonnées normalisées en [0, 1]
        normalized_coords = nodes.astype(np.float32) / np.array(shape)
        # Concaténer avec les features
        node_features = np.column_stack([
            features[:, np.newaxis],
            normalized_coords
        ])  # (N, 3)
    else:
        node_features = features[:, np.newaxis]  # (N, 1)
    
    # Normalisation des features
    if args.normalize:
        # Normaliser seulement la première colonne (intensité)
        node_features[:, 0] = node_features[:, 0]  # Déjà en [0, 1]
    elif args.standardize:
        mean = node_features[:, 0].mean()
        std = node_features[:, 0].std()
        if std > 0:
            node_features[:, 0] = (node_features[:, 0] - mean) / std
    
    # Modifier les arêtes si seuil de distance spécifié
    if args.edge_threshold is not None:
        # Recalculer les arêtes basées sur la distance
        from scipy.spatial.distance import cdist
        distances = cdist(nodes, nodes)
        new_edges = np.argwhere(
            (distances > 0) & (distances <= args.edge_threshold)
        )
        edges = new_edges
    
    return {
        "nodes": nodes.astype(np.int32),
        "node_features": node_features.astype(np.float32),
        "edges": edges.astype(np.int32),
        "shape": shape,
        "timestep": graph_data["timestep"]
    }


def process_simulation(sim_dir, output_dir, args, sim_id):
    """Traite une simulation complète."""
    sim_dir = Path(sim_dir)
    graphs_dir = sim_dir / "graphs"
    
    if not graphs_dir.exists():
        print(f"  Attention: Pas de graphes dans {sim_dir.name}")
        return None
    
    # Lister les fichiers
    graph_files = sorted(graphs_dir.glob("t_*.npy"))
    
    if args.max_timesteps:
        graph_files = graph_files[:args.max_timesteps]
    
    # Métadonnées de simulation
    with open(sim_dir / "metadata.json", "r") as f:
        sim_metadata = json.load(f)
    
    processed_data = {
        "simulation_id": sim_id,
        "original_metadata": sim_metadata,
        "preprocessing_params": vars(args),
        "cnn_samples": [],
        "gnn_samples": [],
        "timesteps": []
    }
    
    # Traiter chaque timestep
    for graph_file in tqdm(graph_files, desc=f"  Sim {sim_id}", leave=False):
        timestep = int(graph_file.stem.split("_")[1])
        
        # Charger pour vérifier infection minimum
        graph_data = np.load(graph_file, allow_pickle=True).item()
        infection_level = graph_data["node_features"].mean()
        
        if infection_level < args.min_infection:
            continue
        
        processed_data["timesteps"].append(timestep)
        
        # Prétraiter pour CNN
        if "cnn" in args.formats:
            cnn_data = preprocess_cnn(graph_file, args)
            
            # Sauvegarder
            cnn_file = output_dir / "cnn" / f"sim_{sim_id:04d}" / f"t_{timestep:04d}.npy"
            cnn_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cnn_file, cnn_data)
            
            processed_data["cnn_samples"].append(str(cnn_file.relative_to(output_dir)))
        
        # Prétraiter pour GNN
        if "gnn" in args.formats:
            gnn_data = preprocess_gnn(graph_file, args)
            
            # Sauvegarder
            gnn_file = output_dir / "gnn" / f"sim_{sim_id:04d}" / f"t_{timestep:04d}.npy"
            gnn_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(gnn_file, gnn_data)
            
            processed_data["gnn_samples"].append(str(gnn_file.relative_to(output_dir)))
    
    return processed_data


def compute_statistics(output_dir, formats):
    """Calcule les statistiques sur les données preprocessées."""
    stats = {}
    
    for fmt in formats:
        fmt_dir = output_dir / fmt
        if not fmt_dir.exists():
            continue
        
        all_files = list(fmt_dir.rglob("t_*.npy"))
        
        if fmt == "cnn":
            # Statistiques CNN
            all_data = []
            for f in all_files[:100]:  # Échantillon
                data = np.load(f, allow_pickle=True).item()
                all_data.append(data["data"])
            
            if all_data:
                all_data = np.array(all_data)
                stats[fmt] = {
                    "num_samples": len(all_files),
                    "shape": all_data[0].shape,
                    "mean": float(all_data.mean()),
                    "std": float(all_data.std()),
                    "min": float(all_data.min()),
                    "max": float(all_data.max())
                }
        
        elif fmt == "gnn":
            # Statistiques GNN
            total_nodes = []
            total_edges = []
            feature_dims = []
            
            for f in all_files[:100]:  # Échantillon
                data = np.load(f, allow_pickle=True).item()
                total_nodes.append(len(data["nodes"]))
                total_edges.append(len(data["edges"]))
                feature_dims.append(data["node_features"].shape[1])
            
            if total_nodes:
                stats[fmt] = {
                    "num_samples": len(all_files),
                    "avg_nodes": float(np.mean(total_nodes)),
                    "avg_edges": float(np.mean(total_edges)),
                    "feature_dim": int(feature_dims[0])
                }
    
    return stats


def main():
    """Fonction principale."""
    args = parse_args()
    
    input_dir = Path(args.input)
    output_base = Path(args.output)
    
    # Vérifier l'entrée
    if not input_dir.exists():
        print(f"Erreur: Dossier d'entrée introuvable: {input_dir}")
        return
    
    # Charger métadonnées
    try:
        gen_metadata = load_metadata(input_dir)
    except Exception as e:
        print(f"Erreur: Impossible de charger les métadonnées: {e}")
        return
    
    # Nom du dataset
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"processed_{timestamp}"
    else:
        dataset_name = args.name
    
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"  PRÉTRAITEMENT DES DONNÉES")
    print(f"{'='*70}")
    print(f"  Entree: {input_dir}")
    print(f"  Sortie: {output_dir}")
    print(f"  Formats: {', '.join(args.formats)}")
    print(f"  Taille cible (CNN): {args.target_size}x{args.target_size}")
    print(f"  Crop: {args.crop}")
    print(f"  Normalisation: {args.normalize or args.standardize}")
    print(f"{'='*70}\n")
    
    # Créer structure de sortie
    for fmt in args.formats:
        (output_dir / fmt).mkdir(exist_ok=True)
    
    # Lister les simulations
    sim_dirs = sorted(input_dir.glob("sim_*"))
    
    if not sim_dirs:
        print("Erreur: Aucune simulation trouvée!")
        return
    
    # Traiter chaque simulation
    processed_metadata = {
        "dataset_name": dataset_name,
        "source_generation": str(input_dir),
        "preprocessing_params": vars(args),
        "timestamp": datetime.now().isoformat(),
        "original_metadata": gen_metadata,
        "simulations": []
    }
    
    print(f"Traitement de {len(sim_dirs)} simulations...\n")
    
    for sim_id, sim_dir in enumerate(sim_dirs):
        print(f"[{sim_id+1}/{len(sim_dirs)}] {sim_dir.name}")
        
        sim_data = process_simulation(sim_dir, output_dir, args, sim_id)
        
        if sim_data:
            processed_metadata["simulations"].append(sim_data)
            print(f"  {len(sim_data['timesteps'])} timesteps traites")
    
    # Calculer statistiques
    print("\nCalcul des statistiques...")
    stats = compute_statistics(output_dir, args.formats)
    processed_metadata["statistics"] = stats
    
    # Sauvegarder métadonnées
    meta_file = output_dir / "preprocessing_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(processed_metadata, f, indent=2)
    
    # Résumé
    print(f"\n{'='*70}")
    print(f"  PRÉTRAITEMENT TERMINÉ")
    print(f"{'='*70}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Simulations: {len(processed_metadata['simulations'])}")
    
    for fmt, st in stats.items():
        print(f"\n  {fmt.upper()}:")
        for key, val in st.items():
            print(f"    {key}: {val}")
    
    print(f"\n  Métadonnées: {meta_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
