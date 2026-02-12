#!/usr/bin/env python3
"""
Script de génération de données pour l'entraînement des modèles GNN et CNN.
Utilise WaterberryFarms pour simuler la propagation de maladies végétales.

Usage:
    python scripts/generate_data.py --output_dir data/simulations --num_simulations 10 --grid_size 50 --timesteps 100

Requirements:
    - WaterberryFarms doit être dans le même dossier parent que ce projet
    - Structure attendue:
        Projet_Long/
        ├── n7_PL-GNN-CNN-based-disease-detection/
        └── WaterberryFarms/
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ajouter le chemin vers WaterberryFarms
WATERBERRYFARMS_PATH = Path(__file__).resolve().parent.parent.parent / "WaterberryFarms"
if not WATERBERRYFARMS_PATH.exists():
    raise ImportError(
        f"WaterberryFarms introuvable à : {WATERBERRYFARMS_PATH}\n"
        f"Assurez-vous que WaterberryFarms est dans le même dossier parent que ce projet."
    )
sys.path.insert(0, str(WATERBERRYFARMS_PATH))

# Imports WaterberryFarms
from environment import EpidemicSpreadEnvironment, DissipationModelEnvironment


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Génère des données de simulation de propagation de maladies végétales"
    )
    
    # Paramètres de sortie
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/simulations",
        help="Répertoire de sortie pour les données générées (défaut: data/simulations)"
    )
    
    # Paramètres de simulation
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10,
        help="Nombre de simulations à générer (défaut: 10)"
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=50,
        help="Taille de la grille (NxN) (défaut: 50)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100,
        help="Nombre de pas de temps par simulation (défaut: 100)"
    )
    
    # Paramètres du modèle épidémique (SIR)
    parser.add_argument(
        "--model_type",
        type=str,
        default="epidemic",
        choices=["epidemic", "dissipation"],
        help="Type de modèle d'environnement (défaut: epidemic)"
    )
    parser.add_argument(
        "--p_transmission",
        type=float,
        default=0.2,
        help="Probabilité de transmission de la maladie (défaut: 0.2)"
    )
    parser.add_argument(
        "--infection_duration",
        type=int,
        default=5,
        help="Durée de l'infection en pas de temps (défaut: 5)"
    )
    parser.add_argument(
        "--spread_dimension",
        type=int,
        default=11,
        help="Dimension de propagation spatiale (défaut: 11)"
    )
    parser.add_argument(
        "--infection_seeds",
        type=int,
        default=-1,
        help="Nombre de foyers infectieux initiaux (-1 pour auto) (défaut: -1)"
    )
    
    # Paramètres du modèle de dissipation
    parser.add_argument(
        "--dissipation_rate",
        type=float,
        default=0.95,
        help="Taux de dissipation pour le modèle de dissipation (défaut: 0.95)"
    )
    parser.add_argument(
        "--p_pollution",
        type=float,
        default=0.1,
        help="Probabilité d'événement de pollution (défaut: 0.1)"
    )
    
    # Paramètres de format de sortie
    parser.add_argument(
        "--output_formats",
        type=str,
        nargs="+",
        default=["images", "graphs"],
        choices=["images", "graphs"],
        help="Formats de sortie à générer (défaut: images graphs)"
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "npy"],
        help="Format des images (défaut: png)"
    )
    
    # Paramètres de seed
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aléatoire pour la reproductibilité (défaut: None)"
    )
    
    return parser.parse_args()


def create_environment(model_type, width, height, seed, args):
    """Crée l'environnement de simulation selon les paramètres."""
    if model_type == "epidemic":
        return EpidemicSpreadEnvironment(
            name="disease_spread",
            width=width,
            height=height,
            seed=seed,
            p_transmission=args.p_transmission,
            infection_duration=args.infection_duration,
            spread_dimension=args.spread_dimension,
            infection_seeds=args.infection_seeds
        )
    elif model_type == "dissipation":
        return DissipationModelEnvironment(
            name="disease_dissipation",
            width=width,
            height=height,
            seed=seed,
            dissipation_rate=args.dissipation_rate,
            p_pollution=args.p_pollution
        )
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")


def save_as_image(data, output_path, image_format):
    """Sauvegarde les données sous forme d'image."""
    if image_format == "npy":
        np.save(output_path.with_suffix(".npy"), data)
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap="viridis", origin="lower")
        plt.colorbar(label="Disease Intensity")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path.with_suffix(f".{image_format}"), dpi=150, bbox_inches="tight")
        plt.close()


def save_as_graph(data, output_path, timestep, status_data=None):
    """Sauvegarde les données sous forme de graphe spatial."""
    height, width = data.shape
    
    # Créer les coordonnées des nœuds
    nodes = []
    node_features = []
    node_status = []  # Ajouter le status SIR
    
    for i in range(height):
        for j in range(width):
            nodes.append((i, j))
            node_features.append(data[i, j])
            if status_data is not None:
                node_status.append(status_data[i, j])
            else:
                node_status.append(0.0)
    
    # Créer les arêtes (connexions 4-voisinage)
    edges = []
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            # Droite
            if j < width - 1:
                edges.append((node_id, node_id + 1))
            # Bas
            if i < height - 1:
                edges.append((node_id, node_id + width))
    
    # Sauvegarder au format NumPy pour faciliter le chargement
    graph_data = {
        "nodes": np.array(nodes),
        "node_features": np.array(node_features),
        "node_status": np.array(node_status),  # Ajouter status SIR
        "edges": np.array(edges),
        "timestep": timestep,
        "shape": (height, width)
    }
    
    np.save(output_path, graph_data)


def run_simulation(sim_id, args, output_base_dir):
    """Exécute une simulation complète."""
    # Créer le dossier de simulation
    sim_dir = output_base_dir / f"sim_{sim_id:04d}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer les sous-dossiers selon les formats demandés
    if "images" in args.output_formats:
        images_dir = sim_dir / "images"
        images_dir.mkdir(exist_ok=True)
    
    if "graphs" in args.output_formats:
        graphs_dir = sim_dir / "graphs"
        graphs_dir.mkdir(exist_ok=True)
    
    # Générer un seed pour cette simulation
    if args.seed is not None:
        sim_seed = args.seed + sim_id
    else:
        sim_seed = np.random.randint(0, 1000000)
    
    # Créer l'environnement
    env = create_environment(
        args.model_type,
        args.grid_size,
        args.grid_size,
        sim_seed,
        args
    )
    
    # Métadonnées de la simulation
    metadata = {
        "simulation_id": sim_id,
        "seed": sim_seed,
        "model_type": args.model_type,
        "grid_size": args.grid_size,
        "timesteps": args.timesteps,
        "parameters": {
            "p_transmission": args.p_transmission if args.model_type == "epidemic" else None,
            "infection_duration": args.infection_duration if args.model_type == "epidemic" else None,
            "spread_dimension": args.spread_dimension if args.model_type == "epidemic" else None,
            "infection_seeds": args.infection_seeds if args.model_type == "epidemic" else None,
            "dissipation_rate": args.dissipation_rate if args.model_type == "dissipation" else None,
            "p_pollution": args.p_pollution if args.model_type == "dissipation" else None,
        },
        "output_formats": args.output_formats,
        "timestamp": datetime.now().isoformat()
    }
    
    # Simulation temporelle
    print(f"\n  Simulation {sim_id}: Exécution de {args.timesteps} pas de temps...")
    for t in tqdm(range(args.timesteps), desc=f"  Sim {sim_id}", leave=False):
        # Capturer status pour le modèle épidémique
        status_data = env.status if hasattr(env, 'status') else None
        
        # Sauvegarder les données au format image
        if "images" in args.output_formats:
            image_path = images_dir / f"t_{t:04d}"
            save_as_image(env.value, image_path, args.image_format)
        
        # Sauvegarder les données au format graphe
        if "graphs" in args.output_formats:
            graph_path = graphs_dir / f"t_{t:04d}.npy"
            save_as_graph(env.value, graph_path, t, status_data)
        
        # Faire évoluer l'environnement
        env.proceed(delta_t=1.0)
    
    # Sauvegarder les métadonnées
    with open(sim_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    """Fonction principale."""
    args = parse_args()
    
    # Créer le répertoire de sortie avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(args.output_dir) / f"generation_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"  GÉNÉRATION DE DONNÉES - WaterberryFarms x N7 Projet Long")
    print(f"{'='*70}")
    print(f"  Repertoire de sortie: {output_base_dir}")
    print(f"  Nombre de simulations: {args.num_simulations}")
    print(f"  Taille de grille: {args.grid_size}x{args.grid_size}")
    print(f"  Pas de temps: {args.timesteps}")
    print(f"  Modele: {args.model_type}")
    print(f"  Formats de sortie: {', '.join(args.output_formats)}")
    print(f"{'='*70}\n")
    
    # Métadonnées globales
    global_metadata = {
        "generation_timestamp": timestamp,
        "num_simulations": args.num_simulations,
        "arguments": vars(args),
        "waterberryfarms_path": str(WATERBERRYFARMS_PATH),
        "simulations": []
    }
    
    # Générer les simulations
    for sim_id in range(args.num_simulations):
        print(f"\n[{sim_id + 1}/{args.num_simulations}] Génération de la simulation {sim_id}...")
        sim_metadata = run_simulation(sim_id, args, output_base_dir)
        global_metadata["simulations"].append(sim_metadata)
    
    # Sauvegarder les métadonnées globales
    with open(output_base_dir / "generation_metadata.json", "w") as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  Generation terminee avec succes")
    print(f"  Donnees sauvegardees dans: {output_base_dir}")
    print(f"  {args.num_simulations} simulations x {args.timesteps} timesteps generees")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
