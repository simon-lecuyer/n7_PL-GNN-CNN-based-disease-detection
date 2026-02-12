#!/usr/bin/env python3
"""
Script de visualisation des datasets prétraités (CNN et GNN).

Permet de visualiser les échantillons individuels, les séquences temporelles,
les distributions de données et les statistiques des datasets.

Usage:
    # Visualiser des échantillons CNN
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/cnn/train.json \\
        --format cnn --mode samples --num_samples 6

    # Visualiser une évolution temporelle
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/cnn/train.json \\
        --format cnn --mode temporal --sim_id 0

    # Visualiser un graphe GNN
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/gnn/train.json \\
        --format gnn --mode samples --num_samples 4

    # Statistiques du dataset
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/cnn/train.json \\
        --format cnn --mode stats

    # Comparer CNN vs GNN côte à côte
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/cnn/train.json \\
        --dataset_gnn data/processed/processed_XXX/datasets/gnn/train.json \\
        --mode compare --sim_id 0 --timestep 5

    # Sauvegarder dans un fichier
    python scripts/visualize_dataset.py \\
        --dataset data/processed/processed_XXX/datasets/cnn/train.json \\
        --format cnn --mode samples --save results/figures/samples.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
try:
    import seaborn as sns
except ImportError:
    sns = None


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Visualisation des datasets de détection de maladies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  samples    Affiche des échantillons aléatoires ou choisis
  temporal   Affiche l'évolution temporelle d'une simulation
  stats      Affiche les statistiques et distributions du dataset
  compare    Compare CNN et GNN côte à côte (nécessite --dataset_gnn)
  grid       Affiche une grille d'échantillons par simulation et timestep
        """
    )

    # Paramètre principal
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Chemin vers le fichier split (ex: datasets/cnn/train.json)"
    )
    parser.add_argument(
        "--dataset_gnn",
        type=str,
        default=None,
        help="Chemin vers le split GNN pour le mode compare"
    )

    # Format
    parser.add_argument(
        "--format",
        type=str,
        default="cnn",
        choices=["cnn", "gnn"],
        help="Format des données (défaut: cnn)"
    )

    # Mode de visualisation
    parser.add_argument(
        "--mode",
        type=str,
        default="samples",
        choices=["samples", "temporal", "stats", "compare", "grid"],
        help="Mode de visualisation (défaut: samples)"
    )

    # Sélection d'échantillons
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Nombre d'échantillons à afficher (défaut: 6)"
    )
    parser.add_argument(
        "--sim_id",
        type=int,
        default=None,
        help="ID de simulation à afficher (pour modes temporal/compare/grid)"
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Timestep spécifique à afficher (pour mode compare)"
    )

    # Options d'affichage
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlOrRd",
        help="Colormap matplotlib (défaut: YlOrRd)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        help="Taille de la figure (largeur hauteur)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Résolution de la figure (défaut: 150)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Titre personnalisé pour la figure"
    )
    parser.add_argument(
        "--no_colorbar",
        action="store_true",
        help="Masquer la barre de couleur"
    )

    # Sortie
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Chemin de sauvegarde de la figure (ex: results/figures/visu.png)"
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Ne pas afficher la figure (utile avec --save)"
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed aléatoire pour la sélection d'échantillons (défaut: 42)"
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Chargement des données
# ──────────────────────────────────────────────────────────────

def load_split(split_file: str) -> Dict:
    """Charge un fichier split JSON."""
    path = Path(split_file)
    if not path.exists():
        print(f"Erreur: fichier introuvable: {split_file}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def load_sample(file_path: str) -> Dict:
    """Charge un fichier .npy contenant un échantillon."""
    return np.load(file_path, allow_pickle=True).item()


def get_cnn_image(data: Dict) -> np.ndarray:
    """Extrait l'image 2D depuis les données CNN."""
    return data["data"]


def get_gnn_data(data: Dict) -> Dict:
    """Extrait les composants du graphe depuis les données GNN."""
    return {
        "nodes": data["nodes"],
        "edges": data["edges"],
        "node_features": data["node_features"],
        "shape": data["shape"],
    }


# ──────────────────────────────────────────────────────────────
# Mode: samples
# ──────────────────────────────────────────────────────────────

def visualize_samples(split_data: Dict, args):
    """Affiche des échantillons choisis aléatoirement du dataset."""
    samples = split_data["samples"]
    rng = np.random.default_rng(args.seed)

    n = min(args.num_samples, len(samples))
    indices = rng.choice(len(samples), size=n, replace=False)
    indices = sorted(indices)

    if args.format == "cnn":
        _plot_cnn_samples(samples, indices, args)
    else:
        _plot_gnn_samples(samples, indices, args)


def _plot_cnn_samples(samples: List, indices: List[int], args):
    """Affiche des échantillons CNN sous forme d'images."""
    n = len(indices)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    figsize = args.figsize or (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        sample = samples[idx]
        data = load_sample(sample["file"])
        img = get_cnn_image(data)

        im = axes[i].imshow(img, cmap=args.cmap, vmin=0, vmax=1)
        axes[i].set_title(
            f"sim={sample['sim_id']} t={sample['timestep']}\n"
            f"infection={sample['infection_level']:.3f}",
            fontsize=9
        )
        axes[i].axis("off")
        if not args.no_colorbar:
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Masquer les axes inutilisés
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    title = args.title or f"Échantillons CNN ({Path(args.dataset).stem})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()


def _plot_gnn_samples(samples: List, indices: List[int], args):
    """Affiche des échantillons GNN sous forme de graphes."""
    n = len(indices)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    figsize = args.figsize or (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi)
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        sample = samples[idx]
        data = load_sample(sample["file"])
        gnn = get_gnn_data(data)

        _draw_graph(axes[i], gnn, args.cmap)
        axes[i].set_title(
            f"sim={sample['sim_id']} t={sample['timestep']}\n"
            f"infection={sample['infection_level']:.3f}\n"
            f"nœuds={len(gnn['nodes'])} arêtes={len(gnn['edges'])}",
            fontsize=9
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    title = args.title or f"Échantillons GNN ({Path(args.dataset).stem})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()


def _draw_graph(ax, gnn: Dict, cmap: str):
    """Dessine un graphe GNN sur un axe matplotlib."""
    nodes = gnn["nodes"]
    edges = gnn["edges"]
    features = gnn["node_features"]
    h, w = gnn["shape"]

    # Positions des nœuds sur la grille
    if nodes.ndim == 1:
        # Indices linéaires → coordonnées 2D
        ys, xs = np.divmod(nodes, w)
    else:
        xs, ys = nodes[:, 0], nodes[:, 1]

    # Couleur = première feature (infection)
    if features.ndim == 2:
        colors = features[:, 0]
    else:
        colors = features

    # Dessiner les arêtes
    for edge in edges:
        src, dst = int(edge[0]), int(edge[1])
        if src < len(xs) and dst < len(xs):
            ax.plot(
                [xs[src], xs[dst]], [ys[src], ys[dst]],
                color="lightgray", linewidth=0.3, zorder=1
            )

    # Dessiner les nœuds
    sc = ax.scatter(
        xs, ys, c=colors, cmap=cmap, s=15,
        vmin=0, vmax=1, edgecolors="none", zorder=2
    )
    ax.set_xlim(-1, w)
    ax.set_ylim(-1, h)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


# ──────────────────────────────────────────────────────────────
# Mode: temporal
# ──────────────────────────────────────────────────────────────

def visualize_temporal(split_data: Dict, args):
    """Affiche l'évolution temporelle d'une simulation."""
    samples = split_data["samples"]

    # Déterminer les simulations disponibles
    sim_ids = sorted(set(s["sim_id"] for s in samples))

    if args.sim_id is None:
        sim_id = sim_ids[0]
        print(f"Simulations disponibles: {sim_ids}")
        print(f"Utilisation de sim_id={sim_id} (par défaut)")
    else:
        sim_id = args.sim_id
        if sim_id not in sim_ids:
            print(f"Erreur: sim_id={sim_id} non trouvé. Disponibles: {sim_ids}")
            sys.exit(1)

    # Filtrer et trier par timestep
    sim_samples = sorted(
        [s for s in samples if s["sim_id"] == sim_id],
        key=lambda s: s["timestep"]
    )

    if not sim_samples:
        print(f"Aucun échantillon pour sim_id={sim_id}")
        sys.exit(1)

    n = len(sim_samples)

    if args.format == "cnn":
        ncols = min(n, 5)
        nrows = (n + ncols - 1) // ncols
        figsize = args.figsize or (4 * ncols, 4 * nrows + 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi)
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, sample in enumerate(sim_samples):
            data = load_sample(sample["file"])
            img = get_cnn_image(data)

            im = axes[i].imshow(img, cmap=args.cmap, vmin=0, vmax=1)
            axes[i].set_title(
                f"t={sample['timestep']}\n"
                f"inf={sample['infection_level']:.3f}",
                fontsize=9
            )
            axes[i].axis("off")

        # Colorbar commune
        fig.colorbar(im, ax=axes[:n].tolist(), fraction=0.02, pad=0.04,
                     label="Niveau d'infection")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

    else:  # GNN
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        figsize = args.figsize or (5 * ncols, 5 * nrows + 1)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi)
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, sample in enumerate(sim_samples):
            data = load_sample(sample["file"])
            gnn = get_gnn_data(data)
            _draw_graph(axes[i], gnn, args.cmap)
            axes[i].set_title(
                f"t={sample['timestep']}  inf={sample['infection_level']:.3f}",
                fontsize=9
            )

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

    title = args.title or f"Évolution temporelle - Simulation {sim_id}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────
# Mode: stats
# ──────────────────────────────────────────────────────────────

def visualize_stats(split_data: Dict, args):
    """Affiche les statistiques et distributions du dataset."""
    samples = split_data["samples"]

    infection_levels = [s["infection_level"] for s in samples]
    timesteps = [s["timestep"] for s in samples]
    sim_ids = [s["sim_id"] for s in samples]

    figsize = args.figsize or (16, 10)
    fig = plt.figure(figsize=figsize, dpi=args.dpi)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # 1. Distribution des niveaux d'infection
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(infection_levels, bins=20, color="coral", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Niveau d'infection")
    ax1.set_ylabel("Nombre d'échantillons")
    ax1.set_title("Distribution infection")
    ax1.axvline(np.mean(infection_levels), color="red", linestyle="--",
                label=f"moyenne={np.mean(infection_levels):.3f}")
    ax1.legend(fontsize=8)

    # 2. Distribution des timesteps
    ax2 = fig.add_subplot(gs[0, 1])
    unique_t, counts_t = np.unique(timesteps, return_counts=True)
    ax2.bar(unique_t, counts_t, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Nombre d'échantillons")
    ax2.set_title("Distribution timesteps")

    # 3. Échantillons par simulation
    ax3 = fig.add_subplot(gs[0, 2])
    unique_s, counts_s = np.unique(sim_ids, return_counts=True)
    ax3.bar(unique_s, counts_s, color="seagreen", edgecolor="white")
    ax3.set_xlabel("Simulation ID")
    ax3.set_ylabel("Nombre d'échantillons")
    ax3.set_title("Échantillons par simulation")

    # 4. Infection vs Timestep
    ax4 = fig.add_subplot(gs[1, 0])
    for sid in sorted(set(sim_ids)):
        mask = [i for i, s in enumerate(sim_ids) if s == sid]
        ts = [timesteps[i] for i in mask]
        inf = [infection_levels[i] for i in mask]
        ax4.plot(ts, inf, "o-", markersize=4, label=f"sim {sid}", alpha=0.8)
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Niveau d'infection")
    ax4.set_title("Infection vs Timestep")
    ax4.legend(fontsize=7, ncol=2)

    # 5. Boxplot infection par simulation
    ax5 = fig.add_subplot(gs[1, 1])
    data_by_sim = {}
    for s, inf in zip(sim_ids, infection_levels):
        data_by_sim.setdefault(s, []).append(inf)
    sim_keys = sorted(data_by_sim.keys())
    ax5.boxplot([data_by_sim[k] for k in sim_keys], labels=sim_keys)
    ax5.set_xlabel("Simulation ID")
    ax5.set_ylabel("Niveau d'infection")
    ax5.set_title("Infection par simulation")

    # 6. Résumé texte
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary = (
        f"Dataset: {Path(args.dataset).stem}\n"
        f"Format: {args.format}\n"
        f"─────────────────────\n"
        f"Échantillons: {len(samples)}\n"
        f"Simulations: {len(set(sim_ids))}\n"
        f"Timesteps: {min(timesteps)} → {max(timesteps)}\n"
        f"─────────────────────\n"
        f"Infection :\n"
        f"  min   = {min(infection_levels):.4f}\n"
        f"  max   = {max(infection_levels):.4f}\n"
        f"  moy   = {np.mean(infection_levels):.4f}\n"
        f"  std   = {np.std(infection_levels):.4f}\n"
        f"  med   = {np.median(infection_levels):.4f}"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.8))

    title = args.title or f"Statistiques du dataset ({Path(args.dataset).stem})"
    fig.suptitle(title, fontsize=14, fontweight="bold")


# ──────────────────────────────────────────────────────────────
# Mode: compare
# ──────────────────────────────────────────────────────────────

def visualize_compare(split_data_cnn: Dict, split_data_gnn: Dict, args):
    """Compare CNN et GNN côte à côte pour un même échantillon."""
    samples_cnn = split_data_cnn["samples"]
    samples_gnn = split_data_gnn["samples"]

    # Trouver les échantillons communs (même sim_id, timestep)
    cnn_index = {(s["sim_id"], s["timestep"]): s for s in samples_cnn}
    gnn_index = {(s["sim_id"], s["timestep"]): s for s in samples_gnn}
    common_keys = sorted(set(cnn_index.keys()) & set(gnn_index.keys()))

    if not common_keys:
        print("Erreur: aucun échantillon commun CNN/GNN trouvé.")
        sys.exit(1)

    # Filtrer par sim_id / timestep si spécifié
    if args.sim_id is not None:
        common_keys = [k for k in common_keys if k[0] == args.sim_id]
    if args.timestep is not None:
        common_keys = [k for k in common_keys if k[1] == args.timestep]

    if not common_keys:
        print("Erreur: aucun échantillon correspondant aux critères.")
        sys.exit(1)

    n = min(len(common_keys), args.num_samples)
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(len(common_keys), size=n, replace=False)
    selected = sorted(selected)

    figsize = args.figsize or (10, 4.5 * n)
    fig, axes = plt.subplots(n, 2, figsize=figsize, dpi=args.dpi)
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, sel_idx in enumerate(selected):
        sim_id, timestep = common_keys[sel_idx]
        sample_cnn = cnn_index[(sim_id, timestep)]
        sample_gnn = gnn_index[(sim_id, timestep)]

        # CNN
        data_cnn = load_sample(sample_cnn["file"])
        img = get_cnn_image(data_cnn)
        im = axes[i, 0].imshow(img, cmap=args.cmap, vmin=0, vmax=1)
        axes[i, 0].set_title(
            f"CNN - sim={sim_id} t={timestep}\n"
            f"inf={sample_cnn['infection_level']:.3f}",
            fontsize=9
        )
        axes[i, 0].axis("off")
        plt.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # GNN
        data_gnn = load_sample(sample_gnn["file"])
        gnn = get_gnn_data(data_gnn)
        _draw_graph(axes[i, 1], gnn, args.cmap)
        axes[i, 1].set_title(
            f"GNN - sim={sim_id} t={timestep}\n"
            f"inf={sample_gnn['infection_level']:.3f}"
            f"  nœuds={len(gnn['nodes'])}",
            fontsize=9
        )

    title = args.title or "Comparaison CNN vs GNN"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────
# Mode: grid
# ──────────────────────────────────────────────────────────────

def visualize_grid(split_data: Dict, args):
    """Affiche une grille simulation × timestep."""
    samples = split_data["samples"]

    sim_ids = sorted(set(s["sim_id"] for s in samples))
    timesteps_all = sorted(set(s["timestep"] for s in samples))

    # Filtrer par sim_id si spécifié
    if args.sim_id is not None:
        sim_ids = [s for s in sim_ids if s == args.sim_id]

    # Indexer
    index = {(s["sim_id"], s["timestep"]): s for s in samples}

    nrows = len(sim_ids)
    ncols = len(timesteps_all)

    if nrows == 0 or ncols == 0:
        print("Erreur: aucune donnée à afficher.")
        sys.exit(1)

    if args.format == "cnn":
        figsize = args.figsize or (3 * ncols, 3 * nrows + 1)
    else:
        figsize = args.figsize or (4 * ncols, 4 * nrows + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=args.dpi,
                             squeeze=False)

    for r, sid in enumerate(sim_ids):
        for c, t in enumerate(timesteps_all):
            ax = axes[r, c]
            key = (sid, t)

            if key not in index:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=12, color="gray", transform=ax.transAxes)
                ax.set_facecolor("#f0f0f0")
                ax.axis("off")
                continue

            sample = index[key]
            data = load_sample(sample["file"])

            if args.format == "cnn":
                img = get_cnn_image(data)
                ax.imshow(img, cmap=args.cmap, vmin=0, vmax=1)
                ax.axis("off")
            else:
                gnn = get_gnn_data(data)
                _draw_graph(ax, gnn, args.cmap)

            # Labels
            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"sim {sid}", fontsize=9)

    title = args.title or f"Grille Simulation × Timestep ({args.format.upper()})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Chargement du dataset: {args.dataset}")
    split_data = load_split(args.dataset)
    print(f"  → {len(split_data['samples'])} échantillons")

    if args.mode == "samples":
        visualize_samples(split_data, args)

    elif args.mode == "temporal":
        visualize_temporal(split_data, args)

    elif args.mode == "stats":
        visualize_stats(split_data, args)

    elif args.mode == "compare":
        if args.dataset_gnn is None:
            print("Erreur: --dataset_gnn requis pour le mode compare")
            sys.exit(1)
        print(f"Chargement du dataset GNN: {args.dataset_gnn}")
        split_data_gnn = load_split(args.dataset_gnn)
        print(f"  → {len(split_data_gnn['samples'])} échantillons GNN")
        visualize_compare(split_data, split_data_gnn, args)

    elif args.mode == "grid":
        visualize_grid(split_data, args)

    # Sauvegarde et/ou affichage
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Figure sauvegardée: {save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
