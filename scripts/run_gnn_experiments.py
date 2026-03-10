#!/usr/bin/env python3
"""
Orchestrateur des expériences GNN early/late stage detection.

Lit configs/gnn_experiments.yaml et pour chaque combinaison
(dataset_config × sequence_length) :
  1. Génère les simulations (une fois par dataset)
  2. Prétraite les données (une fois par dataset)
  3. Crée les splits avec séquences (une fois par dataset × seq_length)
  4. Entraîne le modèle temporel
  5. Évalue et sauvegarde les métriques

En fin de boucle, génère les visualisations de comparaison.

Usage:
    python scripts/run_gnn_experiments.py --config configs/gnn_experiments.yaml
    python scripts/run_gnn_experiments.py --config configs/gnn_experiments.yaml --skip_existing
"""

import sys
import os
import argparse
import json
import subprocess
import tempfile
import csv
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.train_gnn import train as train_model
from scripts.eval_metrics import evaluate_from_config

# ── project root = parent of this file's directory ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run all GNN early/late stage experiments")
    parser.add_argument(
        "--config", type=str, default="configs/gnn_experiments.yaml",
        help="Path to the experiments YAML config"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip data generation / training if outputs already exist"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(msg, width=70):
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


def experiment_header(idx, total, dataset_name, seq_name):
    """Print a clear progress header before each training run."""
    print(f"\n{'─'*70}")
    print(f"  [{idx}/{total}]  {dataset_name}  ×  {seq_name}")
    print(f"{'─'*70}")


def run_subprocess(cmd, description):
    """Run a shell command from PROJECT_ROOT, streaming output."""
    print(f"\n  → {description}")
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


def write_temp_config(config_dict, path):
    """Write a YAML config dict to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def generate_simulations(dataset_cfg, common_cfg, sim_dir, skip_existing):
    """
    Call generate_data.py for a dataset config.
    Simulations are generated once per dataset (shared across all seq lengths).
    """
    if skip_existing and (sim_dir / "generation_metadata.json").exists():
        print(f"  [skip] Simulations already exist: {sim_dir}")
        return

    cmd = [
        "python", "scripts/generate_data.py",
        "--output_dir", str(sim_dir.parent),      # generate_data appends generation_TIMESTAMP
        "--num_simulations", str(common_cfg["num_simulations"]),
        "--grid_size", str(common_cfg["grid_size"]),
        "--timesteps", str(dataset_cfg["timesteps"]),
        "--model_type", dataset_cfg["model_type"],
        "--seed", str(common_cfg["seed"]),
        "--output_formats", "graphs",             # GNN only needs graphs, not images
    ]

    if dataset_cfg["model_type"] == "epidemic":
        cmd += [
            "--p_transmission",    str(dataset_cfg["p_transmission"]),
            "--spread_dimension",  str(dataset_cfg["spread_dimension"]),
            "--infection_duration", str(dataset_cfg["infection_duration"]),
            "--infection_seeds",   str(dataset_cfg["infection_seeds"]),
        ]
    else:  # dissipation
        cmd += [
            "--dissipation_rate", str(dataset_cfg.get("dissipation_rate", 0.95)),
            "--p_pollution",      str(dataset_cfg.get("p_pollution", 0.1)),
        ]

    run_subprocess(cmd, f"Generating simulations for '{dataset_cfg['name']}'")

    # generate_data.py creates a timestamped sub-folder; move/rename it to sim_dir
    # so the path is stable across runs.
    parent = sim_dir.parent
    generated = sorted(parent.glob("generation_*"))
    if generated:
        latest = generated[-1]
        if latest != sim_dir:
            latest.rename(sim_dir)
            print(f"  Renamed {latest.name} → {sim_dir.name}")


def preprocess(sim_dir, processed_dir, common_cfg, skip_existing):
    """
    Call preprocess_data.py on the raw simulations.
    Result is stored at processed_dir (stable path, shared across seq lengths).
    """
    if skip_existing and (processed_dir / "preprocessing_metadata.json").exists():
        print(f"  [skip] Processed data already exists: {processed_dir}")
        return

    processed_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/preprocess_data.py",
        "--input",   str(sim_dir),
        "--output",  str(processed_dir.parent),
        "--name",    processed_dir.name,
        "--target_size", str(common_cfg["grid_size"]),
        "--crop",
        "--normalize",
        "--add_spatial_features",
        "--formats", "gnn",        # only GNN format needed here
    ]

    run_subprocess(cmd, "Preprocessing simulations")


def create_sequence_datasets(processed_dir, seq_cfg, datasets_dir, common_cfg, skip_existing):
    """
    Call create_datasets.py with --create_sequences for this sequence length.
    Outputs train_seqN.json / val_seqN.json / test_seqN.json in datasets_dir/gnn/.
    """
    seq_len = seq_cfg["sequence_length"]
    expected_train = datasets_dir / "gnn" / f"train_seq{seq_len}.json"

    if skip_existing and expected_train.exists():
        print(f"  [skip] Sequence datasets already exist: {expected_train}")
        return

    cmd = [
        "python", "scripts/create_datasets.py",
        "--input",           str(processed_dir),
        "--output",          str(datasets_dir),
        "--create_sequences",
        "--sequence_length", str(seq_len),
        "--sequence_stride", "1",
        "--stratify_by",     "simulation",
        "--seed",            str(common_cfg["seed"]),
        "--formats",         "gnn",
    ]

    run_subprocess(cmd, f"Creating sequence datasets (seq_len={seq_len})")


def build_experiment_config(experiment_name, seq_cfg, datasets_dir, exp_dir, common_cfg):
    """
    Build the YAML config dict that train_gnn.py / eval_metrics.py expect.
    Written to exp_dir/config.yaml so it is also a record of what was run.
    """
    seq_len = seq_cfg["sequence_length"]

    config = {
        "experiment_name": experiment_name,
        "data": {
            "train_path": str(datasets_dir / "gnn" / f"train_seq{seq_len}.json"),
            "val_path":   str(datasets_dir / "gnn" / f"val_seq{seq_len}.json"),
            "test_path":  str(datasets_dir / "gnn" / f"test_seq{seq_len}.json"),
            "checkpoint_path": str(exp_dir / "checkpoints"),
            "log_path":        str(exp_dir / "logs"),
        },
        "model": {
            "in_channels": 3,          # infection + x + y (temporal GNN)
            "hidden_dim":  common_cfg["hidden_dim"],
            "num_layers":  common_cfg["num_layers"],
            "out_channels": 1,
            "dropout":     common_cfg["dropout"],
            "type":        "temporal",
        },
        "training": {
            "epochs":    common_cfg["epochs"],
            "batch_size": common_cfg["batch_size"],
            "lr":        common_cfg["lr"],
            "optimizer": common_cfg["optimizer"],
            "device":    common_cfg["device"],
            "seed":      common_cfg["seed"],
        },
    }

    config_path = exp_dir / "config.yaml"
    write_temp_config(config, config_path)
    return str(config_path)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(csv_path, output_path, title):
    """Load a training log CSV and save a train/val loss curve."""
    epochs, train_losses, val_losses = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Val loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(grid_path, output_path, title):
    """Load a .npy error grid and save it as a heatmap."""
    grid = np.load(grid_path)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="hot", origin="lower")
    fig.colorbar(im, ax=ax, label="Mean Absolute Error")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_early_vs_late_comparison(all_results, output_dir):
    """
    Bar chart comparing MSE and F1 across sequence lengths,
    one group of bars per dataset config.
    """
    seq_names   = ["early_stage_3", "early_stage_5", "late_stage_10"]
    dataset_names = list(all_results.keys())

    x = np.arange(len(dataset_names))
    width = 0.25

    for metric in ("mse", "mae", "f1"):
        fig, ax = plt.subplots(figsize=(max(8, len(dataset_names) * 2), 5))

        for i, seq_name in enumerate(seq_names):
            values = []
            for ds_name in dataset_names:
                m = all_results[ds_name].get(seq_name, {}).get(metric, 0.0)
                values.append(m)
            ax.bar(x + i * width, values, width, label=seq_name)

        ax.set_xticks(x + width)
        ax.set_xticklabels(dataset_names, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Early vs Late Stage — {metric.upper()} by dataset")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = output_dir / f"comparison_{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved comparison plot → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load experiment config ───────────────────────────────────────────────
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        exp_cfg = yaml.safe_load(f)

    common       = exp_cfg["common"]
    datasets     = exp_cfg["datasets"]
    seq_configs  = exp_cfg["sequence_configs"]
    base_out     = PROJECT_ROOT / exp_cfg["base_output_dir"]

    total_experiments = len(datasets) * len(seq_configs)
    exp_counter = 0

    banner(f"GNN Experiments — {exp_cfg['experiment_name']}")
    print(f"  Datasets:        {len(datasets)}")
    print(f"  Seq configs:     {len(seq_configs)}")
    print(f"  Total training:  {total_experiments}")
    print(f"  Output dir:      {base_out}")

    # ── Collect all results for final comparison plots ───────────────────────
    # Structure: all_results[dataset_name][seq_name] = metrics_dict
    all_results = {ds["name"]: {} for ds in datasets}
    all_configs = []   # list of (exp_name, config_path, exp_dir) for vis

    # ─────────────────────────────────────────────────────────────────────────
    for ds_cfg in datasets:
        ds_name = ds_cfg["name"]
        ds_dir  = base_out / ds_name

        sim_dir       = ds_dir / "simulations"
        processed_dir = ds_dir / "processed"

        # ── Step 1: generate simulations (once per dataset) ──────────────────
        banner(f"Dataset: {ds_name}")
        generate_simulations(ds_cfg, common, sim_dir, args.skip_existing)

        # ── Step 2: preprocess (once per dataset) ────────────────────────────
        preprocess(sim_dir, processed_dir, common, args.skip_existing)

        # ── Step 3–5: loop over sequence lengths ─────────────────────────────
        for seq_cfg in seq_configs:
            exp_counter += 1
            seq_name = seq_cfg["name"]
            exp_name = f"{ds_name}__{seq_name}"

            exp_dir         = ds_dir / seq_name
            datasets_dir    = exp_dir / "datasets"
            metrics_dir     = exp_dir / "metrics"
            vis_dir         = exp_dir / "visualizations"
            checkpoints_dir = exp_dir / "checkpoints"

            for d in (exp_dir, datasets_dir, metrics_dir, vis_dir, checkpoints_dir):
                d.mkdir(parents=True, exist_ok=True)

            experiment_header(exp_counter, total_experiments, ds_name, seq_name)
            print(f"  Description: {seq_cfg['description']}")

            # Step 3: create sequence datasets
            create_sequence_datasets(
                processed_dir, seq_cfg, datasets_dir, common, args.skip_existing
            )

            # Step 4: build config + train
            config_file = build_experiment_config(
                exp_name, seq_cfg, datasets_dir, exp_dir, common
            )
            all_configs.append((exp_name, config_file, exp_dir))

            checkpoint = exp_dir / "checkpoints" / f"{exp_name}.pt"
            if args.skip_existing and checkpoint.exists():
                print(f"  [skip] Checkpoint already exists: {checkpoint}")
            else:
                print(f"\n  Training [{exp_counter}/{total_experiments}]  "
                      f"{ds_name} × {seq_name}  "
                      f"(epochs={common['epochs']})")
                train_model(config_path=config_file)

            # Step 5: evaluate
            metrics_json = metrics_dir / "metrics.json"
            if args.skip_existing and metrics_json.exists():
                print(f"  [skip] Metrics already exist: {metrics_json}")
                with open(metrics_json) as f:
                    metrics = json.load(f)
            else:
                label = f"{exp_counter}/{total_experiments} | {ds_name} × {seq_name}"
                metrics, error_grid = evaluate_from_config(
                    config_file,
                    output_dir=str(metrics_dir),
                    experiment_label=label
                )

            all_results[ds_name][seq_name] = metrics

            # ── Per-experiment visualizations ─────────────────────────────────

            # Training curve
            csv_log = exp_dir / "checkpoints" / f"{exp_name}.csv"
            if csv_log.exists():
                plot_training_curves(
                    csv_log,
                    vis_dir / "training_curve.png",
                    title=f"Training — {ds_name} / {seq_name}"
                )

            # Error heatmap
            error_grid_path = metrics_dir / "error_grid.npy"
            if error_grid_path.exists():
                plot_error_heatmap(
                    error_grid_path,
                    vis_dir / "error_heatmap.png",
                    title=f"Error heatmap — {ds_name} / {seq_name}"
                )

    # ── Final comparison visualizations ──────────────────────────────────────
    banner("Generating comparison visualizations")
    comparison_dir = base_out / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    plot_early_vs_late_comparison(all_results, comparison_dir)

    # Save aggregated metrics JSON
    summary_path = base_out / "all_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All metrics saved → {summary_path}")

    # ── Final summary table ───────────────────────────────────────────────────
    banner("Results Summary")
    header = f"{'Dataset':<35} {'Seq':<15} {'MSE':>8} {'MAE':>8} {'F1':>7}"
    print(header)
    print("─" * len(header))

    for ds_name, seq_results in all_results.items():
        for seq_name in ["early_stage_3", "early_stage_5", "late_stage_10"]:
            m = seq_results.get(seq_name, {})
            print(
                f"{ds_name:<35} {seq_name:<15} "
                f"{m.get('mse', float('nan')):>8.5f} "
                f"{m.get('mae', float('nan')):>8.5f} "
                f"{m.get('f1',  float('nan')):>7.4f}"
            )

    banner(f"Done — {total_experiments} experiments completed")
    print(f"  Results in: {base_out}\n")


if __name__ == "__main__":
    main()
