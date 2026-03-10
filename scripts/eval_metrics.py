import sys
import os
import json
import argparse

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gnn.models.disease_gnn import DiseaseGNN
from gnn.models.temporal_disease_gnn import TemporalDiseaseGNN
from utils.datasets import get_dataloader
from scripts.train_gnn import get_device


def evaluate(model, test_loader, device, model_type, mask_threshold=0.3, seed=42):
    """
    Run inference on the test set and collect predictions, targets, and node positions.

    For spatial GNN:  masks a fraction of nodes and evaluates on those masked nodes.
    For temporal GNN: predicts the next timestep from a sequence of graphs.

    Args:
        model:          trained GNN model (DiseaseGNN or TemporalDiseaseGNN)
        test_loader:    DataLoader for the test split
        device:         torch device
        model_type:     "spatial" or "temporal"
        mask_threshold: fraction of nodes to mask (spatial only)
        seed:           random seed for reproducible masking

    Returns:
        preds:     np.array (total_evaluated_nodes,) — predicted infection probability
        targets:   np.array (total_evaluated_nodes,) — true infection values
        positions: np.array (total_evaluated_nodes, 2) — normalized (x, y) node coordinates
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_positions = []

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():

            if model_type == "spatial":
                for graphs in tqdm(test_loader, desc="Evaluating (spatial)"):
                    for graph in graphs:
                        x = graph["node_features"].float().to(device)
                        edge_index = graph["edges"].long().to(device)

                        if edge_index.shape[0] != 2:
                            edge_index = edge_index.t()

                        y = x[:, 0]  # infection channel — shape (N,)
                        num_nodes = x.size(0)

                        hidden_nodes_mask = torch.rand(num_nodes, device=device) < mask_threshold

                        x_masked = x.clone()
                        x_masked[hidden_nodes_mask, 0] = 0.0
                        observed_flag = torch.ones(num_nodes, 1, device=device)
                        observed_flag[hidden_nodes_mask] = 0.0

                        x_augmented = torch.cat([x_masked, observed_flag], dim=1)
                        pred = model(x_augmented, edge_index)  # (N, 1)

                        # Only evaluate on masked (hidden) nodes
                        pred_masked = pred[hidden_nodes_mask].squeeze(1)
                        y_masked = y[hidden_nodes_mask]

                        all_preds.append(pred_masked.cpu().numpy())
                        all_targets.append(y_masked.cpu().numpy())

                        # x[:, 1] = x_coord, x[:, 2] = y_coord (normalized)
                        positions = x[hidden_nodes_mask, 1:3].cpu().numpy()
                        all_positions.append(positions)

            else:  # temporal
                for sequence_batch, target_batch, _ in tqdm(test_loader, desc="Evaluating (temporal)"):
                    for seq, target in zip(sequence_batch, target_batch):
                        for graph in seq:
                            graph["node_features"] = graph["node_features"].float().to(device)
                            graph["edges"] = graph["edges"].long().to(device)

                        target["node_features"] = target["node_features"].float().to(device)

                        pred = model(seq)  # (N, 1)
                        y = target["node_features"][:, 0]  # (N,)

                        all_preds.append(pred.squeeze(1).cpu().numpy())
                        all_targets.append(y.cpu().numpy())

                        # Use last graph in sequence for positions
                        positions = seq[-1]["node_features"][:, 1:3].cpu().numpy()
                        all_positions.append(positions)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    positions = np.concatenate(all_positions)

    return preds, targets, positions


def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute regression and binary classification metrics from predictions.

    Regression:
        MSE and MAE over all predicted nodes.

    Classification:
        Binarize predictions at `threshold` (model output is sigmoid → [0,1]).
        Compute precision, recall, F1, accuracy from the confusion matrix.

    Args:
        preds:     (N,) predicted infection probabilities
        targets:   (N,) true infection values
        threshold: binarization cutoff (default 0.5)

    Returns:
        dict with keys: mse, mae, precision, recall, f1, accuracy, tp, fp, fn, tn, n_samples
    """
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))

    pred_binary = (preds >= threshold).astype(int)
    target_binary = (targets >= threshold).astype(int)

    tp = int(np.sum((pred_binary == 1) & (target_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (target_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (target_binary == 1)))
    tn = int(np.sum((pred_binary == 0) & (target_binary == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(preds) if len(preds) > 0 else 0.0

    return {
        "mse":       mse,
        "mae":       mae,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "accuracy":  float(accuracy),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_samples": len(preds)
    }


def compute_per_node_errors(preds, targets, positions, grid_size=64):
    """
    Accumulate per-position mean absolute error for error heatmap generation.

    Node positions are normalized to [0,1]. If grid_size is provided, positions
    are projected onto a 2D grid of that resolution so the result can be plotted
    as an image alongside CNN error heatmaps.

    Args:
        preds:     (N,) predicted values
        targets:   (N,) true values
        positions: (N, 2) normalized (x, y) node coordinates
        grid_size: resolution of the output grid

    Returns:
        node_errors: dict mapping "(x,y)" string key to mean absolute error
        error_grid:  np.array (grid_size, grid_size) — MAE projected onto grid
    """
    errors = np.abs(preds - targets)

    pos_errors = {}
    for pos, err in zip(positions, errors):
        key = (round(float(pos[0]), 4), round(float(pos[1]), 4))
        if key not in pos_errors:
            pos_errors[key] = []
        pos_errors[key].append(float(err))

    node_errors = {str(k): float(np.mean(v)) for k, v in pos_errors.items()}

    error_grid = np.zeros((grid_size, grid_size))
    count_grid = np.zeros((grid_size, grid_size))

    for (px, py), errs in pos_errors.items():
        xi = min(int(px * grid_size), grid_size - 1)
        yi = min(int(py * grid_size), grid_size - 1)
        error_grid[yi, xi] += np.mean(errs)
        count_grid[yi, xi] += 1

    mask = count_grid > 0
    error_grid[mask] /= count_grid[mask]

    return node_errors, error_grid


def evaluate_from_config(config_path, output_dir=None, experiment_label=None):
    """
    Full evaluation pipeline: load config → load model → run test set → compute and save metrics.

    Can be called standalone (via CLI) or by run_gnn_experiments.py for each experiment.

    Args:
        config_path:       path to a YAML config file (same format as gnn_config.yaml)
        output_dir:        directory to save metrics.json and error_grid.npy
        experiment_label:  optional prefix for console output (e.g. "[3/15 | fast_spread × seq=5]")

    Returns:
        metrics:    dict of all computed metrics
        error_grid: np.array (64, 64) of per-position MAE
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    label = f"[{experiment_label}] " if experiment_label else ""

    device = get_device(config['training'].get('device', 'auto'))
    is_temporal = (config["model"]["type"] == "temporal")
    model_cfg = config["model"]

    test_loader = get_dataloader(
        config["data"]["test_path"],
        format="gnn",
        batch_size=16,
        shuffle=False,
        is_temporal=is_temporal
    )

    if not is_temporal:
        model = DiseaseGNN(
            in_channels=model_cfg["in_channels"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            out_channels=model_cfg["out_channels"],
            dropout=model_cfg["dropout"]
        ).to(device)
    else:
        model = TemporalDiseaseGNN(
            in_channels=model_cfg["in_channels"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            out_channels=model_cfg["out_channels"],
            dropout=model_cfg["dropout"]
        ).to(device)

    name = config["experiment_name"]
    checkpoint_path = os.path.join(config["data"]["checkpoint_path"], f"{name}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"{label}Loaded checkpoint — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")

    mask_threshold = config['training'].get('mask', 0.3)
    seed = config['training'].get('seed', 42)

    preds, targets, positions = evaluate(
        model, test_loader, device,
        model_type=config["model"]["type"],
        mask_threshold=mask_threshold,
        seed=seed
    )

    metrics = compute_metrics(preds, targets)
    metrics["experiment_name"] = name
    metrics["model_type"] = config["model"]["type"]
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])

    print(f"{label}MSE={metrics['mse']:.6f}  MAE={metrics['mae']:.6f}  "
          f"F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}")

    _, error_grid = compute_per_node_errors(preds, targets, positions, grid_size=64)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{label}Metrics saved → {metrics_path}")

        grid_path = os.path.join(output_dir, "error_grid.npy")
        np.save(grid_path, error_grid)
        print(f"{label}Error grid saved → {grid_path}")

    return metrics, error_grid


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN on its test set")
    parser.add_argument("--cfg", type=str, default="configs/gnn_config.yaml",
                        help="Path to the YAML config used for training")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save metrics.json and error_grid.npy")
    args = parser.parse_args()

    evaluate_from_config(args.cfg, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
