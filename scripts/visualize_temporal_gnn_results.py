#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from gnn.models.temporal_disease_gnn import TemporalDiseaseGNN
from scripts.train_gnn import get_device
from utils.datasets import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PNG plots for trained temporal GNN models")
    parser.add_argument(
        "--config_glob",
        type=str,
        default="configs/gnn_*_seq*.yaml",
        help="Glob pattern for model config files",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/gnn_experiments",
        help="Root folder for PNG outputs",
    )
    parser.add_argument(
        "--samples_per_model",
        type=int,
        default=3,
        help="How many test samples to save per model",
    )
    parser.add_argument(
        "--align_with_cnn",
        action="store_true",
        help="Select GNN samples using the same sim_ids as the matching CNN test.json and start_timestep=0",
    )
    return parser.parse_args()


def safe_shape(shape_obj):
    if isinstance(shape_obj, (list, tuple)) and len(shape_obj) == 2:
        return int(shape_obj[0]), int(shape_obj[1])
    if torch.is_tensor(shape_obj) and shape_obj.numel() >= 2:
        flat = shape_obj.flatten().tolist()
        return int(flat[0]), int(flat[1])
    raise ValueError(f"Unsupported shape format: {shape_obj}")


def build_output_dir(output_root, exp_name, seq_len):
    prefix = "gnn_"
    suffix = f"_seq{seq_len}_ep10"
    clean_name = exp_name
    if clean_name.startswith(prefix):
        clean_name = clean_name[len(prefix):]
    if clean_name.endswith(suffix):
        clean_name = clean_name[: -len(suffix)]
    return Path(output_root) / clean_name / f"seq{seq_len}"


def load_model(config, device):
    model_cfg = config["model"]
    model = TemporalDiseaseGNN(
        in_channels=model_cfg["in_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        out_channels=model_cfg["out_channels"],
        dropout=model_cfg["dropout"],
    ).to(device)
    ckpt = Path(config["data"]["checkpoint_path"]) / f"{config['experiment_name']}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def load_gnn_graph_file(path: str, device):
    data = np.load(path, allow_pickle=True).item()
    return {
        "nodes": torch.from_numpy(data["nodes"]),
        "edges": torch.from_numpy(data["edges"]).long().to(device),
        "node_features": torch.from_numpy(data["node_features"]).float().to(device),
        "shape": data["shape"],
    }


def infer_seq_len_from_test_path(test_path: str) -> int:
    seq_token = "_seq"
    if seq_token in test_path:
        return int(test_path.split(seq_token)[1].split(".")[0])
    raise ValueError(f"Cannot infer sequence length from path: {test_path}")


def corresponding_cnn_test_path(gnn_test_seq_path: str) -> Path:
    # .../datasets/gnn/test_seqN.json -> .../datasets/cnn/test.json
    return Path(gnn_test_seq_path.replace("/datasets/gnn/test_seq", "/datasets/cnn/test")).with_name("test.json")


def select_samples_from_json(
    gnn_test_seq_path: str,
    samples_per_model: int,
    align_with_cnn: bool,
) -> List[dict]:
    with open(gnn_test_seq_path, "r", encoding="utf-8") as f:
        gnn_data = json.load(f)

    seq_samples = gnn_data.get("samples", [])

    if not align_with_cnn:
        # Default deterministic selection: first N from test_seq file
        return seq_samples[:samples_per_model]

    cnn_test = corresponding_cnn_test_path(gnn_test_seq_path)
    if not cnn_test.exists():
        print(f"[WARN] CNN test file not found for alignment: {cnn_test}. Falling back to first samples.")
        return seq_samples[:samples_per_model]

    with open(cnn_test, "r", encoding="utf-8") as f:
        cnn_data = json.load(f)

    cnn_sim_ids = sorted({s["sim_id"] for s in cnn_data.get("samples", [])})

    by_key: Dict[Tuple[int, int], dict] = {}
    for s in seq_samples:
        by_key[(int(s["sim_id"]), int(s["start_timestep"]))] = s

    selected = []
    for sim_id in cnn_sim_ids:
        key = (int(sim_id), 0)
        if key in by_key:
            selected.append(by_key[key])
        if len(selected) >= samples_per_model:
            break

    if not selected:
        print("[WARN] No aligned GNN samples found with start_timestep=0. Falling back to first samples.")
        return seq_samples[:samples_per_model]

    return selected


def save_plot(seq, target, pred, meta, out_path):
    y_true = target["node_features"][:, 0].detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()

    h, w = safe_shape(target["shape"])
    true_img = y_true.reshape(h, w)
    pred_img = pred_np.reshape(h, w)
    abs_err = np.abs(pred_img - true_img)

    mae = float(np.mean(abs_err))
    mse = float(np.mean((pred_img - true_img) ** 2))
    rmse = float(np.sqrt(mse))

    seq_len = len(seq)
    fig, axes = plt.subplots(1, seq_len + 3, figsize=(3 * (seq_len + 3), 4))

    for t in range(seq_len):
        img = seq[t]["node_features"][:, 0].detach().cpu().numpy().reshape(h, w)
        axes[t].imshow(img, cmap="YlOrRd", vmin=0, vmax=1)
        axes[t].set_title(f"t-{seq_len - t}")
        axes[t].axis("off")

    axes[seq_len].imshow(true_img, cmap="YlOrRd", vmin=0, vmax=1)
    axes[seq_len].set_title("Target")
    axes[seq_len].axis("off")

    axes[seq_len + 1].imshow(pred_img, cmap="YlOrRd", vmin=0, vmax=1)
    axes[seq_len + 1].set_title("Prediction")
    axes[seq_len + 1].axis("off")

    axes[seq_len + 2].imshow(abs_err, cmap="YlOrRd", vmin=0, vmax=1)
    axes[seq_len + 2].set_title(f"Error\nMAE={mae:.4f}\nRMSE={rmse:.4f}")
    axes[seq_len + 2].axis("off")

    sim_id = meta.get("sim_id", "unknown")
    start_ts = meta.get("start_timestep", "unknown")
    plt.suptitle(f"sim={sim_id} | start_t={start_ts}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {"mae": mae, "mse": mse, "rmse": rmse, "sim_id": sim_id, "start_timestep": start_ts}


def visualize_one_config(config_path, output_root, samples_per_model, align_with_cnn=False):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("model", {}).get("type") != "temporal":
        print(f"[SKIP] not temporal: {config_path}")
        return None

    test_seq_path = str(config["data"]["test_path"])
    seq_len = int(config["training"].get("sequence_length", 0))
    if seq_len == 0:
        seq_len = infer_seq_len_from_test_path(test_seq_path)

    device = get_device(config["training"].get("device", "auto"))
    model = load_model(config, device)

    out_dir = build_output_dir(output_root, config["experiment_name"], seq_len)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = select_samples_from_json(
        gnn_test_seq_path=test_seq_path,
        samples_per_model=samples_per_model,
        align_with_cnn=align_with_cnn,
    )

    sample_metrics = []
    with torch.no_grad():
        for i, sample in enumerate(selected_samples, start=1):
            seq = [load_gnn_graph_file(s["file"], device) for s in sample["sequence"]]
            target = load_gnn_graph_file(sample["target"]["file"], device)
            meta = {
                "sim_id": sample.get("sim_id", "unknown"),
                "start_timestep": sample.get("start_timestep", "unknown"),
            }

            pred = model(seq)
            png_name = f"sample_{i:02d}_sim{meta.get('sim_id', 'x')}_t{meta.get('start_timestep', 'x')}.png"
            png_path = out_dir / png_name
            metrics = save_plot(seq, target, pred, meta, png_path)
            metrics["file"] = str(png_path)
            sample_metrics.append(metrics)

    if sample_metrics:
        run_summary = {
            "config": config_path,
            "experiment_name": config["experiment_name"],
            "saved_samples": len(sample_metrics),
            "mean_mae": float(np.mean([m["mae"] for m in sample_metrics])),
            "mean_mse": float(np.mean([m["mse"] for m in sample_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in sample_metrics])),
            "samples": sample_metrics,
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
        print(f"[OK] {config['experiment_name']} -> {out_dir} ({len(sample_metrics)} PNG)")
        return run_summary

    print(f"[WARN] no sample saved for {config['experiment_name']}")
    return None


def main():
    args = parse_args()
    configs = sorted(glob.glob(args.config_glob))

    if not configs:
        print(f"No config found with glob: {args.config_glob}")
        return

    all_summaries = []
    for cfg in configs:
        try:
            summary = visualize_one_config(
                cfg,
                args.output_root,
                args.samples_per_model,
                align_with_cnn=args.align_with_cnn,
            )
            if summary is not None:
                all_summaries.append(summary)
        except Exception as exc:
            print(f"[ERROR] {cfg}: {exc}")

    if all_summaries:
        global_summary = {
            "num_models": len(all_summaries),
            "models": all_summaries,
        }
        out_path = Path(args.output_root) / "visualization_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2)
        print(f"\nGlobal summary saved to: {out_path}")


if __name__ == "__main__":
    main()
