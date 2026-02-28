import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import yaml
import argparse
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm import tqdm
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.datasets import DiseaseDetectionDataset, get_dataloader
from gnn.models.disease_gnn import DiseaseGNN
from gnn.models.temporal_disease_gnn import TemporalDiseaseGNN   

from scripts.train_gnn import get_device, parse_args, checkpoint_dir

def plot_temporal_gnn_sample(config_path="configs/gnn_config.yaml", grid_size=None):

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = get_device(config['training'].get('device', 'auto'))
    is_temporal = (config["model"]["type"]=="temporal")

    test_loader = get_dataloader(
        config["data"]["test_path"],
        format="gnn",
        batch_size=16,
        shuffle=True,
        is_temporal=is_temporal
    )

    model_cfg = config["model"]

    name = config["experiment_name"]
    checkpoint_path = config["data"]["checkpoint_path"]
    checkpoint_path = f"{checkpoint_path}/{name}.pt"
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    sequences, targets, md = next(iter(test_loader))

    batch_size = len(targets)
    seq_len = len(sequences)
    idx = random.randint(0, batch_size - 1)

    seq = sequences[idx]
    target = targets[idx]
    meta = md[idx]
    
    for graph in seq:
        graph["node_features"] = graph["node_features"].float().to(device)
        graph["edges"] = graph["edges"].long().to(device)

    target["node_features"] = target["node_features"].float().to(device)

    with torch.no_grad():
        pred = model(seq)

    pred = pred.squeeze().cpu().numpy()

    y_true = target["node_features"][:, 0].cpu().numpy()

    H, W = target["shape"]

    pred_img = pred.reshape(H, W)
    true_img = y_true.reshape(H, W)
    error_img = np.abs(pred_img - true_img)

    mae = np.mean(error_img)

    seq_len = len(seq)
    fig, axes = plt.subplots(1, seq_len + 3, figsize=(3*(seq_len+3), 4))
    for t in range(seq_len):
        img = seq[t]["node_features"][:, 0].cpu().numpy().reshape(H, W)
        axes[t].imshow(img, vmin=0, vmax=1)
        axes[t].set_title(f"t-{seq_len-t}")
        axes[t].axis("off")

    # Target
    axes[seq_len].imshow(true_img, vmin=0, vmax=1)
    axes[seq_len].set_title("Target (t+1)")
    axes[seq_len].axis("off")

    # Prediction
    axes[seq_len+1].imshow(pred_img, vmin=0, vmax=1)
    axes[seq_len+1].set_title("Prediction")
    axes[seq_len+1].axis("off")

    # Error
    axes[seq_len+2].imshow(error_img, vmin=0, vmax=1)
    axes[seq_len+2].set_title(f"Error\nMAE={mae:.4f}")
    axes[seq_len+2].axis("off")

    plt.suptitle(
        f"Simulation {meta['sim_id']} | start_t={meta['start_timestep']}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()



def main():
    args = parse_args()
    cfg_path = args.cfg
    plot_temporal_gnn_sample(config_path=cfg_path)

if __name__ == "__main__":
    main()