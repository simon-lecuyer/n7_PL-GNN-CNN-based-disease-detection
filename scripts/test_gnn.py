import sys
import os
import yaml
import argparse
import torch
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
from scripts.train_gnn import get_device

checkpoint_dir = "gnn/checkpoints"

def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN training"
    )
    
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/gnn_config.yaml",
        help="Config file path"
    )

    return parser.parse_args()

def test_model(config_path="configs/gnn_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = get_device(config['training'].get('device', 'auto'))
    print(f"Training on device: {device}")
    is_temporal = (config["model"]["type"]=="temporal")

    test_loader = get_dataloader(
        config["data"]["test_path"],
        format="gnn",
        batch_size=16,
        shuffle=True,
        is_temporal=is_temporal
    )

    print("Testing GNN...")
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
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    criterion = nn.MSELoss()
    model.eval()
    total_test_loss = 0.0

    val_pbar = tqdm(test_loader, desc=f"Testing model after {checkpoint['epoch']} epoch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['training']['seed'])
        with torch.no_grad():
            if model_cfg["type"] == "spatial":
                mask_threshold = 0.3
                if 'mask' in config['training'].keys():
                    mask_threshold = config['training']['mask']
                for graphs in val_pbar:

                    batch_loss = 0.0

                    for graph in graphs: 

                        x = graph["node_features"].float().to(device)
                        edge_index = graph["edges"].long().to(device)

                        if edge_index.shape[0] != 2:
                            edge_index = edge_index.t()

                        y = x[:, 0].unsqueeze(1)

                        num_nodes = x.size(0)
                        hidden_nodes_mask = torch.rand(num_nodes, device=device) < mask_threshold

                        x_masked = x.clone()
                        x_masked[hidden_nodes_mask, 0] = 0.0

                        observed_flag = torch.ones(num_nodes, 1, device=device)
                        observed_flag[hidden_nodes_mask] = 0.0

                        x_augmented = torch.cat([x_masked, observed_flag], dim=1)

                        pred = model(x_augmented, edge_index)

                        loss = criterion(pred[hidden_nodes_mask], y[hidden_nodes_mask])

                        batch_loss += loss.item()

                    total_test_loss += batch_loss
                    val_pbar.set_postfix(val_loss=batch_loss)
                avg_test_loss = total_test_loss/len(test_loader)
                print(f"Test loss = {avg_test_loss:.6f}")

            else:
                for sequence_batch, target_batch, _ in val_pbar:

                    batch_loss = 0.0
                    for seq, target in zip(sequence_batch, target_batch):
                        for graph in seq:
                            graph["node_features"] = graph["node_features"].float().to(device)
                            graph["edges"] = graph["edges"].long().to(device)

                        target["node_features"] = target["node_features"].float().to(device)

                        pred = model(seq)
                        y = target["node_features"][:, 0].unsqueeze(1)
                        loss = criterion(pred, y)
                        batch_loss += loss.item()

                    total_test_loss += batch_loss
                    val_pbar.set_postfix(val_loss=batch_loss)
                    
                avg_test_loss = total_test_loss/len(test_loader)
                print(f"Test loss = {avg_test_loss:.6f}")

    return

def main():
    args = parse_args()
    cfg_path = args.cfg
    test_model(config_path=cfg_path)

if __name__ == "__main__":
    main()
