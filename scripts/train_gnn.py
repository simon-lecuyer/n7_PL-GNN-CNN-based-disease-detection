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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Assumes you have created these files as discussed
from utils.datasets import DiseaseDetectionDataset, get_dataloader
from gnn.models.disease_gnn import DiseaseGNN
from gnn.models.temporal_disease_gnn import TemporalDiseaseGNN    

def get_device(cfg_device="auto"):
    """
    Select computation device.
    Priority: CUDA > MPS > CPU
    """

    if cfg_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available, using CPU.")
            return torch.device("cpu")

    if cfg_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS requested but not available, using CPU.")
            return torch.device("cpu")

    if cfg_device == "cpu":
        return torch.device("cpu")

    # Auto mode
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train(config_path="configs/gnn_config.yaml"):
    # 1. Load Config
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = get_device(config['training'].get('device', 'auto'))
    print(f"Training on device: {device}")
    is_temporal = (config["model"]["type"]=="temporal")

    # 3. Prepare Data
    train_loader = get_dataloader(
        config["data"]["train_path"],
        format="gnn",
        batch_size=16,
        shuffle=True,
        is_temporal=is_temporal
    )
    val_loader = get_dataloader(
        config["data"]["val_path"],
        format="gnn",
        batch_size=16,
        shuffle=True,
        is_temporal=is_temporal
    )

    # 4. Initialize Model & Move to Device
    print("Initializing GNN...")
    model_cfg = config["model"]
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
            dropout=model_cfg["dropout"]
        ).to(device)

    # 5. Optimizer & Loss
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['lr']
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    criterion = nn.MSELoss()

    # 6. Training Loop
    log_dir = f"runs/{config['experiment_name']}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    epochs = config['training']['epochs']

    model.train()
    print("Starting training...")
    if model_cfg["type"] == "spatial":
        for epoch in range(epochs):
            total_loss = 0.0

            for graphs in train_loader:
                batch_loss = 0.0

                for graph in graphs: 
                    x = graph["node_features"].float().to(device)
                    edge_index = graph["edges"].long().to(device)
                    if edge_index.shape[0] != 2:
                        edge_index = edge_index.t()
                    y=x[:, 0].unsqueeze(1)

                    pred = model(x, edge_index)
                    loss = criterion(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                
                total_loss += batch_loss

            train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1:03d} | Loss = {train_loss:.6f}")
    else:
        for epoch in range(epochs):
            total_loss = 0.0
            for sequence, target, metadata in train_loader:
                sequence = sequence[0]  # remove batch wrapper
                target = target[0]
                for g in sequence:
                    g["node_features"] = g["node_features"].to(device)
                    g["edges"] = g["edges"].to(device)

                target["node_features"] = target["node_features"].to(device)
                target["edges"] = target["edges"].to(device)

                # Target infection values
                y = target["node_features"][:, 0].unsqueeze(1)
                pred = model(sequence)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1:03d}] Train Loss = {train_loss:.6f}")

    val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
            temporal=(model_cfg["type"] == "temporal")
        )
    
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    print(f"\nEpoch {epoch+1:03d}")
    print(f"   Train Loss: {train_loss:.6f}")
    print(f"   Val Loss:   {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        ckpt_path = os.path.join(
            ckpt_dir,
            f"{config['experiment_name']}_best.pth"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
            "config": config
        }, ckpt_path)

        print(f"   ✅ Saved new best checkpoint: {ckpt_path}")

def evaluate(model, val_loader, criterion, device, temporal=False):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        if not temporal:
            for graphs in val_loader:
                for graph in graphs:
                    x = graph["node_features"].float().to(device)

                    edge_index = graph["edges"].long().to(device)
                    if edge_index.shape[0] != 2:
                        edge_index = edge_index.t()

                    y = x[:, 0].unsqueeze(1)

                    pred = model(x, edge_index)
                    loss = criterion(pred, y)

                    total_loss += loss.item()

        else:
            for sequence, target, metadata in val_loader:

                # Fix batch wrapper
                sequence = sequence[0]
                target = target

                if isinstance(target, list):
                    target = target[0]

                # Move graphs
                for g in sequence:
                    g["node_features"] = g["node_features"].float().to(device)
                    g["edges"] = g["edges"].long().to(device)

                target["node_features"] = target["node_features"].float().to(device)
                target["edges"] = target["edges"].long().to(device)

                y = target["node_features"][:, 0].unsqueeze(1)

                pred = model(sequence)
                loss = criterion(pred, y)

                total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    train()