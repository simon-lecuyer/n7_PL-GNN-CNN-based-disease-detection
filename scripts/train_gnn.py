import sys
import os
import yaml
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

    # 3. Prepare Data
    print("Loading Datasets...")
    try:
        train_dataset = DiseaseDetectionDataset(config['data']['train_path'], format='gnn', return_metadata=True)
        val_dataset = DiseaseDetectionDataset(config['data']['val_path'], format='gnn', return_metadata=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("   Did you run scripts/create_datasets.py?")
        return

    train_loader = get_dataloader(
        "data/processed/processed_20260211_121548/datasets/gnn/train.json",
        format="gnn",
        batch_size=16,
        shuffle=True
    )
    val_loader = get_dataloader(
        "data/processed/processed_20260211_121548/datasets/gnn/train.json",
        format="gnn",
        batch_size=16,
        shuffle=True
    )

    # 4. Initialize Model & Move to Device
    print("🧠 Initializing SpatioTemporalGNN...")
    model_cfg = config["model"]
    model = DiseaseGNN(
        in_channels=model_cfg["in_channels"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        out_channels=model_cfg["out_channels"],
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
    # Create runs directory if it doesn't exist
    log_dir = f"runs/{config['experiment_name']}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    epochs = config['training']['epochs']

    model.train()
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:03d} | Loss = {avg_loss:.6f}")
        
    #     # Training Step
    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    #     for batch in pbar:
    #         print(batch)
    #         # CRITICAL: Move batch to GPU
    #         batch = batch.to(device)
            
    #         optimizer.zero_grad()
            
    #         # Forward pass
    #         # We pass edge_weight if it exists (for distance-weighted graphs)
    #         out = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
            
    #         # Reshape target if necessary to match output [N, 1]
    #         loss = criterion(out, batch.y)
            
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
            
    #         pbar.set_postfix({'loss': loss.item()})

    #     avg_train_loss = total_loss / len(train_loader)
        
    #     # Validation Step
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             # CRITICAL: Move batch to GPU
    #             batch = batch.to(device)
                
    #             out = model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None), batch.batch)
    #             loss = criterion(out, batch.y)
    #             val_loss += loss.item()
        
    #     avg_val_loss = val_loss / len(val_loader)
        
    #     # Logging
    #     print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    #     writer.add_scalar('Loss/train', avg_train_loss, epoch)
    #     writer.add_scalar('Loss/val', avg_val_loss, epoch)

    #     # Save Checkpoint (e.g., every 10 epochs or if best val loss)
    #     if (epoch + 1) % 10 == 0:
    #         os.makedirs("checkpoints", exist_ok=True)
    #         ckpt_path = f"checkpoints/{config['experiment_name']}_ep{epoch+1}.pth"
    #         torch.save(model.state_dict(), ckpt_path)

    # # Final Save
    # os.makedirs("checkpoints", exist_ok=True)
    # torch.save(model.state_dict(), f"checkpoints/{config['experiment_name']}_final.pth")
    # print("✅ Training Complete. Model Saved.")
    # writer.close()

if __name__ == "__main__":
    train()