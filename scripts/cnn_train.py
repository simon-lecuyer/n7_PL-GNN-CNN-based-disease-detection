#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from datetime import datetime
from xml.parsers.expat import model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from cnn_model import DiseaseCNN
from cnn_dataset import TemporalCNNDataset

# We parse the arguments of the command line  

def parse_args():
    parser = argparse.ArgumentParser("Train temporal CNN on infection level prediction")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--variant", type=str, default="base", help="Model variant (base, etc.)")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--sequence_length", type=int, default=5, help="Number of frames in temporal sequence (L)")
    parser.add_argument(
        "--require_consecutive",
        action="store_true",
        help="Require consecutive timesteps in sequences"
    )
    return parser.parse_args()


# Training

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(args.processed_dir)
    datasets_dir = processed_dir / "datasets" / "cnn"

    train_json = datasets_dir / "train.json"
    val_json = datasets_dir / "val.json"

    assert train_json.exists(), f"Missing {train_json}"
    assert val_json.exists(), f"Missing {val_json}"

    # Dataset - Temporal sequences
    train_ds = TemporalCNNDataset(
        str(train_json),
        sequence_length=args.sequence_length,
        require_consecutive=args.require_consecutive
    )
    val_ds = TemporalCNNDataset(
        str(val_json),
        sequence_length=args.sequence_length,
        require_consecutive=args.require_consecutive
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model - 3D CNN for temporal prediction
    model = DiseaseCNN(in_frames=args.sequence_length, out_channels=1).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create checkpoint directory
    checkpoint_dir = Path("cnn/checkpoints") / f"cnn_{args.variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'model': 'DiseaseCNN_Temporal_3D',
        'variant': args.variant,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'sequence_length': args.sequence_length,
        'device': str(device),
    }
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("────────────────────────────────────")
    print(f"Device:        {device}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint:    {checkpoint_dir}")
    print("────────────────────────────────────")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)  # (B, L, 64, 64)
            y = y.to(device).unsqueeze(1)  # (B,) → (B, 1)

            y = y.to(device).unsqueeze(1)          # (B, 64, 64) → (B, 1, 64, 64)
            pred = model(x)                         # (B, 1, 64, 64)
            loss = criterion(pred, y)              # compare full maps directly

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        #  Validation
        model.eval()
        val_loss = 0.0

        if len(val_loader) == 0:
            val_loss = float("nan")
            history['val_loss'].append(val_loss)
            print("Warning: empty validation set (check require_consecutive or sequence_length).")
        else:
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)  # (B, L, 64, 64)
                    y = y.to(device).unsqueeze(1)  # (B,) → (B, 1)

                    pred = model(x)  # (B, 1, 64, 64)
                    pred_scalar = pred.mean(dim=(2, 3))  # (B, 1) - moyenne spatiale
                    val_loss += criterion(pred_scalar, y).item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    with open(checkpoint_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("✓ Training finished")


if __name__ == "__main__":
    train()