#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Obtenir les stats de normalisation du dataset pour denormaliser les prédictions
    target_mean = train_ds.target_mean
    target_std = train_ds.target_std

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

    # Analyser les distributions des targets
    train_targets = [train_ds.sequences[i][1]['infection_level'] for i in range(len(train_ds))]
    val_targets = [val_ds.sequences[i][1]['infection_level'] for i in range(len(val_ds))]
    
    train_sims = set([train_ds.sequences[i][1]['sim_id'] for i in range(len(train_ds))])
    val_sims = set([val_ds.sequences[i][1]['sim_id'] for i in range(len(val_ds))])
    
    print("────────────────────────────────────")
    print(f"Device:        {device}")
    print(f"Train samples: {len(train_ds)} (sims: {sorted(train_sims)[:5]}...)")
    print(f"Val samples:   {len(val_ds)} (sims: {sorted(val_sims)[:5]}...)")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n📊 Target distributions:")
    print(f"Train - min: {min(train_targets):.6f}, max: {max(train_targets):.6f}, mean: {np.mean(train_targets):.6f}")
    print(f"Val   - min: {min(val_targets):.6f}, max: {max(val_targets):.6f}, mean: {np.mean(val_targets):.6f}")
    overlap = train_sims & val_sims
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} simulations appear in both train and val!")
    else:
        print(f"✅ No overlap: train has {len(train_sims)} sims, val has {len(val_sims)} sims")
    print(f"\nCheckpoint:    {checkpoint_dir}")
    print("────────────────────────────────────")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets_epoch = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)  # (B, L, 64, 64)
            y = y.to(device).unsqueeze(1)  # (B,) → (B, 1)

            pred = model(x)  # (B, 1)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Accuracy: relative error < 15%
            relative_error = torch.abs(pred - y) / (torch.abs(y) + 1e-6)
            correct = (relative_error < 0.15).float().sum()
            train_correct += correct.item()
            train_total += y.size(0)
            
            train_preds.extend(pred.detach().cpu().numpy().flatten())
            train_targets_epoch.extend(y.detach().cpu().numpy().flatten())

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        #  Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets_epoch = []

        if len(val_loader) == 0:
            val_loss = float("nan")
            val_acc = 0.0
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print("Warning: empty validation set (check require_consecutive or sequence_length).")
        else:
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)  # (B, L, 64, 64)
                    y = y.to(device).unsqueeze(1)  # (B,) → (B, 1)

                    pred = model(x)  # (B, 1)
                    val_loss += criterion(pred, y).item()
                    
                    # Accuracy: relative error < 15%
                    relative_error = torch.abs(pred - y) / (torch.abs(y) + 1e-6)
                    correct = (relative_error < 0.15).float().sum()
                    val_correct += correct.item()
                    val_total += y.size(0)
                    
                    val_preds.extend(pred.detach().cpu().numpy().flatten())
                    val_targets_epoch.extend(y.detach().cpu().numpy().flatten())

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        # Diagnostic des prédictions
        # Dénormaliser pour affichage
        train_preds_denorm = [p * target_std + target_mean for p in train_preds]
        val_preds_denorm = [p * target_std + target_mean for p in val_preds] if val_preds else []
        
        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.6f} Acc: {train_acc:.2%} | "
            f"Val MSE: {val_loss:.6f} Acc: {val_acc:.2%}"
        )
        print(
            f"   Predictions - Train: [{min(train_preds_denorm):.4f}, {max(train_preds_denorm):.4f}] μ={np.mean(train_preds_denorm):.4f} | "
            f"Val: [{min(val_preds_denorm):.4f}, {max(val_preds_denorm):.4f}] μ={np.mean(val_preds_denorm):.4f}" if val_preds_denorm else ""
        )
        print(
            f"   Std - Train pred: {np.std(train_preds_denorm):.6f}, Val pred: {np.std(val_preds_denorm):.6f}" if val_preds_denorm else ""
        )
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
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

    # Plot training curves
    print("\n📊 Génération des courbes de training...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, args.epochs + 1)
    
    # Loss curves
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs_range, [acc * 100 for acc in history['train_acc']], 
             label='Train Accuracy', marker='o', linewidth=2)
    ax2.plot(epochs_range, [acc * 100 for acc in history['val_acc']], 
             label='Val Accuracy', marker='s', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = checkpoint_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Courbes sauvegardées: {plot_path}")
    plt.close()

    print("✓ Training finished")


if __name__ == "__main__":
    train()
