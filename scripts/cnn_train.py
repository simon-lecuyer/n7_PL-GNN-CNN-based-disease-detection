#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--variant", type=str, default="base", help="Model variant (base, etc.)")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--sequence_length", type=int, default=5, help="Number of frames in temporal sequence (L)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
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

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_mse = float('inf')
    patience_counter = 0

    # Loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_mse = 0.0
        train_mae = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)        # (B, L, 64, 64)
            y = y.to(device)        # (B, 1, 64, 64)  

            pred = model(x)         # (B, 1, 64, 64)
            loss = criterion_mse(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse += loss.item()
            train_mae += criterion_mae(pred, y).item()

        train_mse /= len(train_loader)
        train_mae /= len(train_loader)
        history['train_mse'].append(train_mse)
        history['train_mae'].append(train_mae)

        #  Validation
        model.eval()
        val_mse = 0.0
        val_mae = 0.0

        if len(val_loader) == 0:
            val_mse = float("nan")
            val_mae = float("nan")
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            print("Warning: empty validation set (check require_consecutive or sequence_length).")
        else:
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)  # (B, L, 64, 64)
                    y = y.to(device)  # (B, 1, 64, 64)
                    pred = model(x)   # (B, 1, 64, 64)
                    val_mse += criterion_mse(pred, y).item()
                    val_mae += criterion_mae(pred, y).item()

            val_mse /= len(val_loader)
            val_mae /= len(val_loader)
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_mse:.6f} MAE: {train_mae:.6f} | "
            f"Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
        )
        
        # Learning rate scheduler
        scheduler.step(val_mse)
        
        # Save best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_mse,
                'val_mae': val_mae,
            }, checkpoint_dir / "best_model.pt")
            print(f"  ✓ Best model saved (val_mse: {val_mse:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
            }, checkpoint_path)
    
    # Save final history
    with open(checkpoint_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves - Enhanced version
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    epochs = np.arange(1, args.epochs + 1)
    
    # Helper function for smoothing
    def smooth_curve(values, window=3):
        if len(values) < window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
        return np.array(smoothed)
    
    # 1. MSE - Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_mse'], 'b-', label='Train', linewidth=2, alpha=0.7)
    ax1.plot(epochs, history['val_mse'], 'r-', label='Validation', linewidth=2, alpha=0.7)
    ax1.fill_between(epochs, history['train_mse'], alpha=0.2, color='blue')
    ax1.fill_between(epochs, history['val_mse'], alpha=0.2, color='red')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('MSE', fontsize=10)
    ax1.set_title('MSE - Original', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE - Smoothed
    ax2 = fig.add_subplot(gs[0, 1])
    train_mse_smooth = smooth_curve(history['train_mse'], window=3)
    val_mse_smooth = smooth_curve(history['val_mse'], window=3)
    ax2.plot(epochs, train_mse_smooth, 'b-', label='Train (smoothed)', linewidth=2.5)
    ax2.plot(epochs, val_mse_smooth, 'r-', label='Val (smoothed)', linewidth=2.5)
    ax2.fill_between(epochs, train_mse_smooth, alpha=0.2, color='blue')
    ax2.fill_between(epochs, val_mse_smooth, alpha=0.2, color='red')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('MSE', fontsize=10)
    ax2.set_title('MSE - Smoothed (Moving Avg)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. MAE - Original
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, history['train_mae'], 'g-', label='Train', linewidth=2, alpha=0.7)
    ax3.plot(epochs, history['val_mae'], 'orange', label='Validation', linewidth=2, alpha=0.7)
    ax3.fill_between(epochs, history['train_mae'], alpha=0.2, color='green')
    ax3.fill_between(epochs, history['val_mae'], alpha=0.2, color='orange')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('MAE', fontsize=10)
    ax3.set_title('MAE - Original', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. MAE - Smoothed
    ax4 = fig.add_subplot(gs[1, 1])
    train_mae_smooth = smooth_curve(history['train_mae'], window=3)
    val_mae_smooth = smooth_curve(history['val_mae'], window=3)
    ax4.plot(epochs, train_mae_smooth, 'g-', label='Train (smoothed)', linewidth=2.5)
    ax4.plot(epochs, val_mae_smooth, 'orange', label='Val (smoothed)', linewidth=2.5)
    ax4.fill_between(epochs, train_mae_smooth, alpha=0.2, color='green')
    ax4.fill_between(epochs, val_mae_smooth, alpha=0.2, color='orange')
    ax4.set_xlabel('Epoch', fontsize=10)
    ax4.set_ylabel('MAE', fontsize=10)
    ax4.set_title('MAE - Smoothed (Moving Avg)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Combined view - both metrics
    ax5 = fig.add_subplot(gs[2, :])
    ax5_twin = ax5.twinx()
    
    mse_line1 = ax5.plot(epochs, history['train_mse'], 'b-', label='Train MSE', linewidth=2.5, marker='o', markersize=3, alpha=0.8)
    mse_line2 = ax5.plot(epochs, history['val_mse'], 'r-', label='Val MSE', linewidth=2.5, marker='s', markersize=3, alpha=0.8)
    
    mae_line1 = ax5_twin.plot(epochs, history['train_mae'], 'g--', label='Train MAE', linewidth=2.5, marker='^', markersize=3, alpha=0.8)
    mae_line2 = ax5_twin.plot(epochs, history['val_mae'], color='orange', linestyle='--', label='Val MAE', linewidth=2.5, marker='v', markersize=3, alpha=0.8)
    
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('MSE', fontsize=11, fontweight='bold', color='blue')
    ax5_twin.set_ylabel('MAE', fontsize=11, fontweight='bold', color='green')
    ax5.tick_params(axis='y', labelcolor='blue')
    ax5_twin.tick_params(axis='y', labelcolor='green')
    ax5.set_title('Training vs Validation - All Metrics', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends
    lines = mse_line1 + mse_line2 + mae_line1 + mae_line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper right', fontsize=10)
    
    # Main title
    fig.suptitle(f'CNN Training Summary - {args.variant.upper()} Model\nFinal Validation MSE: {history["val_mse"][-1]:.6f}', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    plt.savefig(checkpoint_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {checkpoint_dir / 'training_curves.png'}")
    
    print("✓ Training finished")


if __name__ == "__main__":
    train()