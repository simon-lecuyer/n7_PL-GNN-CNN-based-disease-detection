#!/usr/bin/env python3
"""
Script AUTONOME pour entraîner le modèle avec différentes loss functions
et comparer les résultats (training curves, validation curves, predictions)

Ce script est complètement séparé de cnn_train.py et cnn_predict.py
qui restent intacts pour leur usage normal.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cnn_model import DiseaseCNN
from cnn_dataset import TemporalCNNDataset


LOSS_FUNCTIONS = ["mse", "mae", "smooth_l1", "bce"]

LOSS_DESCRIPTIONS = {
    "mse": "MSE (Mean Squared Error)",
    "mae": "MAE (Mean Absolute Error)",
    "smooth_l1": "Smooth L1 Loss (β=0.1)",
    "bce": "Binary Cross-Entropy"
}


# ========== LOSS FUNCTIONS DEFINITIONS ==========
def get_loss_function(loss_name):
    """
    Retourne une fonction de loss et une description
    
    Args:
        loss_name: nom de la loss (mse, mae, smooth_l1, bce)
    
    Returns:
        loss_fn: fonction de loss
        description: description textuelle
    """
    if loss_name == "mse":
        return nn.MSELoss(), "Mean Squared Error (MSE)"
    
    elif loss_name == "mae":
        return nn.L1Loss(), "Mean Absolute Error (MAE)"
    
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=0.1), "Smooth L1 Loss (β=0.1)"
    
    elif loss_name == "bce":
        # BCE avec clipping pour éviter log(0)
        class BCE_Loss(nn.Module):
            def forward(self, pred, target):
                pred_clipped = pred.clamp(1e-7, 1 - 1e-7)
                return -(target * torch.log(pred_clipped) + (1 - target) * torch.log(1 - pred_clipped)).mean()
        return BCE_Loss(), "Binary Cross-Entropy (BCE)"
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def parse_args():
    parser = argparse.ArgumentParser("Train and compare different loss functions")
    parser.add_argument("--processed_dir", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="results/loss_comparison")
    parser.add_argument("--skip_training", action="store_true", 
                       help="Skip training and use existing checkpoints")
    parser.add_argument("--losses", type=str, default="all",
                       help="Comma-separated list of losses (or 'all')")
    return parser.parse_args()


# ========== TRAINING FUNCTION (AUTONOME) ==========
def train_with_loss(processed_dir, loss_function, batch_size, epochs, lr, weight_decay, 
                    sequence_length, patience, device):
    """
    Entraîne le modèle avec une loss function spécifique
    Cette fonction est autonome et n'appelle pas cnn_train.py
    """
    
    print(f"\n{'='*80}")
    print(f"Training with {LOSS_DESCRIPTIONS[loss_function]}")
    print(f"{'='*80}")
    
    # Load datasets
    processed_dir = Path(processed_dir)
    datasets_dir = processed_dir / "datasets" / "cnn"
    train_json = datasets_dir / "train.json"
    val_json = datasets_dir / "val.json"
    
    if not train_json.exists() or not val_json.exists():
        print(f"⚠️  Missing train or val json files")
        return None
    
    # Create datasets
    train_ds = TemporalCNNDataset(
        str(train_json),
        sequence_length=sequence_length,
        require_consecutive=True
    )
    val_ds = TemporalCNNDataset(
        str(val_json),
        sequence_length=sequence_length,
        require_consecutive=True
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Create model
    model = DiseaseCNN(
        in_frames=sequence_length,
        out_channels=1,
        hidden_channels=16
    ).to(device)
    
    # Get loss function
    criterion, loss_description = get_loss_function(loss_function)
    criterion_mse = nn.MSELoss()  # Pour tracking
    criterion_mae = nn.L1Loss()   # Pour tracking
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    # Create checkpoint directory
    checkpoint_dir = Path("cnn/checkpoints") / f"cnn_{loss_function}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'model': 'DiseaseCNN_Temporal_3D',
        'variant': loss_function,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'sequence_length': sequence_length,
        'device': str(device),
        'loss_function': loss_function,
        'loss_description': loss_description,
    }
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Device:        {device}")
    print(f"Loss function: {loss_description}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Parameters:    {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint:    {checkpoint_dir}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += criterion_mse(pred, y).item()
            train_mae += criterion_mae(pred, y).item()
        
        train_loss /= len(train_loader)
        train_mse /= len(train_loader)
        train_mae /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['train_mse'].append(train_mse)
        history['train_mae'].append(train_mae)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_mae = 0.0
        
        if len(val_loader) > 0:
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    val_loss += criterion(pred, y).item()
                    val_mse += criterion_mse(pred, y).item()
                    val_mae += criterion_mae(pred, y).item()
            
            val_loss /= len(val_loader)
            val_mse /= len(val_loader)
            val_mae /= len(val_loader)
        else:
            val_loss = float("nan")
            val_mse = float("nan")
            val_mae = float("nan")
        
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        
        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.6f} | "
            f"MSE: {train_mse:.6f} MAE: {train_mae:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} MAE: {val_mae:.6f}"
        )
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae,
            }, checkpoint_dir / "best_model.pt")
            print(f"  ✓ Best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
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
    
    print(f"✓ Training finished for {loss_function}")
    return str(checkpoint_dir)


def find_latest_checkpoint(loss_function):
    """Trouve le dernier checkpoint pour une loss function"""
    checkpoint_dir = Path(f"cnn/checkpoints")
    
    matching_dirs = sorted([
        d for d in checkpoint_dir.glob(f"cnn_{loss_function}_*")
        if d.is_dir()
    ], reverse=True)
    
    if not matching_dirs:
        print(f"⚠️  No checkpoint found for {loss_function}")
        return None
    
    checkpoint_path = matching_dirs[0] / "best_model.pt"
    if checkpoint_path.exists():
        return str(checkpoint_path)
    
    return None


def load_history(checkpoint_dir_path):
    """Charge l'historique d'entraînement depuis history.json"""
    history_path = Path(checkpoint_dir_path) / "history.json"
    
    if not history_path.exists():
        print(f"⚠️  History file not found: {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        return json.load(f)


def load_config(checkpoint_dir_path):
    """Charge la config du modèle"""
    config_path = Path(checkpoint_dir_path) / "config.json"
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def predict_with_model(checkpoint_path, test_json, sequence_length, device):
    """Fait les prédictions avec un modèle"""
    
    # Charger le modèle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiseaseCNN(
        in_frames=sequence_length,
        out_channels=1,
        hidden_channels=16
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Charger le test set
    test_ds = TemporalCNNDataset(test_json, sequence_length=sequence_length, require_consecutive=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Prédictions
    mses = []
    maes = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            
            mse = torch.mean((pred - y) ** 2).item()
            mae = torch.mean(torch.abs(pred - y)).item()
            
            mses.append(mse)
            maes.append(mae)
    
    return {
        'mse': np.mean(mses),
        'mae': np.mean(maes),
        'rmse': np.sqrt(np.mean(mses))
    }


def plot_training_curves(histories, loss_functions, output_dir):
    """Trace les courbes d'entraînement: un graphe par loss function (train/val superposés)"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Créer une figure avec 6 subplots (ou moins selon le nombre de losses)
    n_losses = len([l for l in loss_functions if histories.get(l)])
    n_cols = 3
    n_rows = (n_losses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # S'assurer que axes est un array même avec un seul subplot
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    colors_train = '#2E86AB'  # Bleu
    colors_val = '#A23B72'    # Pourpre
    
    idx = 0
    for loss_name in loss_functions:
        history = histories.get(loss_name)
        
        if not history:
            continue
        
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Récupérer les epochs et données
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        
        epochs = np.arange(1, len(train_loss) + 1)
        
        # Tracer train et val
        ax.plot(epochs, train_loss, label='Training', linewidth=2.5, color=colors_train, alpha=0.8, marker='o', markersize=4)
        ax.plot(epochs, val_loss, label='Validation', linewidth=2.5, color=colors_val, alpha=0.8, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=10, fontweight='bold')
        ax.set_title(LOSS_DESCRIPTIONS[loss_name], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs finales
        final_train = train_loss[-1] if train_loss else 0
        final_val = val_loss[-1] if val_loss else 0
        ax.text(0.98, 0.02, f'Final: Train={final_train:.4f} | Val={final_val:.4f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        idx += 1
    
    # Supprimer les subplots inutilisés
    for idx in range(idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.suptitle('Training Curves - Train vs Validation for Each Loss Function', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path = output_dir / 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def plot_test_results(test_results, output_dir):
    """Trace les résultats de prédiction sur le test set"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    loss_names = list(test_results.keys())
    mses = [test_results[ln]['mse'] for ln in loss_names]
    maes = [test_results[ln]['mae'] for ln in loss_names]
    rmses = [test_results[ln]['rmse'] for ln in loss_names]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFB347']
    labels = [LOSS_DESCRIPTIONS[ln] for ln in loss_names]
    
    # MSE
    bars = axes[0].bar(range(len(loss_names)), mses, color=colors[:len(loss_names)], alpha=0.7)
    axes[0].set_ylabel('MSE', fontsize=11, fontweight='bold')
    axes[0].set_title('Test Set MSE', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(len(loss_names)))
    axes[0].set_xticklabels([i+1 for i in range(len(loss_names))], fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, mse in zip(bars, mses):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE
    bars = axes[1].bar(range(len(loss_names)), maes, color=colors[:len(loss_names)], alpha=0.7)
    axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
    axes[1].set_title('Test Set MAE', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(loss_names)))
    axes[1].set_xticklabels([i+1 for i in range(len(loss_names))], fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE
    bars = axes[2].bar(range(len(loss_names)), rmses, color=colors[:len(loss_names)], alpha=0.7)
    axes[2].set_ylabel('RMSE', fontsize=11, fontweight='bold')
    axes[2].set_title('Test Set RMSE', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(loss_names)))
    axes[2].set_xticklabels([i+1 for i in range(len(loss_names))], fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, rmse in zip(bars, rmses):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Legend
    legend_text = '\n'.join([f"{i+1}. {labels[i]}" for i in range(len(loss_names))])
    fig.text(0.02, 0.95, legend_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Prediction Performance Comparison on Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.15, 0, 1, 0.96])
    
    output_path = output_dir / 'test_results_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary(histories, test_results, loss_functions):
    """Affiche un résumé des résultats"""
    
    print("\n" + "="*100)
    print("SUMMARY - TRAINING & PREDICTION COMPARISON")
    print("="*100)
    
    print(f"\n{'Loss Function':<40} {'Final Val Loss':<18} {'Test MSE':<15} {'Test MAE':<15} {'Test RMSE':<15}")
    print("-"*100)
    
    results_list = []
    for loss_name in loss_functions:
        history = histories.get(loss_name)
        test_result = test_results.get(loss_name)
        
        if history and test_result:
            final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('nan')
            test_mse = test_result['mse']
            test_mae = test_result['mae']
            test_rmse = test_result['rmse']
            
            results_list.append((loss_name, final_val_loss, test_mse, test_mae, test_rmse))
    
    # Trier par test MSE
    results_list.sort(key=lambda x: x[2])
    
    for i, (loss_name, final_val_loss, test_mse, test_mae, test_rmse) in enumerate(results_list, 1):
        print(f"{i}. {LOSS_DESCRIPTIONS[loss_name]:<38} "
              f"{final_val_loss:<18.6f} {test_mse:<15.6f} {test_mae:<15.6f} {test_rmse:<15.6f}")
    
    print("\n" + "="*100)
    if results_list:
        best_loss = results_list[0][0]
        print(f"🏆 BEST LOSS FUNCTION: {LOSS_DESCRIPTIONS[best_loss]}")
        print(f"   Test MSE: {results_list[0][2]:.6f}")
        print(f"   Test MAE: {results_list[0][3]:.6f}")
        print("="*100)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Déterminer les losses à traiter
    if args.losses == "all":
        loss_functions = LOSS_FUNCTIONS
    else:
        loss_functions = args.losses.split(",")
    
    print(f"\n🔬 LOSS FUNCTIONS COMPARISON (AUTONOMOUS)")
    print(f"Device: {device}")
    print(f"Losses to compare: {', '.join([LOSS_DESCRIPTIONS[l] for l in loss_functions])}")
    print(f"Note: This script is independent of cnn_train.py and cnn_predict.py")
    
    # ========== PHASE 1: TRAINING ==========
    if not args.skip_training:
        print(f"\n{'='*80}")
        print("PHASE 1: TRAINING WITH DIFFERENT LOSS FUNCTIONS")
        print(f"{'='*80}")
        
        for loss_function in loss_functions:
            train_with_loss(
                args.processed_dir,
                loss_function,
                args.batch_size,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.sequence_length,
                args.patience,
                device
            )
    
    # ========== PHASE 2: COLLECT HISTORIES ==========
    print(f"\n{'='*80}")
    print("PHASE 2: COLLECTING TRAINING HISTORIES")
    print(f"{'='*80}")
    
    histories = {}
    for loss_function in loss_functions:
        checkpoint_path = find_latest_checkpoint(loss_function)
        if checkpoint_path:
            checkpoint_dir = str(Path(checkpoint_path).parent)
            history = load_history(checkpoint_dir)
            histories[loss_function] = history
            print(f"✓ Loaded history for {LOSS_DESCRIPTIONS[loss_function]}")
        else:
            print(f"⚠️  No history found for {loss_function}")
            histories[loss_function] = None
    
    # ========== PHASE 3: PREDICTIONS ON TEST SET ==========
    print(f"\n{'='*80}")
    print("PHASE 3: PREDICTIONS ON TEST SET")
    print(f"{'='*80}")
    
    test_results = {}
    for loss_function in loss_functions:
        checkpoint_path = find_latest_checkpoint(loss_function)
        if checkpoint_path:
            print(f"Predicting with {LOSS_DESCRIPTIONS[loss_function]}...")
            results = predict_with_model(checkpoint_path, args.test_json, args.sequence_length, device)
            test_results[loss_function] = results
            print(f"  MSE: {results['mse']:.6f}, MAE: {results['mae']:.6f}, RMSE: {results['rmse']:.6f}")
        else:
            print(f"⚠️  Cannot predict with {loss_function}")
            test_results[loss_function] = None
    
    # ========== PHASE 4: VISUALIZATIONS ==========
    print(f"\n{'='*80}")
    print("PHASE 4: GENERATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_training_curves(histories, loss_functions, args.output_dir)
    # plot_test_results(test_results, args.output_dir)  # Removed: only keeping training curves
    
    # ========== SUMMARY ==========
    print_summary(histories, test_results, loss_functions)
    
    print(f"\n✅ Comparison complete!")
    print(f"   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
