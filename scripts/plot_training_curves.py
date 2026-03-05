#!/usr/bin/env python3
"""
Utilitaire pour générer des courbes d'entraînement améliorées à partir des fichiers history.json
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def smooth_curve(values, window=3):
    """Lisse une courbe avec une moyenne mobile"""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(np.mean(values[start:end]))
    return np.array(smoothed)


def plot_training_curves(history_file, output_file=None):
    """Génère les courbes de formation"""
    
    # Charger l'historique
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = np.arange(1, len(history['train_mse']) + 1)
    
    # Créer la figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
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
    
    # Main title with stats
    best_val_mse = min(history['val_mse'])
    best_epoch = np.argmin(history['val_mse']) + 1
    final_val_mse = history['val_mse'][-1]
    
    fig.suptitle(f'CNN Training Summary\nBest Val MSE: {best_val_mse:.6f} (epoch {best_epoch}) | Final Val MSE: {final_val_mse:.6f}', 
                 fontsize=13, fontweight='bold', y=0.995)
    
    # Sauvegarder
    if output_file is None:
        output_file = Path(history_file).parent / "training_curves.png"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_file}")
    
    # Afficher les stats
    print(f"\n{'='*70}")
    print("STATISTIQUES D'ENTRAÎNEMENT")
    print(f"{'='*70}")
    print(f"Nombre d'epochs:     {len(history['train_mse'])}")
    print(f"\nMSE:")
    print(f"  Train initial:     {history['train_mse'][0]:.6f}")
    print(f"  Train final:       {history['train_mse'][-1]:.6f}")
    print(f"  Val initial:       {history['val_mse'][0]:.6f}")
    print(f"  Val final:         {history['val_mse'][-1]:.6f}")
    print(f"  Val meilleur:      {best_val_mse:.6f} (epoch {best_epoch})")
    print(f"\nMAE:")
    print(f"  Train initial:     {history['train_mae'][0]:.6f}")
    print(f"  Train final:       {history['train_mae'][-1]:.6f}")
    print(f"  Val initial:       {history['val_mae'][0]:.6f}")
    print(f"  Val final:         {history['val_mae'][-1]:.6f}")
    
    # Analyse overfitting
    train_final = history['train_mse'][-1]
    val_final = history['val_mse'][-1]
    if val_final > train_final * 1.5:
        print(f"\n⚠️  Possible OVERFITTING détecté (Val MSE >> Train MSE)")
    elif train_final > val_final:
        print(f"\n⚠️  Possible UNDERFITTING (Train MSE > Val MSE)")
    else:
        print(f"\n✓ Apprentissage équilibré (Val MSE ≈ Train MSE)")
    print(f"{'='*70}\n")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser("Plot training curves from history.json")
    parser.add_argument("history_file", type=str, help="Path to history.json")
    parser.add_argument("--output", type=str, default=None, help="Output image path (default: same dir as history.json)")
    
    args = parser.parse_args()
    plot_training_curves(args.history_file, args.output)


if __name__ == "__main__":
    main()
