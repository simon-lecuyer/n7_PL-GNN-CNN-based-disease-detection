"""
Script pour faire des prédictions avec le modèle CNN entraîné
et visualiser les matrices d'infection prédites vs réelles.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cnn_dataset import TemporalCNNDataset
from cnn_model import DiseaseCNN


def load_model(checkpoint_path, device, sequence_length=5):
    """Charge le modèle depuis un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = DiseaseCNN(
        in_frames=sequence_length,
        out_channels=1,
        hidden_channels=8
    )
    
    # Gérer les deux formats de checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Format complet (checkpoint_epochXXX.pt)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modèle chargé depuis {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val MSE: {checkpoint['val_mse']:.6f}, Val MAE: {checkpoint.get('val_mae', 'N/A')}")
    else:
        # Format state_dict simple (best_model.pt)
        model.load_state_dict(checkpoint)
        print(f"✓ Modèle chargé depuis {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model


def predict_batch(model, dataloader, device, num_samples=5):
    """
    Fait des prédictions sur un batch et retourne les résultats
    
    Returns:
        List of dicts avec 'input', 'target', 'prediction' (matrices 64x64)
    """
    results = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            predictions = model(inputs)
            
            # Convertir en numpy et sauvegarder
            for i in range(inputs.size(0)):
                if len(results) >= num_samples:
                    return results
                
                # Last frame de l'input (t+4)
                last_input_frame = inputs[i, -1].cpu().numpy()  # (64, 64)
                target = targets[i, 0].cpu().numpy()  # (64, 64)
                pred = predictions[i, 0].cpu().numpy()  # (64, 64)
                
                results.append({
                    'last_input_frame': last_input_frame,
                    'target': target,
                    'prediction': pred,
                    'mse': np.mean((target - pred) ** 2),
                    'mae': np.mean(np.abs(target - pred))
                })
    
    return results


def visualize_predictions(results, save_path=None):
    """
    Visualise les prédictions: dernière frame d'input, target, prédiction, différence
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        last_frame = result['last_input_frame']
        target = result['target']
        pred = result['prediction']
        diff = np.abs(target - pred)
        
        # Dernière frame d'input (t+4)
        im0 = axes[i, 0].imshow(last_frame, cmap='YlOrRd', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Sample {i+1}: Input (t+4)')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Target réel (t+5)
        im1 = axes[i, 1].imshow(target, cmap='YlOrRd', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Target (t+5) - Réel')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Prédiction (t+5)
        im2 = axes[i, 2].imshow(pred, cmap='YlOrRd', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prédiction (t+5)\nMSE={result["mse"]:.4f}, MAE={result["mae"]:.4f}')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Différence absolue
        im3 = axes[i, 3].imshow(diff, cmap='Reds', vmin=0, vmax=0.3)
        axes[i, 3].set_title(f'Erreur absolue')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualisation sauvegardée: {save_path}")
    
    plt.show()
    return fig


def save_predictions(results, save_dir):
    """Sauvegarde les matrices prédites en fichiers .npy"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        # Sauvegarder chaque matrice
        np.save(save_dir / f"sample_{i}_target.npy", result['target'])
        np.save(save_dir / f"sample_{i}_prediction.npy", result['prediction'])
        np.save(save_dir / f"sample_{i}_last_input.npy", result['last_input_frame'])
        
        # Sauvegarder les métriques
        with open(save_dir / f"sample_{i}_metrics.json", 'w') as f:
            json.dump({
                'mse': float(result['mse']),
                'mae': float(result['mae'])
            }, f, indent=2)
    
    print(f"✓ {len(results)} matrices sauvegardées dans {save_dir}")


def print_matrix_stats(result, sample_idx):
    """Affiche les statistiques d'une matrice prédite"""
    target = result['target']
    pred = result['prediction']
    
    print(f"\n{'='*60}")
    print(f"SAMPLE {sample_idx + 1}")
    print(f"{'='*60}")
    print(f"\nTarget (Réel):")
    print(f"  Min: {target.min():.4f}, Max: {target.max():.4f}, Mean: {target.mean():.4f}")
    print(f"\nPrédiction:")
    print(f"  Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")
    print(f"\nMétriques:")
    print(f"  MSE:  {result['mse']:.6f}")
    print(f"  MAE:  {result['mae']:.6f}")
    print(f"  RMSE: {np.sqrt(result['mse']):.6f}")
    
    # Quelques valeurs de la matrice
    print(f"\nAperçu de la matrice de prédiction (5 premières lignes, 5 premières colonnes):")
    print(pred[:5, :5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Chemin vers le checkpoint du modèle')
    parser.add_argument('--processed_dir', type=str, required=True,
                        help='Répertoire des données prétraitées')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Longueur de la séquence temporelle')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Nombre de samples à prédire et visualiser')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Split à utiliser pour les prédictions')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Répertoire pour sauvegarder les prédictions')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Charger le modèle
    model = load_model(args.checkpoint, device, sequence_length=args.sequence_length)
    
    # Charger le dataset
    processed_dir = Path(args.processed_dir)
    dataset_dir = processed_dir / "datasets" / "cnn"
    json_file = dataset_dir / f"{args.split}.json"
    
    print(f"\n✓ Chargement du dataset: {json_file}")
    dataset = TemporalCNNDataset(
        json_file=json_file,
        sequence_length=args.sequence_length,
        require_consecutive=True
    )
    print(f"  {len(dataset)} séquences chargées")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Faire les prédictions
    print(f"\n🔮 Génération de {args.num_samples} prédictions...")
    results = predict_batch(model, dataloader, device, num_samples=args.num_samples)
    
    # Afficher les statistiques
    for i, result in enumerate(results):
        print_matrix_stats(result, i)
    
    # Sauvegarder les matrices
    if args.save_dir:
        save_predictions(results, args.save_dir)
    else:
        # Sauvegarder dans le répertoire du checkpoint
        checkpoint_dir = Path(args.checkpoint).parent
        save_dir = checkpoint_dir / "predictions"
        save_predictions(results, save_dir)
    
    # Visualiser
    print(f"\n📊 Génération de la visualisation...")
    if args.save_dir:
        viz_path = Path(args.save_dir) / "predictions_visualization.png"
    else:
        viz_path = Path(args.checkpoint).parent / "predictions_visualization.png"
    
    visualize_predictions(results, save_path=viz_path)


if __name__ == "__main__":
    main()
