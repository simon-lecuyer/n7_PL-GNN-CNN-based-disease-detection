#!/usr/bin/env python3
"""
Script pour faire une prédiction par simulation à partir des N premières frames.
Pour chaque simulation dans le test set, prédit la frame N+1 à partir des frames 0 à N-1.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from cnn_model import DiseaseCNN


def parse_args():
    parser = argparse.ArgumentParser("Predict one sample per simulation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="cnn/predictions_per_sim")
    return parser.parse_args()


def load_model(checkpoint_path, device, sequence_length):
    """Charge le modèle depuis un checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = DiseaseCNN(
        in_frames=sequence_length,
        out_channels=1,
        hidden_channels=16
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Modèle chargé depuis {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Val MSE: {checkpoint['val_mse']:.6f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Modèle chargé depuis {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    return model


def load_test_data(test_json_path, sequence_length):
    """
    Charge le test.json et organise par simulation.
    Pour chaque simulation, prend les frames 0 à sequence_length-1 comme input,
    et frame sequence_length comme target.
    """
    with open(test_json_path, 'r') as f:
        data = json.load(f)
    
    # Grouper par sim_id
    by_sim = defaultdict(list)
    for sample in data['samples']:
        by_sim[sample['sim_id']].append(sample)
    
    # Trier par timestep et construire les séquences
    results = []
    for sim_id in sorted(by_sim.keys()):
        samples = sorted(by_sim[sim_id], key=lambda x: x['timestep'])
        
        # Vérifier qu'on a assez de frames
        if len(samples) < sequence_length + 1:
            print(f"⚠️  Simulation {sim_id}: pas assez de frames ({len(samples)} < {sequence_length + 1})")
            continue
        
        # Prendre les N premières frames comme input
        input_samples = samples[:sequence_length]
        target_sample = samples[sequence_length]
        
        # Charger les frames
        input_frames = []
        for sample in input_samples:
            npy_path = sample['file']
            npy_data = np.load(npy_path, allow_pickle=True).item()
            frame = npy_data['data'].astype(np.float32)  # (64, 64)
            input_frames.append(frame)
        
        # Target
        target_npy = np.load(target_sample['file'], allow_pickle=True).item()
        target = target_npy['data'].astype(np.float32)  # (64, 64)
        
        results.append({
            'sim_id': sim_id,
            'input_frames': np.stack(input_frames, axis=0),  # (L, 64, 64)
            'target': target,  # (64, 64)
            'input_timesteps': [s['timestep'] for s in input_samples],
            'target_timestep': target_sample['timestep']
        })
    
    return results


def predict_simulations(model, data, device):
    """Fait les prédictions pour toutes les simulations"""
    predictions = []
    
    with torch.no_grad():
        for item in data:
            # Préparer l'input
            input_frames = torch.from_numpy(item['input_frames']).unsqueeze(0)  # (1, L, 64, 64)
            input_frames = input_frames.to(device)
            
            # Prédiction
            pred = model(input_frames)  # (1, 1, 64, 64)
            pred = pred.squeeze().cpu().numpy()  # (64, 64)
            
            # Calculer les métriques
            target = item['target']
            mse = np.mean((target - pred) ** 2)
            mae = np.mean(np.abs(target - pred))
            
            predictions.append({
                'sim_id': item['sim_id'],
                'input_frames': item['input_frames'],
                'prediction': pred,
                'target': target,
                'mse': mse,
                'mae': mae,
                'input_timesteps': item['input_timesteps'],
                'target_timestep': item['target_timestep']
            })
    
    return predictions


def visualize_predictions(predictions, sequence_length, output_path):
    """
    Visualise les prédictions : une ligne par simulation.
    Colonnes : input frames (0 à L-1), target (L), prediction (L), error
    """
    n_sims = len(predictions)
    n_cols = sequence_length + 3  # input + target + pred + error
    
    fig, axes = plt.subplots(n_sims, n_cols, figsize=(2.5 * n_cols, 3 * n_sims))
    
    if n_sims == 1:
        axes = axes.reshape(1, -1)
    
    for i, pred_data in enumerate(predictions):
        sim_id = pred_data['sim_id']
        input_frames = pred_data['input_frames']
        target = pred_data['target']
        prediction = pred_data['prediction']
        error = np.abs(target - prediction)
        
        # Afficher toutes les frames d'input
        for t in range(sequence_length):
            ax = axes[i, t]
            im = ax.imshow(input_frames[t], cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_title(f't={t}', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Target
        ax_target = axes[i, sequence_length]
        im_target = ax_target.imshow(target, cmap='YlOrRd', vmin=0, vmax=1)
        ax_target.set_title(f'Target\nt={sequence_length}', fontsize=9, fontweight='bold')
        ax_target.axis('off')
        plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
        
        # Prédiction
        ax_pred = axes[i, sequence_length + 1]
        im_pred = ax_pred.imshow(prediction, cmap='YlOrRd', vmin=0, vmax=1)
        ax_pred.set_title(f'Prediction\nt={sequence_length}', fontsize=9, fontweight='bold', color='blue')
        ax_pred.axis('off')
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        
        # Erreur
        ax_error = axes[i, sequence_length + 2]
        im_error = ax_error.imshow(error, cmap='Reds', vmin=0, vmax=0.5)
        ax_error.set_title(f'Error\nMAE={pred_data["mae"]:.4f}', fontsize=9, fontweight='bold', color='red')
        ax_error.axis('off')
        plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
        
        # Label de la ligne
        axes[i, 0].text(-0.3, 0.5, f'Sim {sim_id}', 
                       transform=axes[i, 0].transAxes,
                       fontsize=12, fontweight='bold',
                       verticalalignment='center',
                       rotation=90)
    
    plt.suptitle(f'CNN Predictions per Simulation (first {sequence_length} frames → frame {sequence_length})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualisation sauvegardée: {output_path}")
    
    plt.show()


def visualize_single_simulation(prediction_data, sequence_length, output_path):
    """
    Visualise les prédictions pour une seule simulation de manière détaillée.
    Affiche : les frames d'input en carrousel, puis target, prédiction et erreur côte à côte.
    """
    sim_id = prediction_data['sim_id']
    input_frames = prediction_data['input_frames']
    target = prediction_data['target']
    prediction = prediction_data['prediction']
    error = np.abs(target - prediction)
    mse = prediction_data['mse']
    mae = prediction_data['mae']
    
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, sequence_length + 3, hspace=0.3, wspace=0.3)
    
    # Première ligne : les frames d'input
    for t in range(sequence_length):
        ax = fig.add_subplot(gs[0, t])
        im = ax.imshow(input_frames[t], cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'Input Frame t={t}', fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Deuxième ligne : target, prédiction, erreur
    # Target
    ax_target = fig.add_subplot(gs[1, :sequence_length//2])
    im_target = ax_target.imshow(target, cmap='YlOrRd', vmin=0, vmax=1)
    ax_target.set_title(f'Target Frame (t={sequence_length})', fontsize=12, fontweight='bold')
    ax_target.axis('off')
    cbar_target = plt.colorbar(im_target, ax=ax_target, fraction=0.046, pad=0.04)
    
    # Prédiction
    ax_pred = fig.add_subplot(gs[1, sequence_length//2:sequence_length])
    im_pred = ax_pred.imshow(prediction, cmap='YlOrRd', vmin=0, vmax=1)
    ax_pred.set_title(f'Predicted Frame (t={sequence_length})', fontsize=12, fontweight='bold', color='blue')
    ax_pred.axis('off')
    cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    
    # Erreur
    ax_error = fig.add_subplot(gs[1, sequence_length:])
    im_error = ax_error.imshow(error, cmap='Reds', vmin=0, vmax=0.5)
    ax_error.set_title(f'Absolute Error\nMAE={mae:.4f} | MSE={mse:.4f}', 
                      fontsize=12, fontweight='bold', color='red')
    ax_error.axis('off')
    cbar_error = plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)
    
    fig.suptitle(f'CNN Prediction Details - Simulation {sim_id}\n(Using first {sequence_length} frames to predict frame {sequence_length})', 
                fontsize=14, fontweight='bold')
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualisation détaillée pour Sim {sim_id}: {output_path}")
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}\n")
    
    # Charger le modèle
    model = load_model(args.checkpoint, device, args.sequence_length)
    
    # Charger les données
    print(f"\n📂 Chargement du test set: {args.test_json}")
    data = load_test_data(args.test_json, args.sequence_length)
    print(f"✓ {len(data)} simulations trouvées")
    
    # Afficher les infos
    for item in data:
        print(f"  - Sim {item['sim_id']}: frames {item['input_timesteps'][0]}-{item['input_timesteps'][-1]} → {item['target_timestep']}")
    
    # Prédictions
    print(f"\n🔮 Génération des prédictions...")
    predictions = predict_simulations(model, data, device)
    
    # Afficher les résultats
    print(f"\n{'='*70}")
    print("RÉSULTATS PAR SIMULATION")
    print(f"{'='*70}")
    for pred in predictions:
        print(f"\nSimulation {pred['sim_id']}:")
        print(f"  MSE:  {pred['mse']:.6f}")
        print(f"  MAE:  {pred['mae']:.6f}")
        print(f"  RMSE: {np.sqrt(pred['mse']):.6f}")
    
    # MSE/MAE moyens
    avg_mse = np.mean([p['mse'] for p in predictions])
    avg_mae = np.mean([p['mae'] for p in predictions])
    print(f"\n{'='*70}")
    print(f"MOYENNE SUR {len(predictions)} SIMULATIONS:")
    print(f"  MSE moyen:  {avg_mse:.6f}")
    print(f"  MAE moyen:  {avg_mae:.6f}")
    print(f"  RMSE moyen: {np.sqrt(avg_mse):.6f}")
    print(f"{'='*70}\n")
    
    # Visualisation générale (toutes les simulations)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "predictions_per_simulation.png"
    visualize_predictions(predictions, args.sequence_length, output_path)
    
    # Visualisation détaillée pour la première simulation
    if len(predictions) > 0:
        first_pred = predictions[0]
        single_sim_path = output_dir / f"prediction_detail_sim_{first_pred['sim_id']}.png"
        visualize_single_simulation(first_pred, args.sequence_length, single_sim_path)
    
    # Sauvegarder les résultats
    results_file = output_dir / "results_per_simulation.json"
    results = {
        'predictions': [
            {
                'sim_id': int(p['sim_id']),
                'mse': float(p['mse']),
                'mae': float(p['mae']),
                'rmse': float(np.sqrt(p['mse']))
            }
            for p in predictions
        ],
        'average': {
            'mse': float(avg_mse),
            'mae': float(avg_mae),
            'rmse': float(np.sqrt(avg_mse))
        }
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Résultats JSON sauvegardés: {results_file}\n")


if __name__ == "__main__":
    main()
