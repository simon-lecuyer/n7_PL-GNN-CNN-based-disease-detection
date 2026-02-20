import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from cnn_model import DiseaseCNN
from cnn_dataset import TemporalCNNDataset
from torch.utils.data import DataLoader
import seaborn as sns


class MetricsEvaluator:
    """
    Comprehensive evaluation metrics for spatiotemporal disease prediction CNN.
    Evaluates both pixel-level accuracy and temporal consistency.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.metrics_history = {
            'pixel_mse': [],
            'pixel_mae': [],
            'pixel_rmse': [],
            'structural_similarity': [],
            'psnr': [],
            'infection_mae': [],  # Scalar infection level error
            'temporal_consistency': [],
            'spatial_gradient_error': [],
            'r2_score': [],
        }
    
    def compute_pixel_metrics(self, predictions, targets):
        """
        Compute pixel-level regression metrics
        
        Args:
            predictions: (B, 1, 64, 64) predicted infection maps
            targets: (B, 1, 64, 64) ground truth infection maps
        
        Returns:
            dict with MSE, MAE, RMSE
        """
        pred_np = predictions.detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        mse = mean_squared_error(target_np, pred_np)
        mae = mean_absolute_error(target_np, pred_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_np, pred_np)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        }
    
    def compute_structural_similarity(self, predictions, targets):
        """
        Compute SSIM (structural similarity) between predicted and actual maps.
        Measures perceptual quality of predicted infection patterns.
        
        Args:
            predictions: (B, 1, 64, 64)
            targets: (B, 1, 64, 64)
        
        Returns:
            dict with mean SSIM and per-sample SSIM
        """
        batch_size = predictions.shape[0]
        ssim_scores = []
        psnr_scores = []
        
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        for i in range(batch_size):
            pred_map = pred_np[i, 0]  # (64, 64)
            target_map = target_np[i, 0]
            
            # Normalize to [0, 1] for SSIM
            pred_norm = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
            target_norm = (target_map - target_map.min()) / (target_map.max() - target_map.min() + 1e-8)
            
            ssim_score = ssim(target_norm, pred_norm, data_range=1.0)
            ssim_scores.append(ssim_score)
            
            psnr_score = psnr(target_norm, pred_norm, data_range=1.0)
            psnr_scores.append(psnr_score)
        
        return {
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'ssim_scores': ssim_scores,
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'psnr_scores': psnr_scores,
        }
    
    def compute_infection_level_error(self, predictions, targets):
        """
        Compute scalar infection level error (integrated intensity).
        Important for overall disease severity prediction.
        
        Args:
            predictions: (B, 1, 64, 64)
            targets: (B, 1, 64, 64)
        
        Returns:
            dict with infection level metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Sum over spatial dimensions to get infection level per sample
        pred_levels = pred_np.reshape(pred_np.shape[0], -1).sum(axis=1)
        target_levels = target_np.reshape(target_np.shape[0], -1).sum(axis=1)
        
        mae = mean_absolute_error(target_levels, pred_levels)
        mse = mean_squared_error(target_levels, pred_levels)
        rmse = np.sqrt(mse)
        
        # Relative error (%)
        relative_error = np.abs(pred_levels - target_levels) / (target_levels + 1e-8) * 100
        mean_relative_error = np.mean(relative_error)
        
        return {
            'infection_level_mae': mae,
            'infection_level_mse': mse,
            'infection_level_rmse': rmse,
            'mean_relative_error_percent': mean_relative_error,
        }
    
    def compute_temporal_consistency(self, pred_sequence, target_sequence):
        """
        Measure temporal consistency - how smooth predictions are over time.
        Computed as gradient smoothness in temporal domain.
        
        Args:
            pred_sequence: (B, L, 64, 64) predicted temporal sequence
            target_sequence: (B, L, 64, 64) target temporal sequence
        
        Returns:
            dict with temporal metrics
        """
        pred_np = pred_sequence.detach().cpu().numpy()
        target_np = target_sequence.detach().cpu().numpy()
        
        batch_size, seq_len = pred_np.shape[0], pred_np.shape[1]
        
        temporal_gradients_pred = []
        temporal_gradients_target = []
        
        for b in range(batch_size):
            for t in range(seq_len - 1):
                # Temporal difference (optical flow-like)
                grad_pred = np.abs(pred_np[b, t+1] - pred_np[b, t]).mean()
                grad_target = np.abs(target_np[b, t+1] - target_np[b, t]).mean()
                
                temporal_gradients_pred.append(grad_pred)
                temporal_gradients_target.append(grad_target)
        
        temporal_consistency_error = mean_absolute_error(
            temporal_gradients_target, 
            temporal_gradients_pred
        )
        
        return {
            'temporal_consistency_error': temporal_consistency_error,
            'mean_temporal_gradient_pred': np.mean(temporal_gradients_pred),
            'mean_temporal_gradient_target': np.mean(temporal_gradients_target),
        }
    
    def compute_spatial_gradient_error(self, predictions, targets):
        """
        Compute spatial gradient error - how well infection boundaries are predicted.
        High accuracy here means good spatial localization.
        
        Args:
            predictions: (B, 1, 64, 64)
            targets: (B, 1, 64, 64)
        
        Returns:
            dict with gradient metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        gradient_errors = []
        
        for i in range(batch_size):
            pred_map = pred_np[i, 0]
            target_map = target_np[i, 0]
            
            # Compute gradients (Sobel-like)
            pred_grad_x = np.abs(np.diff(pred_map, axis=0)).mean()
            pred_grad_y = np.abs(np.diff(pred_map, axis=1)).mean()
            target_grad_x = np.abs(np.diff(target_map, axis=0)).mean()
            target_grad_y = np.abs(np.diff(target_map, axis=1)).mean()
            
            error = np.sqrt((pred_grad_x - target_grad_x)**2 + (pred_grad_y - target_grad_y)**2)
            gradient_errors.append(error)
        
        return {
            'spatial_gradient_error_mean': np.mean(gradient_errors),
            'spatial_gradient_error_std': np.std(gradient_errors),
        }
    
    def compute_infection_thresholded_metrics(self, predictions, targets, threshold=0.5):
        """
        Compute segmentation metrics (IoU, Dice) if we threshold infection maps.
        Useful for assessing detection accuracy.
        
        Args:
            predictions: (B, 1, 64, 64)
            targets: (B, 1, 64, 64)
            threshold: intensity threshold for infection detection
        
        Returns:
            dict with IoU and Dice metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
        target_norm = (target_np - target_np.min()) / (target_np.max() - target_np.min() + 1e-8)
        
        # Binarize
        pred_binary = (pred_norm > threshold).astype(np.float32)
        target_binary = (target_norm > threshold).astype(np.float32)
        
        # IoU (Intersection over Union)
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        iou = intersection / (union + 1e-8)
        
        # Dice coefficient
        dice = 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)
        
        return {
            'iou': iou,
            'dice': dice,
        }
    
    def evaluate_batch(self, predictions, targets, input_sequences=None):
        """
        Evaluate a batch of predictions
        
        Args:
            predictions: (B, 1, 64, 64)
            targets: (B, 1, 64, 64)
            input_sequences: optional (B, L, 64, 64) for temporal metrics
        
        Returns:
            dict with all computed metrics
        """
        batch_metrics = {}
        
        # Pixel-level metrics
        pixel_metrics = self.compute_pixel_metrics(predictions, targets)
        batch_metrics.update(pixel_metrics)
        
        # Structural similarity
        ssim_metrics = self.compute_structural_similarity(predictions, targets)
        batch_metrics.update(ssim_metrics)
        
        # Infection level (scalar)
        infection_metrics = self.compute_infection_level_error(predictions, targets)
        batch_metrics.update(infection_metrics)
        
        # Spatial gradients
        gradient_metrics = self.compute_spatial_gradient_error(predictions, targets)
        batch_metrics.update(gradient_metrics)
        
        # Thresholded segmentation metrics
        seg_metrics = self.compute_infection_thresholded_metrics(predictions, targets)
        batch_metrics.update(seg_metrics)
        
        # Temporal metrics (if input sequences provided)
        if input_sequences is not None:
            temporal_metrics = self.compute_temporal_consistency(input_sequences, targets)
            batch_metrics.update(temporal_metrics)
        
        return batch_metrics
    
    def accumulate_metrics(self, batch_metrics):
        """Store metrics for later averaging"""
        for key, value in batch_metrics.items():
            if isinstance(value, (int, float, np.number)):
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
    
    def get_summary_report(self):
        """Generate summary statistics across all batches"""
        report = {}
        
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                report[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        return report


def evaluate_model(model, dataloader, device='cpu', num_batches=None):
    """
    Evaluate CNN model on test dataset
    
    Args:
        model: DiseaseCNN model
        dataloader: test DataLoader
        device: torch device
        num_batches: if set, only evaluate first N batches
    
    Returns:
        full report dict with metrics
    """
    model = model.to(device)
    model.eval()
    
    evaluator = MetricsEvaluator(device=device)
    
    print("Starting evaluation...")
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    x, y = batch_data
                    input_seq = x
                elif len(batch_data) == 3:
                    x, y, _ = batch_data  # metadata is ignored
                    input_seq = x
                else:
                    continue
            else:
                continue
            
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Evaluate batch
            batch_metrics = evaluator.evaluate_batch(
                predictions, y, 
                input_sequences=input_seq if 'input_seq' in locals() else None
            )
            evaluator.accumulate_metrics(batch_metrics)
            
            batch_count += 1
            if (batch_count + 1) % 10 == 0:
                print(f"  Evaluated {batch_count + 1} batches...")
    
    # Generate report
    report = evaluator.get_summary_report()
    report['total_batches_evaluated'] = batch_count
    
    return report, evaluator


def print_report(report):
    """Pretty print evaluation report"""
    print("\n" + "="*80)
    print("CNN MODEL EVALUATION REPORT")
    print("="*80)
    
    print(f"\nTotal batches evaluated: {report.pop('total_batches_evaluated')}\n")
    
    # Organize metrics by category
    categories = {
        'Pixel-Level Regression': ['mse', 'mae', 'rmse', 'r2'],
        'Image Quality': ['ssim_mean', 'psnr_mean'],
        'Infection Severity': ['infection_level_mae', 'mean_relative_error_percent'],
        'Spatial Accuracy': ['spatial_gradient_error_mean', 'iou', 'dice'],
        'Temporal Dynamics': ['temporal_consistency_error'],
    }
    
    for category, metrics in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for metric_name in metrics:
            if metric_name in report:
                stats = report[metric_name]
                print(f"  {metric_name:.<40} "
                      f"Mean: {stats['mean']:>10.4f} | "
                      f"Std: {stats['std']:>10.4f} | "
                      f"Range: [{stats['min']:>8.4f}, {stats['max']:>8.4f}]")
    
    print("\n" + "="*80)


def save_report(report, output_path):
    """Save report to JSON"""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python(item) for item in obj]
        return obj
    
    report_serializable = convert_to_python(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_serializable, f, indent=4)
    
    print(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN model on test dataset")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Temporal sequence length')
    parser.add_argument('--output_report', type=str, default='evaluation_report.json',
                        help='Path to save evaluation report')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='If set, only evaluate first N batches')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = DiseaseCNN(in_frames=args.sequence_length)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load dataset
    print(f"Loading test data from {args.test_data}...")
    dataset = TemporalCNNDataset(
        json_file=args.test_data,
        sequence_length=args.sequence_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    report, evaluator = evaluate_model(
        model, dataloader, 
        device=args.device,
        num_batches=args.num_batches
    )
    
    # Print and save results
    print_report(report)
    save_report(report, args.output_report)


if __name__ == '__main__':
    main()