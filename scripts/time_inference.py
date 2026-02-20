import torch
import torch.nn as nn
import time
import numpy as np
from cnn_model import DiseaseCNN
import argparse


class InferenceProfiler:
    """Profile inference time and performance metrics for the CNN model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def profile_inference(self, input_shape, num_runs=100, warmup_runs=10):
        """
        Profile inference time
        
        Args:
            input_shape: (batch_size, sequence_length, height, width)
            num_runs: number of inference runs to average
            warmup_runs: warm-up runs before measurement
        
        Returns:
            dict with timing statistics
        """
        # Create dummy input
        x = torch.randn(input_shape, dtype=torch.float32).to(self.device)
        
        # Warmup
        print(f"Warming up with {warmup_runs} runs...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(x)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        print(f"Running {num_runs} inference iterations...")
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                start = time.perf_counter()
                _ = self.model(x)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end = time.perf_counter()
                times.append(end - start)
        
        times = np.array(times[1:])  # Remove first run (it might be slower)
        
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        
        results = {
            'batch_size': batch_size,
            'sequence_length': seq_length,
            'input_shape': input_shape,
            'device': str(self.device),
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'fps': batch_size / np.mean(times),  # Frames per second
            'latency_per_frame_ms': (np.mean(times) * 1000) / seq_length,
        }
        
        return results
    
    def profile_memory(self, input_shape):
        """
        Profile memory usage during inference
        
        Args:
            input_shape: (batch_size, sequence_length, height, width)
        
        Returns:
            dict with memory statistics
        """
        # Only works with CUDA
        if self.device.type != 'cuda':
            print("Memory profiling only available on CUDA devices")
            return None
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        x = torch.randn(input_shape, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _ = self.model(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        
        results = {
            'peak_memory_mb': peak_memory,
            'input_shape': input_shape,
        }
        
        return results
    
    def benchmark_different_configs(self, batch_sizes=[1, 2, 4, 8], 
                                    sequence_lengths=[3, 5, 10], 
                                    num_runs=50):
        """
        Benchmark multiple configurations (useful for drone deployment)
        
        Args:
            batch_sizes: list of batch sizes to test
            sequence_lengths: list of sequence lengths to test
            num_runs: number of runs per config
        
        Returns:
            list of results for each config
        """
        all_results = []
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                input_shape = (batch_size, seq_len, 64, 64)
                print(f"\nBenchmarking: batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    results = self.profile_inference(input_shape, num_runs=num_runs)
                    all_results.append(results)
                    
                    # Print summary
                    print(f"  Mean latency: {results['mean_time_ms']:.2f} ms")
                    print(f"  FPS: {results['fps']:.2f}")
                    print(f"  Per-frame latency: {results['latency_per_frame_ms']:.2f} ms")
                    
                except RuntimeError as e:
                    print(f"  Failed: {e}")
                    continue
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Profile CNN model inference time")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on (cpu/cuda)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Sequence length for temporal input')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run full benchmark with multiple configs')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of inference runs')
    
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = DiseaseCNN(in_frames=args.sequence_length)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create profiler
    profiler = InferenceProfiler(model, device=args.device)
    
    if args.benchmark:
        print("\n" + "="*60)
        print("RUNNING FULL BENCHMARK")
        print("="*60)
        results = profiler.benchmark_different_configs(num_runs=args.num_runs)
        
        # Print summary table
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"{'Batch':<6} {'Seq':<5} {'Latency (ms)':<15} {'FPS':<10} {'Per-Frame (ms)':<15}")
        print("-"*60)
        for r in results:
            print(f"{r['batch_size']:<6} {r['sequence_length']:<5} "
                  f"{r['mean_time_ms']:<15.2f} {r['fps']:<10.2f} "
                  f"{r['latency_per_frame_ms']:<15.2f}")
    
    else:
        print("\n" + "="*60)
        print("SINGLE CONFIGURATION TEST")
        print("="*60)
        input_shape = (args.batch_size, args.sequence_length, 64, 64)
        results = profiler.profile_inference(input_shape, num_runs=args.num_runs)
        
        print(f"\nInput shape: {results['input_shape']}")
        print(f"Mean inference time: {results['mean_time_ms']:.2f} ms ± {results['std_time_ms']:.2f} ms")
        print(f"Min/Max: {results['min_time_ms']:.2f} / {results['max_time_ms']:.2f} ms")
        print(f"Throughput: {results['fps']:.2f} sequences/second")
        print(f"Latency per frame: {results['latency_per_frame_ms']:.2f} ms")


if __name__ == '__main__':
    main()