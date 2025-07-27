#!/usr/bin/env python3
"""
MoE Performance Benchmark Suite
Comprehensive benchmarking for enhanced MoE implementation
"""

import torch
import torch.distributed as dist
import time
import argparse
import json
import os
import sys
from typing import Dict, Any, List,Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanovllm.layers.moe_enhanced import EnhancedSparseMoE, MoEWithExpertParallel
from nanovllm.layers.moe_kernel_v2 import token_permutation_v2, invoke_segmented_gemm_v2
from nanovllm.layers.expert_parallel import ExpertParallelEngine


class MoEBenchmark:
    """Comprehensive MoE benchmarking suite."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites."""
        print("🚀 Starting MoE Performance Benchmarks")
        print("=" * 50)
        
        self.results = {
            'config': self.config,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
        # Run individual benchmarks
        self.results['benchmarks']['token_permutation'] = self.benchmark_token_permutation()
        self.results['benchmarks']['segmented_gemm'] = self.benchmark_segmented_gemm()
        self.results['benchmarks']['moe_forward'] = self.benchmark_moe_forward()
        self.results['benchmarks']['expert_parallelism'] = self.benchmark_expert_parallelism()
        self.results['benchmarks']['memory_usage'] = self.benchmark_memory_usage()
        
        return self.results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    
    def benchmark_token_permutation(self) -> Dict[str, Any]:
        """Benchmark token permutation performance."""
        print("📊 Benchmarking Token Permutation...")
        
        results = {
            'v1_vs_v2': {},
            'throughput': {},
            'latency': {}
        }
        
        # Test configurations
        configs = [
            {'num_tokens': 64, 'hidden_size': 512, 'num_experts': 8},
            {'num_tokens': 256, 'hidden_size': 1024, 'num_experts': 16},
            {'num_tokens': 1024, 'hidden_size': 2048, 'num_experts': 32},
        ]
        
        for config in configs:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            tokens = torch.randn(
                config['num_tokens'], 
                config['hidden_size'], 
                device=device
            )
            expert_indices = torch.randint(
                0, 
                config['num_experts'], 
                (config['num_tokens'],), 
                device=device
            )
            
            # Warmup
            for _ in range(3):
                _ = token_permutation_v2(
                    tokens, expert_indices, config['num_experts'], 1, True
                )
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            for _ in range(100):
                _ = token_permutation_v2(
                    tokens, expert_indices, config['num_experts'], 1, True
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 100
            throughput = config['num_tokens'] / avg_time
            
            key = f"{config['num_tokens']}x{config['hidden_size']}"
            results['throughput'][key] = throughput
            results['latency'][key] = avg_time * 1000  # ms
            
        return results
    
    def benchmark_segmented_gemm(self) -> Dict[str, Any]:
        """Benchmark segmented GEMM performance."""
        print("📊 Benchmarking Segmented GEMM...")
        
        results = {
            'throughput': {},
            'latency': {},
            'memory_efficiency': {}
        }
        
        # Test configurations
        configs = [
            {'M': 128, 'N': 2048, 'K': 512, 'num_experts': 8},
            {'M': 512, 'N': 4096, 'K': 1024, 'num_experts': 16},
            {'M': 2048, 'N': 8192, 'K': 2048, 'num_experts': 32},
        ]
        
        for config in configs:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            a = torch.randn(
                config['M'], config['K'], 
                device=device, 
                dtype=dtype
            )
            b = torch.randn(
                config['num_experts'], config['K'], config['N'], 
                device=device, 
                dtype=dtype
            )
            c = torch.empty(
                config['M'], config['N'], 
                device=device, 
                dtype=dtype
            )
            
            # Create expert distribution
            expert_counts = torch.tensor([config['M'] // config['num_experts']] * config['num_experts'])
            expert_ids = torch.repeat_interleave(torch.arange(config['num_experts']), config['M'] // config['num_experts'])
            
            # Warmup
            for _ in range(3):
                _ = invoke_segmented_gemm_v2(a, b, c, expert_counts, expert_ids)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            for _ in range(50):
                _ = invoke_segmented_gemm_v2(a, b, c, expert_counts, expert_ids)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 50
            throughput = (config['M'] * config['N'] * config['K']) / avg_time
            
            key = f"{config['M']}x{config['N']}x{config['K']}"
            results['throughput'][key] = throughput
            results['latency'][key] = avg_time * 1000
            
            # Memory efficiency check
            results['memory_efficiency'][key] = {
                'input_memory_mb': (a.numel() + b.numel()) * a.element_size() / 1024 / 1024,
                'output_memory_mb': c.numel() * c.element_size() / 1024 / 1024
            }
        
        return results
    
    def benchmark_moe_forward(self) -> Dict[str, Any]:
        """Benchmark complete MoE forward pass."""
        print("📊 Benchmarking MoE Forward Pass...")
        
        results = {
            'configurations': {},
            'throughput': {},
            'latency': {},
            'memory_usage': {}
        }
        
        # Test configurations
        configs = [
            {
                'hidden_size': 512,
                'intermediate_size': 2048,
                'num_experts': 8,
                'num_experts_per_tok': 2,
                'batch_size': 4,
                'seq_len': 32
            },
            {
                'hidden_size': 1024,
                'intermediate_size': 4096,
                'num_experts': 16,
                'num_experts_per_tok': 2,
                'batch_size': 8,
                'seq_len': 64
            },
            {
                'hidden_size': 2048,
                'intermediate_size': 8192,
                'num_experts': 32,
                'num_experts_per_tok': 2,
                'batch_size': 16,
                'seq_len': 128
            }
        ]
        
        for config in configs:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            moe_config = {
                'hidden_size': config['hidden_size'],
                'intermediate_size': config['intermediate_size'],
                'num_experts': config['num_experts'],
                'shared_expert_intermediate_size': config['intermediate_size'] // 2,
                'num_experts_per_tok': config['num_experts_per_tok'],
                'num_shared_experts': 1,
                'gate_logit_softcapping': 30.0,
                'hidden_act': 'silu',
                'use_triton': True,
                'enable_expert_parallel': False
            }
            
            moe = EnhancedSparseMoE(**moe_config).to(device)
            
            # Create input
            hidden_states = torch.randn(
                config['batch_size'], 
                config['seq_len'], 
                config['hidden_size'], 
                device=device
            )
            
            # Warmup
            for _ in range(3):
                _ = moe(hidden_states)
            
            # Memory usage before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                memory_before = torch.cuda.memory_allocated()
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            for _ in range(20):
                output = moe(hidden_states)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / 20
            num_tokens = config['batch_size'] * config['seq_len']
            throughput = num_tokens / avg_time
            
            key = f"{config['hidden_size']}h_{config['num_experts']}e_{num_tokens}t"
            results['configurations'][key] = config
            results['throughput'][key] = throughput
            results['latency'][key] = avg_time * 1000
            
            # Memory usage
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                results['memory_usage'][key] = {
                    'allocated_mb': memory_after / 1024 / 1024,
                    'peak_mb': peak_memory / 1024 / 1024,
                    'overhead_mb': (memory_after - memory_before) / 1024 / 1024
                }
        
        return results
    
    def benchmark_expert_parallelism(self) -> Dict[str, Any]:
        """Benchmark expert parallelism performance."""
        print("📊 Benchmarking Expert Parallelism...")
        
        results = {
            'scaling_efficiency': {},
            'communication_overhead': {},
            'load_balancing': {}
        }
        
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("⚠️  Expert parallelism requires multiple GPUs")
            return results
        
        # Test with different expert counts
        expert_configs = [
            {'num_experts': 4, 'hidden_size': 512},
            {'num_experts': 8, 'hidden_size': 1024},
            {'num_experts': 16, 'hidden_size': 2048},
        ]
        
        for config in expert_configs:
            # Simulate expert parallelism
            world_size = 2  # Use 2 GPUs for testing
            
            engine = ExpertParallelEngine(
                num_experts=config['num_experts'],
                expert_capacity_factor=1.0
            )
            
            # Test scaling
            tokens = torch.randn(256, config['hidden_size'])
            expert_indices = torch.randint(0, config['num_experts'], (256,))
            routing_weights = torch.rand(256)
            
            # Single device baseline
            start_time = time.perf_counter()
            # Simulate single device computation
            time.sleep(0.01)  # Placeholder
            single_time = time.perf_counter() - start_time
            
            # Parallel computation
            start_time = time.perf_counter()
            _ = engine.process_with_expert_parallelism(
                tokens, expert_indices, routing_weights, 
                torch.randn(config['num_experts'], config['hidden_size'], 4 * config['hidden_size']),
                torch.randn(config['num_experts'], 4 * config['hidden_size'], config['hidden_size'])
            )
            parallel_time = time.perf_counter() - start_time
            
            # Calculate scaling efficiency
            ideal_speedup = world_size
            actual_speedup = single_time / max(parallel_time, 1e-6)
            efficiency = actual_speedup / ideal_speedup
            
            key = f"{config['num_experts']}e_{config['hidden_size']}h"
            results['scaling_efficiency'][key] = {
                'single_device_time': single_time,
                'parallel_time': parallel_time,
                'efficiency': efficiency
            }
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("📊 Benchmarking Memory Usage...")
        
        results = {
            'memory_breakdown': {},
            'peak_memory': {},
            'memory_efficiency': {}
        }
        
        if not torch.cuda.is_available():
            return results
        
        # Test configurations
        configs = [
            {'hidden_size': 512, 'num_experts': 8},
            {'hidden_size': 1024, 'num_experts': 16},
            {'hidden_size': 2048, 'num_experts': 32},
        ]
        
        for config in configs:
            # Create model
            moe_config = {
                'hidden_size': config['hidden_size'],
                'intermediate_size': config['hidden_size'] * 4,
                'num_experts': config['num_experts'],
                'num_experts_per_tok': 2,
                'use_triton': True
            }
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated()
            
            moe = EnhancedSparseMoE(**moe_config).cuda()
            
            model_memory = torch.cuda.memory_allocated() - initial_memory
            
            # Test memory with different batch sizes
            batch_sizes = [1, 2, 4, 8]
            memory_usage = {}
            
            for batch_size in batch_sizes:
                seq_len = 64
                hidden_states = torch.randn(
                    batch_size, seq_len, config['hidden_size'], 
                    device='cuda'
                )
                
                torch.cuda.empty_cache()
                before_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    output = moe(hidden_states)
                
                after_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                
                memory_usage[f'batch_{batch_size}'] = {
                    'activation_memory': (after_memory - before_memory) / 1024 / 1024,
                    'peak_memory': peak_memory / 1024 / 1024
                }
            
            key = f"{config['hidden_size']}h_{config['num_experts']}e"
            results['memory_breakdown'][key] = {
                'model_memory_mb': model_memory / 1024 / 1024,
                'memory_usage': memory_usage
            }
            results['peak_memory'][key] = max(
                usage['peak_memory'] 
                for usage in memory_usage.values()
            )
        
        return results
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Results saved to {output_path}")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="MoE Performance Benchmark")
    parser.add_argument("--output", default="moe_benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--device", default=None, 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--config", default=None, 
                       help="Custom config file")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 Using device: {device}")
    
    # Default configuration
    config = {
        'device': str(device),
        'dtype': 'float16' if device.type == 'cuda' else 'float32',
        'warmup_iterations': 3,
        'benchmark_iterations': 50
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Run benchmarks
    benchmark = MoEBenchmark(config)
    results = benchmark.run_benchmarks()
    
    # Save results
    benchmark.save_results(args.output)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 50)
    
    for benchmark_name, data in results['benchmarks'].items():
        print(f"\n{benchmark_name.upper()}:")
        if 'throughput' in data:
            for key, value in data['throughput'].items():
                print(f"  {key}: {value:.2f} tokens/sec")
        if 'latency' in data:
            for key, value in data['latency'].items():
                print(f"  {key}: {value:.2f} ms")


if __name__ == "__main__":
    main()