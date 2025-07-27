import pytest
import time
import statistics
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager


class TestBenchmark:
    """Benchmark tests for performance regression detection"""

    def benchmark_engine_initialization(self):
        """Benchmark engine initialization time"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Warm up
            for _ in range(3):
                LLMEngine(config)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                engine = LLMEngine(config)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            max_time = max(times)
            
            # Regression thresholds (adjust based on hardware)
            assert avg_time < 0.1, f"Engine init too slow: {avg_time:.3f}s"
            assert max_time < 0.5, f"Engine init max too slow: {max_time:.3f}s"
            
            return {
                'avg_init_time': avg_time,
                'max_init_time': max_time,
                'times': times
            }

    def benchmark_request_processing(self):
        """Benchmark request processing throughput"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner') as mock_model_runner:
            
            # Setup fast mock
            mock_output = Mock()
            mock_output.next_tokens = [42]
            mock_output.finished = [False]
            mock_model_runner.return_value.forward.return_value = mock_output
            
            engine = LLMEngine(config)
            
            # Add test requests
            for i in range(100):
                sampling_params = SamplingParams(max_tokens=1)
                engine.add_request(f"bench_req{i}", f"prompt {i}", sampling_params)
            
            # Benchmark processing
            times = []
            tokens_processed = []
            
            for _ in range(50):
                start_time = time.perf_counter()
                results = engine.step()
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                tokens_processed.append(len(results))
            
            total_time = sum(times)
            total_tokens = sum(tokens_processed)
            throughput = total_tokens / total_time if total_time > 0 else 0
            
            # Performance thresholds
            assert throughput > 10, f"Throughput too low: {throughput:.1f} tokens/sec"
            assert statistics.mean(times) < 0.01, f"Step too slow: {statistics.mean(times):.3f}s"
            
            return {
                'throughput_tokens_per_sec': throughput,
                'total_tokens': total_tokens,
                'total_time': total_time,
                'avg_step_time': statistics.mean(times)
            }

    def benchmark_scalability(self):
        """Benchmark scalability with different load levels"""
        config = Config("test_model", num_gpu_blocks=1000, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner') as mock_model_runner:
            
            mock_output = Mock()
            mock_output.next_tokens = [42]
            mock_output.finished = [False]
            mock_model_runner.return_value.forward.return_value = mock_output
            
            results = {}
            
            # Test different load levels
            load_levels = [1, 10, 50, 100, 200]
            
            for load in load_levels:
                engine = LLMEngine(config)
                
                # Add requests
                for i in range(load):
                    sampling_params = SamplingParams(max_tokens=1)
                    engine.add_request(f"scalability_req{i}", f"prompt {i}", sampling_params)
                
                # Benchmark
                start_time = time.perf_counter()
                for _ in range(min(load, 50)):  # Limit iterations for large loads
                    engine.step()
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                throughput = load / total_time if total_time > 0 else 0
                
                results[load] = {
                    'load': load,
                    'total_time': total_time,
                    'throughput': throughput,
                    'latency_per_request': total_time / min(load, 50)
                }
                
                del engine
            
            # Verify scalability
            baseline = results[1]['throughput']
            peak_throughput = max(r['throughput'] for r in results.values())
            
            # Should scale reasonably (not perfect linear scaling due to overhead)
            scalability_ratio = peak_throughput / baseline
            assert scalability_ratio > 10, f"Poor scalability: {scalability_ratio:.1f}x"
            
            return results

    def benchmark_memory_efficiency(self):
        """Benchmark memory usage efficiency"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create engine
            engine = LLMEngine(config)
            
            # Add many requests
            for i in range(1000):
                sampling_params = SamplingParams(max_tokens=1)
                engine.add_request(f"mem_req{i}", f"prompt {i}", sampling_params)
            
            # Process some requests
            for _ in range(100):
                engine.step()
            
            # Check memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_request = (peak_memory - initial_memory) / 1000
            
            # Memory efficiency thresholds
            assert memory_per_request < 0.1, f"High memory per request: {memory_per_request:.3f}MB"
            assert peak_memory < initial_memory + 100, f"Excessive memory usage: {peak_memory:.1f}MB"
            
            return {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_per_request_mb': memory_per_request
            }

    def benchmark_block_allocation_performance(self):
        """Benchmark block allocation/deallocation performance"""
        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.config import Config
        
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            
            config = Config("test_model", num_gpu_blocks=10000, kvcache_block_size=16)
            
            block_manager = BlockManager(config)
            
            # Benchmark allocation
            allocation_times = []
            for size in [1, 10, 100, 1000]:
                start_time = time.perf_counter()
                for _ in range(100):
                    block_manager.allocate(size)
                    block_manager.free(size)
                end_time = time.perf_counter()
                
                total_time = end_time - start_time
                avg_time_per_op = total_time / 200  # 100 alloc + 100 free
                
                allocation_times.append({
                    'size': size,
                    'total_time': total_time,
                    'avg_time_per_op': avg_time_per_op,
                    'ops_per_sec': 200 / total_time if total_time > 0 else 0
                })
            
            # Performance thresholds
            for result in allocation_times:
                assert result['avg_time_per_op'] < 0.001, \
                    f"Slow allocation: {result['avg_time_per_op']:.4f}s for size {result['size']}"
            
            return allocation_times

    def benchmark_sequence_creation_performance(self):
        """Benchmark sequence creation performance"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        creation_times = []
        
        # Test different sequence sizes
        sizes = [1, 10, 100, 1000]
        
        for size in sizes:
            tokens = list(range(size))
            sampling_params = SamplingParams(max_tokens=10)
            
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                seq = Sequence(tokens, sampling_params)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            creation_times.append({
                'size': size,
                'avg_creation_time': avg_time,
                'creations_per_sec': 1 / avg_time if avg_time > 0 else 0
            })
        
        # Performance regression thresholds
        for result in creation_times:
            assert result['avg_creation_time'] < 0.001, \
                f"Slow sequence creation: {result['avg_creation_time']:.4f}s for size {result['size']}"
        
        return creation_times

    def benchmark_scheduler_performance(self):
        """Benchmark scheduler performance under load"""
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.config import Config
        
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            
            config = Config("test_model", num_gpu_blocks=1000, kvcache_block_size=16)
            
            # Test different load levels
            load_results = []
            
            for num_sequences in [10, 100, 500, 1000]:
                scheduler = Scheduler(config)
                
                # Create sequences
                sequences = []
                for i in range(num_sequences):
                    tokens = list(range(50))  # 50 tokens each
                    sampling_params = SamplingParams(max_tokens=1)
                    seq = Sequence(tokens, sampling_params)
                    sequences.append(seq)
                    scheduler.add(seq)
                
                # Benchmark batch selection
                times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    batch = scheduler.get_next_batch()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                avg_time = statistics.mean(times)
                
                load_results.append({
                    'num_sequences': num_sequences,
                    'batch_size': len(batch),
                    'avg_selection_time': avg_time,
                    'selections_per_sec': 1 / avg_time if avg_time > 0 else 0
                })
                
                del scheduler
                del sequences
            
            # Performance regression thresholds
            for result in load_results:
                assert result['avg_selection_time'] < 0.01, \
                    f"Slow batch selection: {result['avg_selection_time']:.4f}s for {result['num_sequences']} sequences"
            
            return load_results

    def test_performance_regression_suite(self):
        """Complete performance regression test suite"""
        results = {}
        
        # Run all benchmarks
        results['engine_init'] = self.benchmark_engine_initialization()
        results['request_processing'] = self.benchmark_request_processing()
        results['scalability'] = self.benchmark_scalability()
        results['memory_efficiency'] = self.benchmark_memory_efficiency()
        results['block_allocation'] = self.benchmark_block_allocation_performance()
        results['sequence_creation'] = self.benchmark_sequence_creation_performance()
        results['scheduler_performance'] = self.benchmark_scheduler_performance()
        
        # Generate performance report
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_benchmarks': len(results),
                'all_passed': True,
                'slowest_component': max(results.items(), key=lambda x: x[1]['avg_time'] if 'avg_time' in x[1] else 0)
            },
            'details': results
        }
        
        return report

    @pytest.mark.benchmark
    def test_all_performance_regressions(self):
        """Run complete performance regression test"""
        report = self.test_performance_regression_suite()
        
        # Assert no regressions
        assert report['summary']['all_passed'], "Performance regressions detected"
        
        # Print summary for CI/CD
        print("\n=== Performance Benchmark Results ===")
        for component, metrics in report.items():
            if component != 'summary' and component != 'timestamp':
                print(f"{component}: {metrics}")
        
        return report