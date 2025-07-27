import pytest
import gc
import psutil
import os
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
import threading
import time


class TestMemory:
    """Tests for memory management and leak detection"""
    
    def get_memory_usage(self):
        """Helper to get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_no_memory_leak_in_engine_lifecycle(self):
        """Test that engine doesn't leak memory across requests"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            initial_memory = self.get_memory_usage()
            
            # Run multiple engine cycles
            for i in range(10):
                engine = LLMEngine(config)
                
                # Add and process requests
                sampling_params = SamplingParams(max_tokens=5)
                for j in range(5):
                    engine.add_request(f"req_{i}_{j}", f"prompt {j}", sampling_params)
                
                # Process steps
                for _ in range(3):
                    results = engine.step()
                
                # Clean up
                del engine
                gc.collect()
            
            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Allow for some variance, but no significant leak
            assert memory_increase < 50, f"Memory leak detected: {memory_increase}MB"
    
    def test_block_manager_memory_tracking(self):
        """Test block manager tracks memory correctly"""
        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.config import Config
        
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            
            config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
            
            initial_memory = self.get_memory_usage()
            
            # Create block manager
            block_manager = BlockManager(config)
            
            # Allocate and free blocks multiple times
            for _ in range(100):
                block_manager.allocate(10)
                block_manager.free(10)
            
            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Memory should not increase significantly
            assert memory_increase < 10, f"Block manager memory leak: {memory_increase}MB"
            
            # Verify all blocks are free
            assert block_manager.get_num_free_blocks() == 100
    
    def test_sequence_memory_cleanup(self):
        """Test sequence memory is properly cleaned up"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        initial_memory = self.get_memory_usage()
        
        sequences = []
        
        # Create many sequences
        for i in range(1000):
            sampling_params = SamplingParams(max_tokens=10)
            seq = Sequence([1, 2, 3] * 100, sampling_params)  # Long sequences
            sequences.append(seq)
        
        # Store current memory
        mid_memory = self.get_memory_usage()
        
        # Delete sequences
        del sequences
        gc.collect()
        
        final_memory = self.get_memory_usage()
        
        # Memory should decrease significantly
        assert final_memory < mid_memory, "Sequence memory not cleaned up"
    
    def test_scheduler_memory_stability(self):
        """Test scheduler maintains memory stability"""
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.config import Config
        
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        initial_memory = self.get_memory_usage()
        
        # Create scheduler and add/remove many sequences
        scheduler = Scheduler(config)
        
        for i in range(100):
            sampling_params = SamplingParams(max_tokens=5)
            seq = Sequence([1, 2, 3], sampling_params)
            scheduler.add(seq)
            
            # Process and remove
            batch = scheduler.get_next_batch()
            scheduler.abort(seq.seq_id)
        
        # Clean up
        del scheduler
        gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not leak memory
        assert memory_increase < 20, f"Scheduler memory leak: {memory_increase}MB"
    
    def test_config_memory_efficiency(self):
        """Test config objects don't consume excessive memory"""
        from nanovllm.config import Config
        
        initial_memory = self.get_memory_usage()
        
        # Create many config objects
        configs = []
        for i in range(1000):
            config = Config(
                f"test_model_{i}",
                dtype="float16",
                kvcache_block_size=16 + (i % 10),
                num_gpu_blocks=100 + (i % 50)
            )
            configs.append(config)
        
        mid_memory = self.get_memory_usage()
        
        # Delete configs
        del configs
        gc.collect()
        
        final_memory = self.get_memory_usage()
        
        # Memory should be cleaned up
        assert final_memory < mid_memory, "Config memory not cleaned up"
    
    def test_memory_pressure_simulation(self):
        """Test system behavior under memory pressure"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager') as mock_block_manager, \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Simulate low memory conditions
            mock_block_manager_instance = Mock()
            mock_block_manager_instance.get_num_free_blocks.return_value = 1
            mock_block_manager_instance.allocate.side_effect = [True, False, False]
            mock_block_manager.return_value = mock_block_manager_instance
            
            engine = LLMEngine(config)
            
            initial_memory = self.get_memory_usage()
            
            # Add many requests under memory pressure
            sampling_params = SamplingParams(max_tokens=50)
            for i in range(50):
                try:
                    engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
                except Exception:
                    # Expected due to memory pressure
                    pass
            
            # Process under pressure
            for _ in range(10):
                try:
                    results = engine.step()
                except Exception:
                    # Expected under pressure
                    pass
            
            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Should not leak significantly
            assert memory_increase < 30, f"Memory leak under pressure: {memory_increase}MB"
    
    def test_concurrent_memory_usage(self):
        """Test memory usage with concurrent operations"""
        config = Config("test_model", num_gpu_blocks=50, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            initial_memory = self.get_memory_usage()
            
            def worker(thread_id):
                engine = LLMEngine(config)
                
                for i in range(10):
                    sampling_params = SamplingParams(max_tokens=2)
                    engine.add_request(f"thread{thread_id}_req{i}", f"prompt {i}", sampling_params)
                    
                    for _ in range(2):
                        engine.step()
                
                del engine
            
            # Run concurrent operations
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            gc.collect()
            
            final_memory = self.get_memory_usage()
            memory_increase = final_memory - initial_memory
            
            # Should not leak significantly
            assert memory_increase < 50, f"Concurrent memory leak: {memory_increase}MB"
    
    def test_large_sequence_memory(self):
        """Test memory usage with very large sequences"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        initial_memory = self.get_memory_usage()
        
        sequences = []
        
        # Create sequences with very large token lists
        for i in range(100):
            large_token_list = list(range(1000))  # 1000 tokens
            sampling_params = SamplingParams(max_tokens=10)
            seq = Sequence(large_token_list, sampling_params)
            sequences.append(seq)
        
        mid_memory = self.get_memory_usage()
        
        # Clean up
        del sequences
        gc.collect()
        
        final_memory = self.get_memory_usage()
        
        # Memory should be reclaimed
        memory_reclaimed = mid_memory - final_memory
        assert memory_reclaimed > 0, "Large sequence memory not reclaimed"
    
    def test_circular_reference_cleanup(self):
        """Test circular references are properly cleaned up"""
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.config import Config
        
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        initial_memory = self.get_memory_usage()
        
        # Create scheduler with sequences
        scheduler = Scheduler(config)
        
        sequences = []
        for i in range(50):
            sampling_params = SamplingParams(max_tokens=1)
            seq = Sequence([1, 2, 3], sampling_params)
            sequences.append(seq)
            scheduler.add(seq)
        
        # Create circular references
        for seq in sequences:
            seq.scheduler_ref = scheduler
        
        mid_memory = self.get_memory_usage()
        
        # Clear all references
        del sequences
        del scheduler
        gc.collect()
        
        final_memory = self.get_memory_usage()
        
        # Memory should be cleaned up despite circular references
        memory_reclaimed = mid_memory - final_memory
        assert memory_reclaimed > 0, "Circular reference memory not cleaned up"
    
    def test_memory_usage_reporting(self):
        """Test memory usage reporting functionality"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Add some requests
            sampling_params = SamplingParams(max_tokens=5)
            for i in range(10):
                engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
            
            # Get memory info
            memory_before = self.get_memory_usage()
            
            # Process some steps
            for _ in range(3):
                results = engine.step()
            
            memory_after = self.get_memory_usage()
            
            # Memory should not explode
            memory_increase = memory_after - memory_before
            assert memory_increase < 100, f"Memory usage exploded: {memory_increase}MB"
            
            # Verify we can get engine stats
            free_blocks = engine.get_num_free_blocks()
            assert isinstance(free_blocks, int)
            assert free_blocks >= 0