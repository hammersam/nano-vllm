import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
import concurrent.futures


class TestConcurrent:
    """Tests for concurrent request handling and thread safety"""
    
    def test_concurrent_request_addition(self):
        """Test adding requests concurrently"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            results = []
            errors = []
            
            def add_request_worker(worker_id, num_requests):
                try:
                    for i in range(num_requests):
                        sampling_params = SamplingParams(max_tokens=2)
                        engine.add_request(
                            f"worker{worker_id}_req{i}",
                            f"prompt from worker {worker_id} request {i}",
                            sampling_params
                        )
                    results.append(f"worker{worker_id}_done")
                except Exception as e:
                    errors.append(str(e))
            
            # Start multiple threads adding requests
            threads = []
            for i in range(5):
                thread = threading.Thread(
                    target=add_request_worker,
                    args=(i, 10)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify no errors
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5
            
            # Verify requests were added
            assert engine.scheduler.add.call_count == 50
    
    def test_concurrent_step_operations(self):
        """Test concurrent step operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Setup mock behavior
            engine.scheduler.is_finished.return_value = False
            engine.scheduler.get_next_batch.return_value = []
            
            results_queue = queue.Queue()
            errors = []
            
            def step_worker(worker_id, iterations):
                try:
                    for i in range(iterations):
                        result = engine.step()
                        results_queue.put((worker_id, i, len(result)))
                        time.sleep(0.001)  # Small delay to increase concurrency
                except Exception as e:
                    errors.append(str(e))
            
            # Add some initial requests
            for i in range(10):
                sampling_params = SamplingParams(max_tokens=1)
                engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
            
            # Start concurrent step operations
            threads = []
            for i in range(3):
                thread = threading.Thread(
                    target=step_worker,
                    args=(i, 5)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            # Verify operations completed
            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results) > 0
    
    def test_concurrent_add_and_step(self):
        """Test concurrent add and step operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Setup mock behavior
            engine.scheduler.is_finished.return_value = False
            mock_output = Mock()
            mock_output.next_tokens = [42]
            mock_output.finished = [False]
            engine.model_runner.forward.return_value = mock_output
            
            add_results = []
            step_results = []
            errors = []
            stop_flag = threading.Event()
            
            def add_worker():
                try:
                    for i in range(50):
                        if stop_flag.is_set():
                            break
                        sampling_params = SamplingParams(max_tokens=1)
                        engine.add_request(f"concurrent_req{i}", f"prompt {i}", sampling_params)
                        add_results.append(i)
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(f"add: {str(e)}")
            
            def step_worker():
                try:
                    for i in range(30):
                        if stop_flag.is_set():
                            break
                        result = engine.step()
                        step_results.append(len(result))
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(f"step: {str(e)}")
            
            # Start concurrent operations
            add_thread = threading.Thread(target=add_worker)
            step_thread = threading.Thread(target=step_worker)
            
            add_thread.start()
            step_thread.start()
            
            # Let them run for a bit
            time.sleep(0.1)
            stop_flag.set()
            
            add_thread.join()
            step_thread.join()
            
            # Verify no errors
            assert len(errors) == 0, f"Errors: {errors}"
            assert len(add_results) > 0
            assert len(step_results) > 0
    
    def test_concurrent_abort_operations(self):
        """Test concurrent abort operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Add initial requests
            for i in range(20):
                sampling_params = SamplingParams(max_tokens=1)
                engine.add_request(f"req{i}", f"prompt {i}", sampling_params)
            
            abort_results = []
            errors = []
            
            def abort_worker():
                try:
                    for i in range(10):
                        engine.abort_request(f"req{i}")
                        abort_results.append(i)
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(str(e))
            
            # Start concurrent abort operations
            threads = []
            for i in range(3):
                thread = threading.Thread(target=abort_worker)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            assert len(errors) == 0, f"Errors: {errors}"
            assert len(abort_results) > 0
    
    def test_thread_pool_execution(self):
        """Test using thread pool for concurrent operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            def process_request(request_id):
                sampling_params = SamplingParams(max_tokens=2)
                engine.add_request(request_id, f"prompt for {request_id}", sampling_params)
                
                # Process a few steps
                results = []
                for _ in range(3):
                    step_result = engine.step()
                    results.extend(step_result)
                
                return len(results)
            
            # Use thread pool to process requests concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i in range(20):
                    future = executor.submit(process_request, f"pool_req{i}")
                    futures.append(future)
                
                # Collect results
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # All requests should complete
            assert len(results) == 20
            assert all(isinstance(r, int) for r in results)
    
    def test_request_queue_overflow(self):
        """Test handling request queue overflow"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Simulate queue overflow with many rapid requests
            def flood_worker():
                for i in range(100):
                    try:
                        sampling_params = SamplingParams(max_tokens=1)
                        engine.add_request(f"flood_req{i}", f"prompt {i}", sampling_params)
                    except Exception as e:
                        return str(e)
                return None
            
            # Run multiple flood workers
            threads = []
            errors = []
            
            for i in range(10):
                thread = threading.Thread(target=lambda: errors.append(flood_worker()))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # System should handle gracefully
            non_none_errors = [e for e in errors if e is not None]
            # Some errors might be expected under extreme load
            assert len(non_none_errors) <= 5  # Allow some failures
    
    def test_concurrent_engine_creation(self):
        """Test creating multiple engine instances concurrently"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        engines = []
        errors = []
        
        def create_engine():
            try:
                with patch('nanovllm.engine.llm_engine.BlockManager'), \
                     patch('nanovllm.engine.llm_engine.Scheduler'), \
                     patch('nanovllm.engine.llm_engine.ModelRunner'):
                    
                    engine = LLMEngine(config)
                    engines.append(engine)
            except Exception as e:
                errors.append(str(e))
        
        # Create engines concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_engine)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(engines) == 5
    
    def test_shared_resource_access(self):
        """Test concurrent access to shared resources"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            results = []
            errors = []
            
            def resource_worker(worker_id):
                try:
                    # Concurrent access to engine methods
                    free_blocks = engine.get_num_free_blocks()
                    
                    sampling_params = SamplingParams(max_tokens=1)
                    engine.add_request(f"shared_req{worker_id}", f"prompt {worker_id}", sampling_params)
                    
                    result = engine.step()
                    results.append((worker_id, free_blocks, len(result)))
                except Exception as e:
                    errors.append(str(e))
            
            # Run concurrent resource access
            threads = []
            for i in range(10):
                thread = threading.Thread(target=resource_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            assert len(errors) == 0, f"Errors: {errors}"
            assert len(results) > 0
            
            # All should have accessed the same engine state
            blocks = [r[1] for r in results]
            assert all(b == blocks[0] for b in blocks)  # Consistent state
    
    def test_deadlock_prevention(self):
        """Test prevention of deadlocks in concurrent operations"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            def mixed_operations():
                for i in range(20):
                    sampling_params = SamplingParams(max_tokens=1)
                    engine.add_request(f"deadlock_req{i}", f"prompt {i}", sampling_params)
                    
                    if i % 3 == 0:
                        engine.abort_request(f"deadlock_req{i-1}")
                    
                    engine.step()
                    time.sleep(0.001)
            
            # Run operations that could cause deadlocks
            threads = []
            for i in range(3):
                thread = threading.Thread(target=mixed_operations)
                threads.append(thread)
                thread.start()
            
            # Use timeout to detect deadlocks
            start_time = time.time()
            for thread in threads:
                thread.join(timeout=5.0)  # 5 second timeout
                assert not thread.is_alive(), "Potential deadlock detected"
            
            assert time.time() - start_time < 5.0, "Operations took too long, possible deadlock"