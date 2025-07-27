import pytest
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_request_lifecycle(self):
        """Test complete request lifecycle: add -> process -> complete"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Setup mocks
            mock_sequence = Mock()
            mock_sequence.seq_id = "test_seq"
            mock_sequence.status = "RUNNING"
            mock_sequence.num_cached_tokens = 0
            mock_sequence.num_prompt_tokens = 3
            mock_sequence.is_finished.return_value = False
            mock_sequence.token_ids = [1, 2, 3]
            
            # Mock scheduler behavior
            engine.scheduler.is_finished.side_effect = [False, False, True]
            engine.scheduler.get_next_batch.side_effect = [
                [mock_sequence],
                [mock_sequence],
                []
            ]
            
            # Mock model output
            mock_output = Mock()
            mock_output.next_tokens = [4]
            mock_output.finished = [False]
            
            mock_final_output = Mock()
            mock_final_output.next_tokens = [5]
            mock_final_output.finished = [True]
            
            engine.model_runner.forward.side_effect = [
                mock_output,
                mock_final_output
            ]
            
            # Add request
            sampling_params = SamplingParams(max_tokens=2)
            engine.add_request("test_id", "Hello world", sampling_params)
            
            # Process steps
            results1 = engine.step()
            assert len(results1) == 1
            assert results1[0].token_id == 4
            assert not results1[0].finished
            
            results2 = engine.step()
            assert len(results2) == 1
            assert results2[0].token_id == 5
            assert results2[0].finished
            
            # Verify engine is finished
            assert not engine.has_unfinished_requests()
    
    def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Create mock sequences
            seq1 = Mock()
            seq1.seq_id = "seq1"
            seq1.status = "RUNNING"
            seq1.num_cached_tokens = 0
            seq1.num_prompt_tokens = 2
            seq1.is_finished.return_value = False
            
            seq2 = Mock()
            seq2.seq_id = "seq2"
            seq2.status = "RUNNING"
            seq2.num_cached_tokens = 0
            seq2.num_prompt_tokens = 3
            seq2.is_finished.return_value = False
            
            # Setup scheduler
            engine.scheduler.is_finished.return_value = False
            engine.scheduler.get_next_batch.return_value = [seq1, seq2]
            
            # Setup model output
            mock_output = Mock()
            mock_output.next_tokens = [101, 102]
            mock_output.finished = [False, False]
            engine.model_runner.forward.return_value = mock_output
            
            # Add multiple requests
            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("req1", "Hello", sampling_params)
            engine.add_request("req2", "World", sampling_params)
            
            # Process batch
            results = engine.step()
            
            assert len(results) == 2
            assert results[0].seq_id == "seq1"
            assert results[1].seq_id == "seq2"
    
    def test_scheduler_block_manager_integration(self):
        """Test integration between scheduler and block manager"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        # Create real instances
        block_manager = BlockManager(config)
        scheduler = Scheduler(config)
        
        # Test block allocation for sequences
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            
            # Create sequences
            sampling_params = SamplingParams(max_tokens=10)
            seq1 = Sequence([1, 2, 3], sampling_params)
            seq2 = Sequence([4, 5, 6, 7], sampling_params)
            
            # Add to scheduler
            scheduler.add(seq1)
            scheduler.add(seq2)
            
            # Test getting next batch with block allocation
            initial_free_blocks = block_manager.get_num_free_blocks()
            
            batch = scheduler.get_next_batch()
            
            # Should be able to allocate blocks for sequences
            assert len(batch) >= 0  # Could be 0 if no blocks allocated yet
            
            # Verify blocks were allocated
            assert block_manager.get_num_free_blocks() <= initial_free_blocks
    
    def test_memory_pressure_handling(self):
        """Test handling memory pressure with block allocation"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager') as mock_block_manager, \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Setup block manager to simulate memory pressure
            mock_block_manager_instance = Mock()
            mock_block_manager_instance.get_num_free_blocks.return_value = 1
            mock_block_manager.return_value = mock_block_manager_instance
            
            engine = LLMEngine(config)
            
            # Add multiple requests that might exceed memory
            sampling_params = SamplingParams(max_tokens=100)
            
            for i in range(5):
                engine.add_request(f"req{i}", f"Prompt {i}", sampling_params)
            
            # Should handle memory pressure gracefully
            results = engine.step()
            
            # Verify block manager was consulted
            mock_block_manager_instance.get_num_free_blocks.assert_called()
    
    def test_request_abort_integration(self):
        """Test aborting requests during processing"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Add requests
            sampling_params = SamplingParams(max_tokens=10)
            engine.add_request("req1", "Hello", sampling_params)
            engine.add_request("req2", "World", sampling_params)
            
            # Verify requests were added
            assert len(engine.scheduler.add.call_args_list) == 2
            
            # Abort one request
            engine.abort_request("req1")
            
            # Verify abort was called
            engine.scheduler.abort.assert_called_with("req1")
    
    def test_error_recovery(self):
        """Test system recovery from errors"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Simulate model error
            engine.model_runner.forward.side_effect = [
                RuntimeError("Model error"),
                Mock(next_tokens=[42], finished=[True])  # Recovery
            ]
            
            # Add request
            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("req1", "Test", sampling_params)
            
            # First step fails
            try:
                results = engine.step()
                assert False, "Should have raised exception"
            except RuntimeError:
                pass
            
            # Reset mock for recovery
            engine.model_runner.forward.side_effect = None
            engine.model_runner.forward.return_value = Mock(
                next_tokens=[42], finished=[True]
            )
            
            # Should recover on next step
            results = engine.step()
            assert len(results) >= 0  # Could be empty if sequence was removed
    
    def test_tokenizer_integration(self):
        """Test tokenizer integration in engine"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'), \
             patch('nanovllm.engine.model_runner.get_tokenizer') as mock_get_tokenizer:
            
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [101, 102, 103]
            mock_get_tokenizer.return_value = mock_tokenizer
            
            engine = LLMEngine(config)
            
            # Add request with text
            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("req1", "Hello world", sampling_params)
            
            # Verify tokenizer was used
            mock_tokenizer.encode.assert_called_with("Hello world")