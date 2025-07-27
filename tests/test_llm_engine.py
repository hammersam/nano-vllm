import pytest
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams


class TestLLMEngine:
    
    def test_engine_initialization(self):
        """Test basic engine initialization"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager') as mock_block_manager, \
             patch('nanovllm.engine.llm_engine.Scheduler') as mock_scheduler, \
             patch('nanovllm.engine.llm_engine.ModelRunner') as mock_model_runner:
            
            mock_block_manager_instance = Mock()
            mock_block_manager.return_value = mock_block_manager_instance
            
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance
            
            mock_model_runner_instance = Mock()
            mock_model_runner.return_value = mock_model_runner_instance
            
            engine = LLMEngine(config)
            
            assert engine.config == config
            assert engine.block_manager == mock_block_manager_instance
            assert engine.scheduler == mock_scheduler_instance
            assert engine.model_runner == mock_model_runner_instance
    
    def test_add_request(self):
        """Test adding a request to the engine"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            prompt = "Hello world"
            sampling_params = SamplingParams(max_tokens=10)
            
            engine.add_request("test_id", prompt, sampling_params)
            
            assert len(engine.scheduler.add.call_args_list) == 1
            call_args = engine.scheduler.add.call_args[0][0]
            assert call_args.token_ids == [1, 2, 3]  # Mock tokenizer would return this
    
    def test_step_empty_scheduler(self):
        """Test step when scheduler has no requests"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            engine.scheduler.is_finished.return_value = True
            
            result = engine.step()
            
            assert result == []
            engine.model_runner.forward.assert_not_called()
    
    def test_step_with_requests(self):
        """Test step with active requests"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Mock scheduler state
            mock_sequence = Mock()
            mock_sequence.seq_id = "test_seq"
            mock_sequence.status = "RUNNING"
            mock_sequence.num_cached_tokens = 0
            mock_sequence.num_prompt_tokens = 3
            mock_sequence.is_finished.return_value = False
            
            engine.scheduler.is_finished.return_value = False
            engine.scheduler.get_next_batch.return_value = [mock_sequence]
            
            # Mock model output
            mock_output = Mock()
            mock_output.next_tokens = [42]
            mock_output.finished = [False]
            engine.model_runner.forward.return_value = mock_output
            
            result = engine.step()
            
            assert len(result) == 1
            assert result[0].seq_id == "test_seq"
            assert result[0].token_id == 42
            assert result[0].finished == False
    
    def test_abort_request(self):
        """Test aborting a request"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            engine.abort_request("test_id")
            
            engine.scheduler.abort.assert_called_once_with("test_id")
    
    def test_get_num_free_blocks(self):
        """Test getting number of free blocks"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            engine.block_manager.get_num_free_blocks.return_value = 5
            
            assert engine.get_num_free_blocks() == 5
    
    def test_has_unfinished_requests(self):
        """Test checking for unfinished requests"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Test when scheduler has finished
            engine.scheduler.is_finished.return_value = True
            assert not engine.has_unfinished_requests()
            
            # Test when scheduler has unfinished requests
            engine.scheduler.is_finished.return_value = False
            assert engine.has_unfinished_requests()