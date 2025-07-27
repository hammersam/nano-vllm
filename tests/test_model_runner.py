import pytest
import torch
from unittest.mock import Mock, patch
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config


class TestModelRunner:
    
    def test_model_runner_initialization(self):
        """Test basic model runner initialization"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model') as mock_get_model:
            mock_model = Mock()
            mock_get_model.return_value = mock_model
            
            runner = ModelRunner(config)
            
            assert runner.config == config
            assert runner.model == mock_model
            assert runner.dtype == torch.float16  # Default dtype
    
    def test_prepare_inputs(self):
        """Test input preparation for model forward pass"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model'):
            runner = ModelRunner(config)
            
            # Mock sequences
            seq1 = Mock()
            seq1.token_ids = [1, 2, 3]
            seq1.num_cached_tokens = 0
            seq1.block_table = [0, 1]
            
            seq2 = Mock()
            seq2.token_ids = [4, 5]
            seq2.num_cached_tokens = 1
            seq2.block_table = [2, 3]
            
            sequences = [seq1, seq2]
            
            inputs = runner._prepare_inputs(sequences)
            
            assert "input_ids" in inputs
            assert "position_ids" in inputs
            assert "kvcache_block_table" in inputs
            assert inputs["input_ids"].shape[0] == 2  # Batch size
    
    def test_forward_pass_cpu_only(self):
        """Test forward pass logic (CPU-only version)"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = (torch.randn(2, 100), None)
            mock_get_model.return_value = mock_model
            
            runner = ModelRunner(config)
            
            # Mock sequences
            seq1 = Mock()
            seq1.token_ids = [1, 2, 3]
            seq1.num_cached_tokens = 0
            seq1.block_table = [0, 1]
            seq1.seq_id = "seq1"
            
            seq2 = Mock()
            seq2.token_ids = [4, 5]
            seq2.num_cached_tokens = 1
            seq2.block_table = [2, 3]
            seq2.seq_id = "seq2"
            
            sequences = [seq1, seq2]
            
            # Mock sampler
            with patch.object(runner, 'sampler') as mock_sampler:
                mock_sampler.return_value = torch.tensor([42, 43])
                
                output = runner.forward(sequences)
                
                assert len(output.next_tokens) == 2
                assert len(output.finished) == 2
                assert output.next_tokens == [42, 43]
                assert output.finished == [False, False]
    
    def test_forward_pass_empty_sequences(self):
        """Test forward pass with empty sequences"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model'):
            runner = ModelRunner(config)
            
            sequences = []
            
            output = runner.forward(sequences)
            
            assert len(output.next_tokens) == 0
            assert len(output.finished) == 0
    
    def test_forward_pass_single_sequence(self):
        """Test forward pass with single sequence"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model') as mock_get_model:
            mock_model = Mock()
            mock_model.return_value = (torch.randn(1, 100), None)
            mock_get_model.return_value = mock_model
            
            runner = ModelRunner(config)
            
            # Mock sequence
            seq = Mock()
            seq.token_ids = [1, 2, 3]
            seq.num_cached_tokens = 0
            seq.block_table = [0, 1]
            seq.seq_id = "seq1"
            
            sequences = [seq]
            
            # Mock sampler
            with patch.object(runner, 'sampler') as mock_sampler:
                mock_sampler.return_value = torch.tensor([42])
                
                output = runner.forward(sequences)
                
                assert len(output.next_tokens) == 1
                assert len(output.finished) == 1
                assert output.next_tokens == [42]
                assert output.finished == [False]
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model'), \
             patch('nanovllm.engine.model_runner.get_tokenizer') as mock_get_tokenizer:
            
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mock_get_tokenizer.return_value = mock_tokenizer
            
            runner = ModelRunner(config)
            
            assert runner.tokenizer == mock_tokenizer
            
            # Test tokenize method
            tokens = runner.tokenize("Hello world")
            assert tokens == [1, 2, 3]
    
    def test_detokenize(self):
        """Test detokenization"""
        config = Config("test_model", num_gpu_blocks=10, kvcache_block_size=16)
        
        with patch('nanovllm.engine.model_runner.get_model'), \
             patch('nanovllm.engine.model_runner.get_tokenizer') as mock_get_tokenizer:
            
            mock_tokenizer = Mock()
            mock_tokenizer.decode.return_value = "Hello world"
            mock_get_tokenizer.return_value = mock_tokenizer
            
            runner = ModelRunner(config)
            
            text = runner.detokenize([1, 2, 3])
            assert text == "Hello world"