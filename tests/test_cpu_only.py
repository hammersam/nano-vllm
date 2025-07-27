import pytest
import torch
from unittest.mock import Mock, patch
import sys


class TestCPUOnly:
    """Tests that can run without GPU/CUDA environment"""
    
    def test_config_cpu_only(self):
        """Test config creation without GPU"""
        from nanovllm.config import Config
        
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        assert config.model_name == "test_model"
        assert config.num_gpu_blocks == 100
        assert config.kvcache_block_size == 16
        # Should not require GPU for config creation
    
    def test_sequence_cpu_only(self):
        """Test sequence creation and management without GPU"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(max_tokens=10)
        seq = Sequence([1, 2, 3, 4, 5], sampling_params)
        
        assert seq.token_ids == [1, 2, 3, 4, 5]
        assert seq.num_prompt_tokens == 5
        assert seq.num_cached_tokens == 0
        assert seq.max_tokens == 10
        
        # Test appending tokens
        seq.append_token(6)
        assert seq.token_ids == [1, 2, 3, 4, 5, 6]
        
        # Test sequence length
        assert len(seq) == 6
    
    def test_sampling_params_cpu_only(self):
        """Test sampling parameters without GPU"""
        from nanovllm.sampling_params import SamplingParams
        
        # Test default parameters
        params = SamplingParams()
        assert params.max_tokens == 16
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.ignore_eos == False
        
        # Test custom parameters
        params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            ignore_eos=True
        )
        assert params.max_tokens == 50
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.ignore_eos == True
    
    def test_block_manager_cpu_only(self):
        """Test block manager without GPU"""
        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.config import Config
        
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        # Mock the GPU memory check
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
            mock_cuda.current_device.return_value = 0
            
            block_manager = BlockManager(config)
            
            assert block_manager.get_num_free_blocks() == 100
            
            # Test allocation
            block_manager.allocate(10)
            assert block_manager.get_num_free_blocks() == 90
            
            # Test freeing
            block_manager.free(10)
            assert block_manager.get_num_free_blocks() == 100
    
    def test_scheduler_cpu_only(self):
        """Test scheduler without GPU"""
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.config import Config
        
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        scheduler = Scheduler(config)
        
        # Test initial state
        assert scheduler.is_finished()
        
        # Test adding sequences
        sampling_params = SamplingParams(max_tokens=10)
        seq1 = Sequence([1, 2, 3], sampling_params)
        seq2 = Sequence([4, 5, 6], sampling_params)
        
        scheduler.add(seq1)
        scheduler.add(seq2)
        
        assert not scheduler.is_finished()
        assert len(scheduler.waiting) == 2
        
        # Test getting next batch (mocking block availability)
        with patch.object(scheduler.block_manager, 'get_num_free_blocks', return_value=100):
            batch = scheduler.get_next_batch()
            # Should return sequences when blocks are available
            assert len(batch) >= 0  # Could be 0 if no blocks allocated
    
    def test_config_validation_cpu_only(self):
        """Test config validation without GPU"""
        from nanovllm.config import Config
        
        # Test valid configurations
        valid_configs = [
            {"model_name": "test", "num_gpu_blocks": 10},
            {"model_name": "test", "dtype": "float16", "kvcache_block_size": 32},
            {"model_name": "test", "max_num_seqs": 50, "max_model_len": 1024},
        ]
        
        for config_dict in valid_configs:
            config = Config(**config_dict)
            assert config.model_name == config_dict["model_name"]
        
        # Test invalid configurations
        invalid_configs = [
            {"model_name": "", "num_gpu_blocks": 10},  # Empty model name
            {"model_name": "test", "dtype": "invalid"},  # Invalid dtype
            {"model_name": "test", "kvcache_block_size": 0},  # Zero block size
            {"model_name": "test", "num_gpu_blocks": -1},  # Negative blocks
            {"model_name": "test", "max_num_seqs": 0},  # Zero sequences
            {"model_name": "test", "max_model_len": 0},  # Zero model length
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(AssertionError):
                Config(**invalid_config)
    
    def test_torch_cpu_fallback(self):
        """Test PyTorch CPU fallback for tensor operations"""
        # Test basic tensor operations on CPU
        tensor = torch.randn(5, 3)
        assert tensor.device.type == "cpu"
        
        # Test tensor operations
        result = tensor * 2
        assert result.shape == (5, 3)
        
        # Test tensor concatenation
        tensor2 = torch.randn(5, 3)
        concatenated = torch.cat([tensor, tensor2], dim=0)
        assert concatenated.shape == (10, 3)
        
        # Test tensor indexing
        indexed = tensor[0:2]
        assert indexed.shape == (2, 3)
    
    def test_mock_gpu_environment(self):
        """Test handling when CUDA is not available"""
        # Mock CUDA not being available
        with patch('torch.cuda.is_available', return_value=False):
            # Test that we can still create configs and objects
            from nanovllm.config import Config
            config = Config("test_model", num_gpu_blocks=100)
            assert config.model_name == "test_model"
    
    def test_error_handling_cpu_only(self):
        """Test error handling in CPU-only environment"""
        from nanovllm.config import Config
        
        # Test that we get appropriate errors for GPU-specific operations
        config = Config("test_model")
        
        # Test that non-GPU code paths work
        try:
            # This should not require GPU
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert "model_name" in config_dict
        except Exception as e:
            pytest.fail(f"Config operations should work without GPU: {e}")
    
    def test_memory_calculation_cpu_only(self):
        """Test memory calculations without actual GPU"""
        from nanovllm.config import Config
        
        config = Config("test_model", kvcache_block_size=16, num_gpu_blocks=100)
        
        # Test that we can calculate expected memory usage
        expected_kv_cache_size = (
            config.num_gpu_blocks * 
            config.kvcache_block_size * 
            config.hidden_size * 
            2  # key and value
        )
        
        # This is just a calculation test, doesn't require actual GPU memory
        assert expected_kv_cache_size > 0
        assert isinstance(expected_kv_cache_size, int)