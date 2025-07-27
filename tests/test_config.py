import pytest
import tempfile
import json
import os
from nanovllm.config import Config


class TestConfig:
    
    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = Config("test_model")
        
        assert config.model_name == "test_model"
        assert config.dtype == "float16"
        assert config.kvcache_block_size == 16
        assert config.num_gpu_blocks > 0
        assert config.max_num_seqs == 128
        assert config.max_model_len > 0
    
    def test_custom_config_creation(self):
        """Test creating config with custom values"""
        config = Config(
            model_name="custom_model",
            dtype="bfloat16",
            kvcache_block_size=32,
            num_gpu_blocks=1000,
            max_num_seqs=256,
            max_model_len=2048
        )
        
        assert config.model_name == "custom_model"
        assert config.dtype == "bfloat16"
        assert config.kvcache_block_size == 32
        assert config.num_gpu_blocks == 1000
        assert config.max_num_seqs == 256
        assert config.max_model_len == 2048
    
    def test_invalid_dtype(self):
        """Test config creation with invalid dtype"""
        with pytest.raises(AssertionError):
            Config("test_model", dtype="invalid_dtype")
    
    def test_invalid_kvcache_block_size(self):
        """Test config creation with invalid kvcache_block_size"""
        with pytest.raises(AssertionError):
            Config("test_model", kvcache_block_size=0)
        
        with pytest.raises(AssertionError):
            Config("test_model", kvcache_block_size=-1)
    
    def test_invalid_num_gpu_blocks(self):
        """Test config creation with invalid num_gpu_blocks"""
        with pytest.raises(AssertionError):
            Config("test_model", num_gpu_blocks=0)
        
        with pytest.raises(AssertionError):
            Config("test_model", num_gpu_blocks=-1)
    
    def test_invalid_max_num_seqs(self):
        """Test config creation with invalid max_num_seqs"""
        with pytest.raises(AssertionError):
            Config("test_model", max_num_seqs=0)
        
        with pytest.raises(AssertionError):
            Config("test_model", max_num_seqs=-1)
    
    def test_invalid_max_model_len(self):
        """Test config creation with invalid max_model_len"""
        with pytest.raises(AssertionError):
            Config("test_model", max_model_len=0)
        
        with pytest.raises(AssertionError):
            Config("test_model", max_model_len=-1)
    
    def test_from_json_file(self):
        """Test creating config from JSON file"""
        config_data = {
            "model_name": "json_model",
            "dtype": "float32",
            "kvcache_block_size": 64,
            "num_gpu_blocks": 2000,
            "max_num_seqs": 512,
            "max_model_len": 4096
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = Config.from_json(temp_file)
            
            assert config.model_name == "json_model"
            assert config.dtype == "float32"
            assert config.kvcache_block_size == 64
            assert config.num_gpu_blocks == 2000
            assert config.max_num_seqs == 512
            assert config.max_model_len == 4096
        finally:
            os.unlink(temp_file)
    
    def test_from_json_file_missing_fields(self):
        """Test creating config from JSON file with missing fields (should use defaults)"""
        config_data = {
            "model_name": "partial_model",
            "dtype": "bfloat16"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            config = Config.from_json(temp_file)
            
            assert config.model_name == "partial_model"
            assert config.dtype == "bfloat16"
            assert config.kvcache_block_size == 16  # Default
            assert config.num_gpu_blocks > 0  # Default
        finally:
            os.unlink(temp_file)
    
    def test_from_json_file_invalid_json(self):
        """Test creating config from invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                Config.from_json(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_from_json_file_nonexistent(self):
        """Test creating config from nonexistent file"""
        with pytest.raises(FileNotFoundError):
            Config.from_json("nonexistent_file.json")
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = Config(
            model_name="test_model",
            dtype="float16",
            kvcache_block_size=32,
            num_gpu_blocks=1000
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "test_model"
        assert config_dict["dtype"] == "float16"
        assert config_dict["kvcache_block_size"] == 32
        assert config_dict["num_gpu_blocks"] == 1000
    
    def test_str_representation(self):
        """Test string representation of config"""
        config = Config("test_model")
        
        str_repr = str(config)
        
        assert "Config" in str_repr
        assert "test_model" in str_repr
        assert "float16" in str_repr
    
    def test_equality(self):
        """Test config equality"""
        config1 = Config("test_model", dtype="float16", kvcache_block_size=32)
        config2 = Config("test_model", dtype="float16", kvcache_block_size=32)
        config3 = Config("test_model", dtype="bfloat16", kvcache_block_size=32)
        
        assert config1 == config2
        assert config1 != config3
        assert config1 != "not a config"
    
    def test_hash(self):
        """Test config hash"""
        config1 = Config("test_model", dtype="float16")
        config2 = Config("test_model", dtype="float16")
        
        assert hash(config1) == hash(config2)
    
    def test_get_model_config_path(self):
        """Test getting model config path"""
        config = Config("test_model")
        
        # This should return a valid path structure
        model_path = config.get_model_path()
        assert isinstance(model_path, str)
        assert len(model_path) > 0
    
    def test_model_name_validation(self):
        """Test model name validation"""
        # Valid model names
        valid_names = [
            "test_model",
            "test-model",
            "test_model_v1",
            "123model",
            "model123"
        ]
        
        for name in valid_names:
            config = Config(name)
            assert config.model_name == name
        
        # Test empty model name
        with pytest.raises(AssertionError):
            Config("")
    
    def test_dtype_options(self):
        """Test all valid dtype options"""
        valid_dtypes = ["float16", "bfloat16", "float32"]
        
        for dtype in valid_dtypes:
            config = Config("test_model", dtype=dtype)
            assert config.dtype == dtype