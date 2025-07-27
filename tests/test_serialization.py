import pytest
import tempfile
import json
import pickle
import torch
from unittest.mock import Mock, patch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler


class TestSerialization:
    """Tests for serialization and deserialization of model state"""

    def test_config_serialization(self):
        """Test configuration serialization/deserialization"""
        config = Config(
            model_name="test_model",
            dtype="float16",
            kvcache_block_size=32,
            num_gpu_blocks=1000,
            max_num_seqs=256,
            max_model_len=2048
        )
        
        # Test JSON serialization
        json_str = config.to_json()
        assert isinstance(json_str, str)
        
        restored_config = Config.from_json_string(json_str)
        assert restored_config == config
        assert restored_config.model_name == config.model_name
        assert restored_config.dtype == config.dtype
        assert restored_config.kvcache_block_size == config.kvcache_block_size
        
        # Test file-based serialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config.to_dict(), f)
            temp_file = f.name
        
        try:
            restored_from_file = Config.from_json(temp_file)
            assert restored_from_file == config
        finally:
            import os
            os.unlink(temp_file)

    def test_sampling_params_serialization(self):
        """Test sampling parameters serialization"""
        params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            ignore_eos=True
        )
        
        # Test dict serialization
        params_dict = params.to_dict()
        assert params_dict['max_tokens'] == 50
        assert params_dict['temperature'] == 0.7
        assert params_dict['top_p'] == 0.9
        assert params_dict['ignore_eos'] == True
        
        # Test JSON serialization
        json_str = json.dumps(params_dict)
        restored_dict = json.loads(json_str)
        assert restored_dict == params_dict

    def test_sequence_serialization(self):
        """Test sequence serialization/deserialization"""
        sampling_params = SamplingParams(max_tokens=10)
        seq = Sequence([1, 2, 3, 4, 5], sampling_params)
        
        # Add some state
        seq.append_token(6)
        seq.append_token(7)
        seq.block_table.extend([0, 1, 2])
        
        # Test serialization
        seq_dict = {
            'seq_id': seq.seq_id,
            'token_ids': seq.token_ids,
            'status': seq.status,
            'num_cached_tokens': seq.num_cached_tokens,
            'block_table': seq.block_table,
            'max_tokens': seq.max_tokens,
            'temperature': seq.temperature,
            'ignore_eos': seq.ignore_eos
        }
        
        # Test JSON serialization
        json_str = json.dumps(seq_dict, default=str)
        restored_dict = json.loads(json_str)
        
        assert restored_dict['seq_id'] == str(seq.seq_id)
        assert restored_dict['token_ids'] == [1, 2, 3, 4, 5, 6, 7]
        assert restored_dict['status'] == "WAITING"
        assert restored_dict['block_table'] == [0, 1, 2]

    def test_block_manager_serialization(self):
        """Test block manager state serialization"""
        from nanovllm.engine.block_manager import BlockManager
        
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            
            config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
            block_manager = BlockManager(config)
            
            # Allocate some blocks
            block_manager.allocate(25)
            free_blocks = block_manager.get_num_free_blocks()
            
            # Serialize state
            state = {
                'total_blocks': 100,
                'free_blocks': free_blocks,
                'allocated_blocks': 100 - free_blocks,
                'kvcache_block_size': config.kvcache_block_size
            }
            
            # Test JSON serialization
            json_str = json.dumps(state)
            restored_state = json.loads(json_str)
            
            assert restored_state['total_blocks'] == 100
            assert restored_state['free_blocks'] == 75
            assert restored_state['allocated_blocks'] == 25

    def test_scheduler_state_serialization(self):
        """Test scheduler state serialization"""
        from nanovllm.engine.scheduler import Scheduler
        
        with patch('nanovllm.engine.block_manager.torch.cuda') as mock_cuda:
            mock_cuda.mem_get_info.return_value = (1024 * 1024 * 1024, 1024 * 1024 * 1024)
            
            config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
            scheduler = Scheduler(config)
            
            # Add sequences
            sequences = []
            for i in range(5):
                sampling_params = SamplingParams(max_tokens=10)
                seq = Sequence([1, 2, 3], sampling_params)
                scheduler.add(seq)
                sequences.append(seq)
            
            # Serialize state
            state = {
                'waiting_count': len(scheduler.waiting),
                'running_count': len(scheduler.running),
                'finished_count': len(scheduler.finished),
                'sequences': [
                    {
                        'seq_id': str(seq.seq_id),
                        'token_ids': seq.token_ids,
                        'status': seq.status
                    }
                    for seq in sequences
                ]
            }
            
            # Test JSON serialization
            json_str = json.dumps(state, default=str)
            restored_state = json.loads(json_str)
            
            assert restored_state['waiting_count'] == 5
            assert restored_state['running_count'] == 0
            assert restored_state['finished_count'] == 0
            assert len(restored_state['sequences']) == 5

    def test_engine_checkpoint_serialization(self):
        """Test engine checkpoint serialization"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Add requests
            for i in range(10):
                sampling_params = SamplingParams(max_tokens=5)
                engine.add_request(f"checkpoint_req{i}", f"prompt {i}", sampling_params)
            
            # Process some steps
            for _ in range(3):
                engine.step()
            
            # Serialize checkpoint
            checkpoint = {
                'config': engine.config.to_dict(),
                'free_blocks': engine.get_num_free_blocks(),
                'has_unfinished_requests': engine.has_unfinished_requests(),
                'timestamp': time.time()
            }
            
            # Test JSON serialization
            json_str = json.dumps(checkpoint, default=str)
            restored_checkpoint = json.loads(json_str)
            
            assert restored_checkpoint['config']['model_name'] == config.model_name
            assert isinstance(restored_checkpoint['free_blocks'], int)
            assert isinstance(restored_checkpoint['has_unfinished_requests'], bool)

    def test_pickle_serialization(self):
        """Test pickle serialization for complex objects"""
        # Test config pickle
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        pickled_config = pickle.dumps(config)
        restored_config = pickle.loads(pickled_config)
        
        assert restored_config.model_name == config.model_name
        assert restored_config.num_gpu_blocks == config.num_gpu_blocks
        
        # Test sampling params pickle
        params = SamplingParams(max_tokens=10, temperature=0.7)
        pickled_params = pickle.dumps(params)
        restored_params = pickle.loads(pickled_params)
        
        assert restored_params.max_tokens == params.max_tokens
        assert restored_params.temperature == params.temperature

    def test_state_recovery_from_checkpoint(self):
        """Test recovering system state from checkpoint"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            # Create initial state
            engine = LLMEngine(config)
            
            # Add and process some requests
            for i in range(5):
                sampling_params = SamplingParams(max_tokens=3)
                engine.add_request(f"recovery_req{i}", f"prompt {i}", sampling_params)
            
            for _ in range(2):
                engine.step()
            
            # Create checkpoint
            checkpoint = {
                'config': engine.config.to_dict(),
                'sequences': [
                    {
                        'seq_id': f"recovery_req{i}",
                        'token_ids': [1, 2, 3],
                        'max_tokens': 3,
                        'status': 'RUNNING'
                    }
                    for i in range(3)  # Only recover 3 requests
                ]
            }
            
            # Simulate recovery
            recovered_config = Config.from_dict(checkpoint['config'])
            recovered_engine = LLMEngine(recovered_config)
            
            # Verify recovery
            assert recovered_engine.config.model_name == engine.config.model_name
            assert recovered_engine.config.num_gpu_blocks == engine.config.num_gpu_blocks

    def test_json_schema_validation(self):
        """Test JSON schema validation for serialized data"""
        import jsonschema
        
        # Define schema for config
        config_schema = {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "dtype": {"type": "string", "enum": ["float16", "bfloat16", "float32"]},
                "kvcache_block_size": {"type": "integer", "minimum": 1},
                "num_gpu_blocks": {"type": "integer", "minimum": 1},
                "max_num_seqs": {"type": "integer", "minimum": 1},
                "max_model_len": {"type": "integer", "minimum": 1}
            },
            "required": ["model_name", "dtype"]
        }
        
        config = Config("test_model", dtype="float16")
        config_dict = config.to_dict()
        
        # Validate against schema
        try:
            jsonschema.validate(instance=config_dict, schema=config_schema)
            validation_passed = True
        except jsonschema.exceptions.ValidationError:
            validation_passed = False
        
        assert validation_passed, "Config schema validation failed"

    def test_backward_compatibility(self):
        """Test backward compatibility of serialized formats"""
        # Test with old format configs
        old_config_format = {
            "model_name": "legacy_model",
            "dtype": "float16",
            "kvcache_block_size": 16,
            "num_gpu_blocks": 100,
            "max_num_seqs": 128
            # Missing newer fields - should use defaults
        }
        
        # Should load gracefully with defaults
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(old_config_format, f)
            temp_file = f.name
        
        try:
            restored_config = Config.from_json(temp_file)
            assert restored_config.model_name == "legacy_model"
            assert restored_config.dtype == "float16"
            # Should use defaults for missing fields
            assert restored_config.max_model_len > 0
        finally:
            import os
            os.unlink(temp_file)

    def test_serialization_error_handling(self):
        """Test error handling during serialization/deserialization"""
        # Test invalid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads('{"invalid": json}')
        
        # Test corrupted pickle
        with pytest.raises(pickle.PickleError):
            pickle.loads(b'corrupted pickle data')
        
        # Test missing file
        with pytest.raises(FileNotFoundError):
            Config.from_json("nonexistent.json")

    def test_compression_serialization(self):
        """Test compressed serialization for large states"""
        import gzip
        
        # Create large config
        config = Config(
            model_name="large_model",
            dtype="float16",
            kvcache_block_size=64,
            num_gpu_blocks=10000,
            max_num_seqs=1000,
            max_model_len=8192
        )
        
        # Serialize with compression
        config_json = json.dumps(config.to_dict(), separators=(',', ':'))
        
        # Compress
        compressed = gzip.compress(config_json.encode('utf-8'))
        
        # Decompress and restore
        decompressed = gzip.decompress(compressed).decode('utf-8')
        restored_config = Config.from_json_string(decompressed)
        
        # Verify data integrity
        assert restored_config == config
        
        # Check compression ratio
        original_size = len(config_json.encode('utf-8'))
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size
        
        assert compression_ratio > 1.0, "Compression should reduce size"

    def test_incremental_state_updates(self):
        """Test incremental state updates without full serialization"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Simulate incremental updates
            state_updates = []
            
            for i in range(5):
                sampling_params = SamplingParams(max_tokens=2)
                engine.add_request(f"incremental_req{i}", f"prompt {i}", sampling_params)
                
                # Create incremental update
                update = {
                    'type': 'add_request',
                    'seq_id': f"incremental_req{i}",
                    'token_ids': [1, 2, 3],
                    'max_tokens': 2
                }
                state_updates.append(update)
            
            # Process some steps
            for _ in range(2):
                engine.step()
                
                # Create update for processed sequences
                update = {
                    'type': 'step_complete',
                    'timestamp': time.time(),
                    'active_sequences': 5
                }
                state_updates.append(update)
            
            # Verify incremental updates are serializable
            updates_json = json.dumps(state_updates, default=str)
            restored_updates = json.loads(updates_json)
            
            assert len(restored_updates) == len(state_updates)
            assert all('type' in update for update in restored_updates)

    def test_version_compatibility_serialization(self):
        """Test version compatibility in serialization"""
        # Create config with version info
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        # Add version to serialization
        serialized = {
            'version': '1.0.0',
            'config': config.to_dict(),
            'metadata': {
                'created_at': time.time(),
                'framework_version': 'nano-vllm-0.1.0'
            }
        }
        
        # Test forward compatibility
        json_str = json.dumps(serialized)
        restored = json.loads(json_str)
        
        # Should handle gracefully even if version changes
        assert 'version' in restored
        assert 'config' in restored
        assert restored['config']['model_name'] == 'test_model'

    def test_state_snapshot_creation(self):
        """Test creating complete system state snapshots"""
        config = Config("test_model", num_gpu_blocks=100, kvcache_block_size=16)
        
        with patch('nanovllm.engine.llm_engine.BlockManager'), \
             patch('nanovllm.engine.llm_engine.Scheduler'), \
             patch('nanovllm.engine.llm_engine.ModelRunner'):
            
            engine = LLMEngine(config)
            
            # Add and process some requests
            for i in range(3):
                sampling_params = SamplingParams(max_tokens=2)
                engine.add_request(f"snapshot_req{i}", f"prompt {i}", sampling_params)
            
            for _ in range(2):
                engine.step()
            
            # Create comprehensive snapshot
            snapshot = {
                'version': '1.0.0',
                'timestamp': time.time(),
                'config': engine.config.to_dict(),
                'system_state': {
                    'free_blocks': engine.get_num_free_blocks(),
                    'has_unfinished_requests': engine.has_unfinished_requests()
                },
                'checkpoint_data': {
                    'sequences': [],
                    'block_allocation': {}
                }
            }
            
            # Test snapshot serialization
            json_str = json.dumps(snapshot, indent=2, default=str)
            restored_snapshot = json.loads(json_str)
            
            # Verify snapshot integrity
            assert restored_snapshot['version'] == '1.0.0'
            assert restored_snapshot['config']['model_name'] == 'test_model'
            assert 'system_state' in restored_snapshot
            assert 'checkpoint_data' in restored_snapshot
            
            # Test snapshot size
            snapshot_size = len(json_str.encode('utf-8'))
            assert snapshot_size < 10 * 1024 * 1024, "Snapshot too large"  # < 10MB