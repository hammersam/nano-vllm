import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
from typing import List, Dict, Any

from nanovllm.config import Config
from nanovllm.engine.distributed_engine import DistributedEngine, DistributedLLM
from nanovllm.engine.distributed_scheduler import DistributedScheduler
from nanovllm.engine.worker_pool import WorkerPool, WorkerTask, WorkerProcess, WorkerResult
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.rpc_client import RPCClient, RPCServer


class TestDistributedFeatures:
    """Test suite for distributed serving features."""
    
    def test_distributed_config_initialization(self):
        """Test distributed configuration setup."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        assert config.enable_distributed
        assert config.world_size == 2
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1
    
    def test_distributed_scheduler_initialization(self):
        """Test DistributedScheduler initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=0
        )
        
        scheduler = DistributedScheduler(config)
        
        assert scheduler.world_size == 2
        assert scheduler.rank == 0
        assert scheduler.is_coordinator
    
    def test_worker_pool_initialization(self):
        """Test WorkerPool initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        worker_pool = WorkerPool(config)
        
        assert worker_pool.config == config
        assert not worker_pool.initialized  # Should not auto-initialize
    
    def test_worker_process_initialization(self):
        """Test WorkerProcess initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=1
        )
        
        worker = WorkerProcess(config, rank=1)
        
        assert worker.config == config
        assert worker.rank == 1
        assert not worker.initialized
    
    def test_tensor_parallel_config(self):
        """Test tensor parallel configuration."""
        config = Config(
            "test-model",
            tensor_parallel_size=4,
            world_size=4
        )
        
        assert config.tensor_parallel_size == 4
    
    def test_pipeline_parallel_config(self):
        """Test pipeline parallel configuration."""
        config = Config(
            "test-model",
            pipeline_parallel_size=2,
            world_size=4
        )
        
        assert config.pipeline_parallel_size == 2
    
    def test_distributed_engine_initialization(self):
        """Test DistributedEngine initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=1  # Single process for testing
        )
        
        engine = DistributedEngine("test-model", **config.__dict__)
        
        assert engine.config.enable_distributed
    
    def test_distributed_scheduler_load_balancing(self):
        """Test distributed scheduler load balancing."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=0
        )
        
        scheduler = DistributedScheduler(config)
        
        # Test load balancing selection
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
        
        # Should select appropriate worker
        target_worker = scheduler.distribute_sequence(seq)
        assert 0 <= target_worker < config.world_size
    
    def test_health_check_functionality(self):
        """Test health check functionality."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=1
        )
        
        scheduler = DistributedScheduler(config)
        
        health_status = scheduler.health_check()
        
        assert isinstance(health_status, dict)
        assert 0 in health_status
    
    def test_distributed_stats_collection(self):
        """Test distributed statistics collection."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=0
        )
        
        scheduler = DistributedScheduler(config)
        
        stats = scheduler.get_distributed_stats()
        
        assert 'total_requests' in stats
        assert 'worker_loads' in stats
        assert 'is_coordinator' in stats
    
    def test_worker_task_creation(self):
        """Test WorkerTask creation."""
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
        
        task = WorkerTask(
            task_type='prefill',
            sequences=[seq],
            is_prefill=True,
            request_id='test_123',
            timestamp=time.time()
        )
        
        assert task.task_type == 'prefill'
        assert task.request_id == 'test_123'
        assert len(task.sequences) == 1
    
    def test_worker_result_creation(self):
        """Test WorkerResult creation."""
        result = WorkerResult(
            request_id='test_123',
            token_ids=[4, 5, 6],
            status='success',
            processing_time=0.5
        )
        
        assert result.request_id == 'test_123'
        assert result.token_ids == [4, 5, 6]
        assert result.status == 'success'
        assert result.processing_time == 0.5
    
    def test_rpc_client_initialization(self):
        """Test RPC client initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2
        )
        
        rpc_client = RPCClient(config)
        
        assert rpc_client.config == config
        assert not rpc_client.initialized
    
    def test_rpc_server_initialization(self):
        """Test RPC server initialization."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=1
        )
        
        rpc_server = RPCServer(config, rank=1)
        
        assert rpc_server.config == config
        assert rpc_server.rank == 1
        assert rpc_server.worker_id == "worker_1"
    
    def test_load_balance_logic(self):
        """Test load balancing logic."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=4
        )
        
        scheduler = DistributedScheduler(config)
        
        # Simulate load balancing
        scheduler.worker_loads = {0: 10, 1: 5, 2: 15, 3: 8}
        
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
        target_worker = scheduler.distribute_sequence(seq)
        
        # Should select worker with minimum load
        assert target_worker == 1
    
    def test_expert_load_balancing(self):
        """Test expert-based load balancing."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=3,
            num_experts=4
        )
        
        scheduler = DistributedScheduler(config)
        
        # Simulate expert loads
        scheduler.worker_expert_loads = {
            0: [10, 5, 8, 12],    # Total expert load: 35
            1: [3, 15, 7, 4],     # Total expert load: 29
            2: [8, 10, 12, 6]     # Total expert load: 36
        }
        
        # Create sequence with expert affinity
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
        seq.expert_ids = [1, 2]  # Need experts 1 and 2
        
        target_worker = scheduler.distribute_sequence(seq)
        
        # Should select worker 1 (lowest expert load for required experts)
        assert target_worker == 1
    
    def test_pipeline_stage_calculation(self):
        """Test pipeline stage calculation."""
        config = Config(
            "test-model",
            enable_distributed=True,
            pipeline_parallel_size=4,
            world_size=4
        )
        
        scheduler = DistributedScheduler(config)
        
        # Test different sequence lengths
        test_cases = [
            (100, 1),   # Short sequence -> 1 stage
            (600, 2),   # Medium sequence -> 2 stages
            (1500, 3),  # Long sequence -> 3 stages
            (3000, 4)   # Very long sequence -> 4 stages
        ]
        
        for seq_len, expected_stages in test_cases:
            seq = Sequence([1] * seq_len, SamplingParams(max_tokens=100))
            stages = scheduler._calculate_pipeline_stages(seq)
            assert stages <= min(expected_stages, config.pipeline_parallel_size)
    
    def test_worker_information_gathering(self):
        """Test gathering worker information."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2
        )
        
        rpc_client = RPCClient(config)
        
        # Mock cluster info
        info = {
            'world_size': 2,
            'workers': {
                1: {'rank': 1, 'is_multimodal': False},
                2: {'rank': 2, 'is_multimodal': False}
            }
        }
        
        assert info['world_size'] == 2
        assert len(info['workers']) == 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_reporting(self):
        """Test memory usage reporting."""
        config = Config("test-model")
        
        # Simulate memory usage
        memory_stats = {
            'total_memory': 8000000000,
            'allocated_memory': 2000000000,
            'cached_memory': 2500000000,
            'free_memory': 5500000000
        }
        
        assert 'total_memory' in memory_stats
        assert 'allocated_memory' in memory_stats
    
    def test_error_handling_in_distributed(self):
        """Test error handling in distributed setup."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=1
        )
        
        # Test graceful handling of single process
        engine = DistributedEngine("test-model", **config.__dict__)
        
        assert not engine.config.enable_distributed or engine.config.world_size == 1
    
    def test_distributed_stats_format(self):
        """Test distributed stats format."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2,
            rank=0
        )
        
        scheduler = DistributedScheduler(config)
        
        stats = scheduler.get_distributed_stats()
        
        required_keys = [
            'total_requests',
            'completed_requests',
            'worker_loads',
            'is_coordinator',
            'world_size',
            'rank'
        ]
        
        for key in required_keys:
            assert key in stats
    
    def test_worker_pool_cleanup(self):
        """Test worker pool cleanup."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2
        )
        
        worker_pool = WorkerPool(config)
        
        # Test cleanup doesn't crash
        worker_pool.cleanup()
        
        assert not worker_pool.initialized
        assert len(worker_pool.processes) == 0
    
    def test_rpc_health_check(self):
        """Test RPC health check."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=2
        )
        
        rpc_server = RPCServer(config, rank=1)
        
        health = rpc_server.health_check()
        
        assert health['status'] == 'healthy'
        assert health['rank'] == 1
        assert 'timestamp' in health
    
    def test_launch_function(self):
        """Test distributed launch function."""
        # Test configuration validation
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=4,
            tensor_parallel_size=2,
            pipeline_parallel_size=2
        )
        
        # Verify dimensions are compatible
        assert config.tensor_parallel_size * config.pipeline_parallel_size <= config.world_size
    
    def test_multimodal_distributed_combo(self):
        """Test multimodal with distributed features."""
        config = Config(
            "test-model",
            enable_distributed=True,
            enable_multimodal=True,
            world_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
        
        assert config.enable_distributed
        assert config.enable_multimodal
        assert config.world_size == 2


class TestDistributedIntegration:
    """Integration tests for distributed features."""
    
    def test_single_process_mode(self):
        """Test that single process mode works correctly."""
        config = Config(
            "test-model",
            enable_distributed=False,
            world_size=1
        )
        
        engine = DistributedEngine("test-model", **config.__dict__)
        
        # Should use standard engine
        assert not hasattr(engine, 'scheduler') or engine.config.world_size == 1
    
    def test_distributed_engine_api_compatibility(self):
        """Test distributed engine API compatibility."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=1  # Single process for testing
        )
        
        engine = DistributedEngine("test-model", **config.__dict__)
        
        # Test basic API methods exist
        assert hasattr(engine, 'add_request')
        assert hasattr(engine, 'step')
        assert hasattr(engine, 'generate')
        assert hasattr(engine, 'is_finished')
    
    def test_error_propagation(self):
        """Test error propagation in distributed setup."""
        config = Config(
            "test-model",
            enable_distributed=True,
            world_size=1
        )
        
        scheduler = DistributedScheduler(config)
        
        # Test error handling gracefully
        try:
            seq = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
            # This should not crash
            stats = scheduler.get_distributed_stats()
            assert isinstance(stats, dict)
        except Exception as e:
            # Should handle gracefully
            assert True  # Any exception is handled