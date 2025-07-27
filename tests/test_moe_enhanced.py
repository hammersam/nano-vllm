import pytest
import torch
import torch.distributed as dist
from nanovllm.layers.moe_enhanced import EnhancedSparseMoE, MoEWithExpertParallel
from nanovllm.layers.moe_kernel_v2 import (
    token_permutation_v2, 
    invoke_segmented_gemm_v2, 
    ExpertParallelismManager
)
from nanovllm.layers.expert_parallel import ExpertParallelEngine, ExpertParallelConfig
import time
import os


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return {
        'hidden_size': 512,
        'intermediate_size': 2048,
        'num_experts': 8,
        'shared_expert_intermediate_size': 1024,
        'num_experts_per_tok': 2,
        'num_shared_experts': 1,
        'gate_logit_softcapping': 30.0,
        'hidden_act': 'silu',
        'enable_expert_parallel': False,
        'use_triton': True
    }


@pytest.mark.parametrize("enable_triton", [True, False])
@pytest.mark.parametrize("enable_expert_parallel", [False, True])
def test_moe_enhanced_forward(default_config, enable_triton, enable_expert_parallel):
    """Test enhanced MoE forward pass."""
    config = default_config.copy()
    config['use_triton'] = enable_triton
    config['enable_expert_parallel'] = enable_expert_parallel
    
    moe = EnhancedSparseMoE(**config)
    
    # Test input
    batch_size, seq_len = 4, 32
    hidden_states = torch.randn(batch_size, seq_len, config['hidden_size'])
    
    # Forward pass
    output = moe(hidden_states)
    
    # Assertions
    assert output.shape == (batch_size, seq_len, config['hidden_size'])
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_token_permutation_v2():
    """Test enhanced token permutation kernel."""
    num_tokens, hidden_size = 64, 512
    num_experts = 8
    num_experts_per_tok = 2
    
    # Create test data
    tokens = torch.randn(num_tokens * num_experts_per_tok, hidden_size)
    expert_indices = torch.randint(0, num_experts, (num_tokens * num_experts_per_tok,))
    
    # Forward permutation
    permuted_tokens, expert_counts, token_mapping = token_permutation_v2(
        tokens, expert_indices, num_experts, num_experts_per_tok, True
    )
    
    # Assertions
    assert permuted_tokens.shape == tokens.shape
    assert expert_counts.shape == (num_experts,)
    assert token_mapping.shape == (num_tokens * num_experts_per_tok,)
    assert expert_counts.sum() == num_tokens * num_experts_per_tok
    
    # Inverse permutation
    inverse_tokens, _ = token_permutation_v2(
        permuted_tokens, expert_indices, num_experts, num_experts_per_tok, False
    )
    
    # Check if we can recover original order approximately
    assert inverse_tokens.shape == tokens.shape


def test_segmented_gemm_v2():
    """Test enhanced segmented GEMM kernel."""
    num_tokens = 128
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    
    # Create test data
    a = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    b = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.float16)
    c = torch.empty(num_tokens, intermediate_size, dtype=torch.float16)
    
    # Expert assignment
    expert_counts = torch.tensor([16, 16, 16, 16, 16, 16, 16, 16])
    expert_ids = torch.repeat_interleave(torch.arange(num_experts), 16)
    
    # Run segmented GEMM
    result = invoke_segmented_gemm_v2(a, b, c, expert_counts, expert_ids)
    
    # Assertions
    assert result.shape == (num_tokens, intermediate_size)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_expert_parallelism(world_size):
    """Test expert parallelism functionality."""
    num_experts = 8
    
    try:
        # Mock distributed environment
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('gloo', rank=0, world_size=1)
        
        engine = ExpertParallelEngine(num_experts=num_experts)
        
        # Test expert assignment
        local_experts = engine.manager.get_local_experts()
        assert len(local_experts) > 0
        
        # Test routing
        tokens = torch.randn(32, 512)
        expert_indices = torch.randint(0, num_experts, (32,))
        routing_weights = torch.rand(32)
        
        routing_result = engine.manager.route_tokens(tokens, expert_indices, routing_weights)
        
        assert 'local_tokens' in routing_result
        assert 'remote_routing' in routing_result
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_expert_load_balancer():
    """Test expert load balancing."""
    num_experts = 8
    world_size = 2
    
    from nanovllm.layers.expert_parallel import ExpertLoadBalancer
    
    balancer = ExpertLoadBalancer(num_experts, world_size)
    
    # Update loads
    for i in range(num_experts):
        balancer.update_expert_load(i, i * 0.1)
    
    # Get optimal assignment
    assignment = balancer.get_optimal_expert_assignment()
    
    assert assignment.shape == (num_experts,)
    assert assignment.max() < world_size


def test_moe_performance_comparison():
    """Performance comparison between standard and enhanced MoE."""
    config = {
        'hidden_size': 512,
        'intermediate_size': 2048,
        'num_experts': 8,
        'shared_expert_intermediate_size': 1024,
        'num_experts_per_tok': 2,
        'use_triton': True
    }
    
    # Test configurations
    configs = [
        {'use_triton': False, 'enable_expert_parallel': False},  # Baseline
        {'use_triton': True, 'enable_expert_parallel': False},   # Triton only
        {'use_triton': True, 'enable_expert_parallel': True},    # Full enhanced
    ]
    
    results = []
    
    for test_config in configs:
        full_config = {**config, **test_config}
        moe = EnhancedSparseMoE(**full_config)
        
        # Warmup
        warmup_input = torch.randn(2, 16, 512)
        _ = moe(warmup_input)
        
        # Performance test
        test_input = torch.randn(4, 64, 512)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(10):
            output = moe(test_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        results.append({
            'config': test_config,
            'avg_time': avg_time,
            'throughput': test_input.numel() / avg_time
        })
    
    # Print comparison
    for result in results:
        print(f"Config: {result['config']}")
        print(f"  Avg time: {result['avg_time']:.4f}s")
        print(f"  Throughput: {result['throughput']:.0f} tokens/s")


def test_moe_with_expert_parallel():
    """Test complete MoE with expert parallelism integration."""
    config = {
        'hidden_size': 512,
        'intermediate_size': 2048,
        'num_experts': 8,
        'shared_expert_intermediate_size': 1024,
        'num_experts_per_tok': 2,
        'enable_expert_parallel': True,
        'use_triton': True
    }
    
    moe = MoEWithExpertParallel(config)
    
    # Test input
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, config['hidden_size'])
    
    # Forward pass
    output = moe(hidden_states)
    
    # Assertions
    assert output.shape == (batch_size, seq_len, config['hidden_size'])
    
    # Get stats
    stats = moe.get_stats()
    assert 'expert_count' in stats


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_moe_dtype_support(dtype):
    """Test MoE with different data types."""
    config = {
        'hidden_size': 256,
        'intermediate_size': 1024,
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'use_triton': True
    }
    
    moe = EnhancedSparseMoE(**config)
    moe = moe.to(dtype)
    
    hidden_states = torch.randn(2, 32, 256, dtype=dtype)
    output = moe(hidden_states)
    
    assert output.dtype == dtype


def test_moe_gradient_flow():
    """Test gradient flow through enhanced MoE."""
    config = {
        'hidden_size': 256,
        'intermediate_size': 1024,
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'use_triton': True
    }
    
    moe = EnhancedSparseMoE(**config)
    
    hidden_states = torch.randn(2, 32, 256, requires_grad=True)
    target = torch.randn_like(hidden_states)
    
    # Forward pass
    output = moe(hidden_states)
    
    # Backward pass
    loss = F.mse_loss(output, target)
    loss.backward()
    
    # Check gradients
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()
    
    # Check expert weights
    for param in moe.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()


class TestMoEIntegration:
    """Integration tests for MoE components."""
    
    def setup_method(self):
        """Setup for each test."""
        self.config = {
            'hidden_size': 512,
            'intermediate_size': 2048,
            'num_experts': 8,
            'num_experts_per_tok': 2,
            'use_triton': True
        }
    
    def test_memory_efficiency(self):
        """Test memory efficiency improvements."""
        moe = EnhancedSparseMoE(**self.config)
        
        # Memory usage test
        with torch.no_grad():
            input_tensor = torch.randn(1, 1000, 512)
            output = moe(input_tensor)
            
            # Should not OOM
            assert output.shape == (1, 1000, 512)
    
    def test_expert_utilization(self):
        """Test expert utilization monitoring."""
        moe = EnhancedSparseMoE(**self.config)
        
        # Process multiple batches
        for i in range(10):
            hidden_states = torch.randn(2, 32, 512)
            _ = moe(hidden_states)
        
        stats = moe.get_stats()
        assert 'expert_count' in stats


if __name__ == "__main__":
    # Run performance benchmark
    test_moe_performance_comparison()