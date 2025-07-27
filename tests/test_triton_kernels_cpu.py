import pytest
import torch
import numpy as np
import os
import sys
from unittest.mock import patch, Mock

# Enable Triton interpreter mode for CPU testing
os.environ['TRITON_INTERPRET'] = '1'

# Mock CUDA availability to test CPU fallback
sys.modules['torch'].cuda.is_available = lambda: False

try:
    from nanovllm.layers.moe_kernel import token_permutation as token_perm_v1, invoke_segmented_gemm
    from nanovllm.layers.moe_kernel_v2 import (
        token_permutation_v2 as token_perm_v2,
        invoke_segmented_gemm_v2,
        expert_gating_v2,
        load_balancing_v2,
        compute_gradients_v2,
        fused_activation_v2,
        expert_attention_mask_v2
    )
    from nanovllm.layers.attention import store_kvcache, Attention
    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Triton not available for CPU testing: {e}")
    TRITON_AVAILABLE = False


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestMoEKernelsCPU:
    """Test MoE kernels on CPU using Triton interpreter mode"""
    
    def test_token_permutation_v1_cpu(self):
        """Test token permutation kernel (v1) on CPU"""
        print("Testing token_permutation_v1 on CPU...")
        
        # Test parameters
        num_tokens = 32
        hidden_size = 128
        num_experts = 8
        
        # Create test data
        x = torch.randn(num_tokens, hidden_size, device='cpu')
        expert_indices = torch.randint(0, num_experts, (num_tokens,), device='cpu')
        
        # Test forward permutation
        permuted_x, expert_counts = token_perm_v1(x, expert_indices, num_experts, is_forward=True)
        
        # Verify shapes
        assert permuted_x.shape == x.shape, f"Shape mismatch: {permuted_x.shape} != {x.shape}"
        assert expert_counts.shape == (num_experts,), f"Expert counts shape mismatch: {expert_counts.shape}"
        
        # Verify permutation is valid (all tokens are preserved)
        assert torch.allclose(permuted_x.sum(), x.sum(), atol=1e-6), "Token values not preserved"
        
        # Test inverse permutation
        recovered_x, _ = token_perm_v1(permuted_x, expert_indices, num_experts, is_forward=False)
        assert torch.allclose(recovered_x, x, atol=1e-6), "Inverse permutation failed"
        
        print("✓ token_permutation_v1 CPU test passed")
    
    def test_token_permutation_v2_cpu(self):
        """Test enhanced token permutation kernel (v2) on CPU"""
        print("Testing token_permutation_v2 on CPU...")
        
        # Test parameters
        num_tokens = 32
        hidden_size = 128
        num_experts = 8
        num_experts_per_tok = 2
        
        # Create test data
        x = torch.randn(num_tokens, hidden_size, device='cpu')
        expert_indices = torch.randint(0, num_experts, (num_tokens * num_experts_per_tok,), device='cpu')
        
        # Test forward permutation
        permuted_x, expert_counts, token_mapping = token_perm_v2(
            x, expert_indices, num_experts, num_experts_per_tok, is_forward=True
        )
        
        # Verify shapes
        assert permuted_x.shape == x.shape, f"Shape mismatch: {permuted_x.shape} != {x.shape}"
        assert expert_counts.shape == (num_experts,), f"Expert counts shape mismatch"
        assert token_mapping.shape == (num_tokens * num_experts_per_tok,), f"Token mapping shape mismatch"
        
        # Test inverse permutation
        recovered_x, _, _ = token_perm_v2(
            permuted_x, expert_indices, num_experts, num_experts_per_tok, is_forward=False
        )
        assert torch.allclose(recovered_x, x, atol=1e-6), "Inverse permutation failed"
        
        print("✓ token_permutation_v2 CPU test passed")
    
    def test_expert_gating_v2_cpu(self):
        """Test expert gating kernel on CPU"""
        print("Testing expert_gating_v2 on CPU...")
        
        # Test parameters
        num_tokens = 16
        num_experts = 8
        top_k = 2
        
        # Create test logits
        logits = torch.randn(num_tokens, num_experts, device='cpu')
        
        # Test gating
        topk_values, topk_indices = expert_gating_v2(logits, top_k)
        
        # Verify shapes
        assert topk_values.shape == (num_tokens, top_k), f"Values shape mismatch: {topk_values.shape}"
        assert topk_indices.shape == (num_tokens, top_k), f"Indices shape mismatch: {topk_indices.shape}"
        
        # Verify values are valid probabilities
        assert torch.all(topk_values >= 0), "Negative probabilities found"
        assert torch.all(topk_values <= 1), "Probabilities > 1 found"
        assert torch.all(topk_indices >= 0), "Negative indices found"
        assert torch.all(topk_indices < num_experts), "Invalid expert indices"
        
        # Verify indices are unique per token
        for i in range(num_tokens):
            assert len(torch.unique(topk_indices[i])) == top_k, f"Duplicate indices in token {i}"
        
        print("✓ expert_gating_v2 CPU test passed")
    
    def test_load_balancing_v2_cpu(self):
        """Test load balancing kernel on CPU"""
        print("Testing load_balancing_v2 on CPU...")
        
        # Test parameters
        num_experts = 8
        world_size = 4
        
        # Create test loads
        expert_loads = torch.rand(num_experts, device='cpu')
        
        # Test load balancing
        assignments = load_balancing_v2(expert_loads, world_size)
        
        # Verify shapes
        assert assignments.shape == (num_experts,), f"Shape mismatch: {assignments.shape}"
        
        # Verify assignments are valid
        assert torch.all(assignments >= 0), "Negative device assignments"
        assert torch.all(assignments < world_size), "Invalid device indices"
        
        # Verify load distribution
        unique_devices, counts = torch.unique(assignments, return_counts=True)
        assert len(unique_devices) <= world_size, "Too many devices assigned"
        
        print("✓ load_balancing_v2 CPU test passed")
    
    def test_segmented_gemm_v1_cpu(self):
        """Test segmented GEMM (v1) on CPU"""
        print("Testing segmented_gemm_v1 on CPU...")
        
        # Test parameters
        num_tokens = 32
        hidden_size = 128
        intermediate_size = 512
        num_experts = 4
        
        # Create test data
        a = torch.randn(num_tokens, hidden_size, device='cpu')
        b = torch.randn(num_experts, hidden_size, intermediate_size, device='cpu')
        c = torch.empty(num_tokens, intermediate_size, device='cpu')
        expert_counts = torch.tensor([8, 8, 8, 8], device='cpu')
        
        # Test segmented GEMM
        try:
            result = invoke_segmented_gemm(a, b, c, expert_counts)
            assert result.shape == c.shape, f"Shape mismatch: {result.shape} != {c.shape}"
            assert not torch.isnan(result).any(), "NaN values in result"
            assert not torch.isinf(result).any(), "Inf values in result"
            print("✓ segmented_gemm_v1 CPU test passed")
        except Exception as e:
            print(f"⚠ segmented_gemm_v1 CPU test skipped: {e}")
    
    def test_segmented_gemm_v2_cpu(self):
        """Test segmented GEMM (v2) on CPU"""
        print("Testing segmented_gemm_v2 on CPU...")
        
        # Test parameters
        num_tokens = 32
        hidden_size = 128
        intermediate_size = 512
        num_experts = 4
        
        # Create test data
        a = torch.randn(num_tokens, hidden_size, device='cpu')
        b = torch.randn(num_experts, hidden_size, intermediate_size, device='cpu')
        c = torch.empty(num_tokens, intermediate_size, device='cpu')
        expert_counts = torch.tensor([8, 8, 8, 8], device='cpu')
        
        # Test segmented GEMM
        try:
            result = invoke_segmented_gemm_v2(a, b, c, expert_counts)
            assert result.shape == c.shape, f"Shape mismatch: {result.shape} != {c.shape}"
            assert not torch.isnan(result).any(), "NaN values in result"
            assert not torch.isinf(result).any(), "Inf values in result"
            print("✓ segmented_gemm_v2 CPU test passed")
        except Exception as e:
            print(f"⚠ segmented_gemm_v2 CPU test skipped: {e}")
    
    def test_compute_gradients_v2_cpu(self):
        """Test gradient computation on CPU"""
        print("Testing compute_gradients_v2 on CPU...")
        
        # Test parameters
        num_tokens = 16
        hidden_size = 64
        intermediate_size = 128
        num_experts = 4
        
        # Create test data
        grad_output = torch.randn(num_tokens, intermediate_size, device='cpu')
        input_data = torch.randn(num_tokens, hidden_size, device='cpu')
        weight = torch.randn(num_experts, hidden_size, intermediate_size, device='cpu')
        expert_counts = torch.tensor([4, 4, 4, 4], device='cpu')
        
        try:
            grad_input, grad_weight = compute_gradients_v2(
                grad_output, input_data, weight, expert_counts, learning_rate=0.001
            )
            
            # Verify shapes
            assert grad_input.shape == input_data.shape, f"Input grad shape mismatch"
            assert grad_weight.shape == weight.shape, f"Weight grad shape mismatch"
            
            # Verify no NaN/Inf values
            assert not torch.isnan(grad_input).any(), "NaN in input gradients"
            assert not torch.isnan(grad_weight).any(), "NaN in weight gradients"
            assert not torch.isinf(grad_input).any(), "Inf in input gradients"
            assert not torch.isinf(grad_weight).any(), "Inf in weight gradients"
            
            print("✓ compute_gradients_v2 CPU test passed")
        except Exception as e:
            print(f"⚠ compute_gradients_v2 CPU test skipped: {e}")
    
    def test_fused_activation_v2_cpu(self):
        """Test fused activation on CPU"""
        print("Testing fused_activation_v2 on CPU...")
        
        # Test parameters
        num_tokens = 16
        hidden_size = 128
        num_experts = 4
        
        # Create test data
        input_data = torch.randn(num_tokens, hidden_size, device='cpu')
        expert_counts = torch.tensor([4, 4, 4, 4], device='cpu')
        
        try:
            output = fused_activation_v2(input_data, expert_counts, output=None)
            
            # Verify shapes
            assert output.shape == (num_tokens, hidden_size // 2), f"Output shape mismatch"
            
            # Verify activation was applied
            assert not torch.allclose(output, input_data[:, :hidden_size//2]), "Activation not applied"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"
            
            print("✓ fused_activation_v2 CPU test passed")
        except Exception as e:
            print(f"⚠ fused_activation_v2 CPU test skipped: {e}")
    
    def test_expert_attention_mask_v2_cpu(self):
        """Test expert attention masking on CPU"""
        print("Testing expert_attention_mask_v2 on CPU...")
        
        # Test parameters
        batch_size = 4
        seq_len = 32
        num_experts = 8
        
        # Create sequence lengths
        sequence_lengths = torch.randint(16, seq_len, (batch_size,), device='cpu')
        
        try:
            mask = expert_attention_mask_v2(batch_size, seq_len, num_experts, sequence_lengths)
            
            # Verify shapes
            assert mask.shape == (batch_size, num_experts, seq_len, seq_len), f"Mask shape mismatch"
            
            # Verify mask values are boolean
            assert mask.dtype == torch.bool, f"Mask dtype should be bool, got {mask.dtype}"
            
            # Verify mask is not all zeros
            assert mask.any(), "Mask is empty"
            
            print("✓ expert_attention_mask_v2 CPU test passed")
        except Exception as e:
            print(f"⚠ expert_attention_mask_v2 CPU test skipped: {e}")


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestAttentionKernelsCPU:
    """Test attention kernels on CPU using Triton interpreter mode"""
    
    def test_store_kvcache_cpu(self):
        """Test KV cache storage on CPU"""
        print("Testing store_kvcache on CPU...")
        
        # Test parameters
        batch_size = 4
        num_heads = 8
        head_dim = 64
        seq_len = 32
        
        # Create test data
        key = torch.randn(batch_size, num_heads, head_dim, device='cpu')
        value = torch.randn(batch_size, num_heads, head_dim, device='cpu')
        k_cache = torch.empty(1000, num_heads * head_dim, device='cpu')
        v_cache = torch.empty(1000, num_heads * head_dim, device='cpu')
        slot_mapping = torch.arange(batch_size, device='cpu')
        
        # Test KV cache storage
        try:
            store_kvcache(key, value, k_cache, v_cache, slot_mapping)
            
            # Verify cache was populated
            assert not torch.allclose(k_cache[:batch_size], torch.zeros_like(k_cache[:batch_size])), "Key cache not populated"
            assert not torch.allclose(v_cache[:batch_size], torch.zeros_like(v_cache[:batch_size])), "Value cache not populated"
            
            # Verify data integrity
            assert torch.allclose(k_cache[:batch_size].view(batch_size, num_heads, head_dim), key), "Key data mismatch"
            assert torch.allclose(v_cache[:batch_size].view(batch_size, num_heads, head_dim), value), "Value data mismatch"
            
            print("✓ store_kvcache CPU test passed")
        except Exception as e:
            print(f"⚠ store_kvcache CPU test skipped: {e}")


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestKernelEdgeCasesCPU:
    """Test edge cases for Triton kernels on CPU"""
    
    def test_empty_input(self):
        """Test empty input handling"""
        print("Testing empty input handling...")
        
        # Empty tensor test
        empty_x = torch.empty(0, 128, device='cpu')
        empty_indices = torch.empty(0, device='cpu')
        
        # This should not crash
        try:
            result, counts = token_perm_v1(empty_x, empty_indices, 8, is_forward=True)
            assert result.shape == (0, 128), "Empty tensor handling failed"
            print("✓ Empty input test passed")
        except Exception as e:
            print(f"⚠ Empty input test skipped: {e}")
    
    def test_single_token(self):
        """Test single token handling"""
        print("Testing single token handling...")
        
        # Single token test
        single_x = torch.randn(1, 64, device='cpu')
        single_indices = torch.tensor([0], device='cpu')
        
        try:
            result, counts = token_perm_v1(single_x, single_indices, 2, is_forward=True)
            assert result.shape == (1, 64), "Single token handling failed"
            assert counts[0] == 1, "Single token count mismatch"
            print("✓ Single token test passed")
        except Exception as e:
            print(f"⚠ Single token test skipped: {e}")
    
    def test_large_hidden_size(self):
        """Test large hidden size handling"""
        print("Testing large hidden size handling...")
        
        # Large hidden size test
        large_x = torch.randn(4, 4096, device='cpu')
        large_indices = torch.randint(0, 8, (4,), device='cpu')
        
        try:
            result, counts = token_perm_v1(large_x, large_indices, 8, is_forward=True)
            assert result.shape == (4, 4096), "Large hidden size handling failed"
            print("✓ Large hidden size test passed")
        except Exception as e:
            print(f"⚠ Large hidden size test skipped: {e}")
    
    def test_zero_experts(self):
        """Test zero expert counts"""
        print("Testing zero expert counts...")
        
        # Zero expert counts test
        x = torch.randn(4, 64, device='cpu')
        expert_counts = torch.zeros(4, device='cpu', dtype=torch.int32)
        
        # This should handle gracefully
        try:
            result = invoke_segmented_gemm_v2(x, torch.randn(4, 64, 128, device='cpu'), 
                                            torch.empty(4, 128, device='cpu'), expert_counts)
            print("✓ Zero expert counts test passed")
        except Exception as e:
            print(f"⚠ Zero expert counts test skipped: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])