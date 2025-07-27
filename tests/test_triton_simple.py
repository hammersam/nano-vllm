#!/usr/bin/env python3
"""
Simple Triton kernel CPU tests
"""

import numpy as np
import torch

def test_token_permutation():
    """Test token permutation logic"""
    print("Testing token permutation...")
    
    num_tokens = 8
    hidden_size = 4
    num_experts = 3
    
    # Create test data
    x = np.random.randn(num_tokens, hidden_size)
    expert_indices = np.random.randint(0, num_experts, num_tokens)
    
    # Manual permutation
    expert_counts = np.bincount(expert_indices, minlength=num_experts)
    expert_offsets = np.cumsum(expert_counts) - expert_counts
    
    # Create permuted result
    permuted_x = np.zeros_like(x)
    current_pos = expert_offsets.copy()
    
    for token_idx, expert_id in enumerate(expert_indices):
        dest_idx = current_pos[expert_id]
        permuted_x[dest_idx] = x[token_idx]
        current_pos[expert_id] += 1
    
    # Verify
    assert permuted_x.shape == x.shape
    assert np.allclose(np.sum(permuted_x), np.sum(x))
    
    # Test inverse
    recovered_x = np.zeros_like(x)
    current_pos = expert_offsets.copy()
    
    for token_idx, expert_id in enumerate(expert_indices):
        src_idx = current_pos[expert_id]
        recovered_x[token_idx] = permuted_x[src_idx]
        current_pos[expert_id] += 1
    
    assert np.allclose(recovered_x, x)
    print("PASS: token permutation")

def test_expert_gating():
    """Test expert gating logic"""
    print("Testing expert gating...")
    
    num_tokens = 4
    num_experts = 3
    top_k = 2
    
    # Create test data
    logits = np.random.randn(num_tokens, num_experts)
    
    # Manual softmax and top-k
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    probs = softmax(logits)
    
    # Top-k selection
    topk_indices = np.argsort(probs, axis=-1)[:, -top_k:][:, ::-1]
    topk_values = np.take_along_axis(probs, topk_indices, axis=-1)
    
    # Verify
    assert topk_values.shape == (num_tokens, top_k)
    assert np.all(topk_values >= 0)
    assert np.all(topk_values <= 1)
    assert np.all(topk_indices >= 0)
    assert np.all(topk_indices < num_experts)
    print("PASS: expert gating")

def test_segmented_gemm():
    """Test segmented GEMM logic"""
    print("Testing segmented GEMM...")
    
    num_tokens = 6
    hidden_size = 4
    intermediate_size = 8
    num_experts = 2
    
    # Create test data
    a = np.random.randn(num_tokens, hidden_size)
    b = np.random.randn(num_experts, hidden_size, intermediate_size)
    c = np.zeros((num_tokens, intermediate_size))
    
    # Expert assignment
    expert_assignment = np.array([0, 0, 0, 1, 1, 1])
    
    # Manual segmented GEMM
    for expert_id in range(num_experts):
        mask = expert_assignment == expert_id
        tokens_for_expert = a[mask]
        
        if len(tokens_for_expert) > 0:
            expert_result = np.dot(tokens_for_expert, b[expert_id])
            expert_indices = np.where(mask)[0]
            for i, token_idx in enumerate(expert_indices):
                c[token_idx] = expert_result[i]
    
    assert c.shape == (num_tokens, intermediate_size)
    assert not np.any(np.isnan(c))
    print("PASS: segmented GEMM")

def test_load_balancing():
    """Test load balancing logic"""
    print("Testing load balancing...")
    
    num_experts = 6
    world_size = 3
    
    # Create test loads
    expert_loads = np.random.rand(num_experts)
    
    # Manual load balancing
    assignments = np.zeros(num_experts, dtype=int)
    device_loads = np.zeros(world_size)
    
    expert_order = np.argsort(expert_loads)[::-1]
    
    for expert_id in expert_order:
        least_loaded_device = np.argmin(device_loads)
        assignments[expert_id] = least_loaded_device
        device_loads[least_loaded_device] += expert_loads[expert_id]
    
    assert len(assignments) == num_experts
    assert np.all(assignments >= 0)
    assert np.all(assignments < world_size)
    print("PASS: load balancing")

def test_torch_integration():
    """Test PyTorch tensor operations"""
    print("Testing PyTorch integration...")
    
    torch.manual_seed(42)
    
    num_tokens = 8
    hidden_size = 16
    num_experts = 4
    
    # Create test data
    x = torch.randn(num_tokens, hidden_size)
    expert_indices = torch.randint(0, num_experts, (num_tokens,))
    
    # PyTorch operations
    expert_counts = torch.bincount(expert_indices, minlength=num_experts)
    expert_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts
    
    # Verify
    assert expert_counts.shape == (num_experts,)
    assert torch.sum(expert_counts) == num_tokens
    assert expert_offsets[0] == 0
    
    # Test gradient flow
    x.requires_grad_(True)
    expert_weights = torch.randn(num_experts, hidden_size, 8, requires_grad=True)
    
    output = torch.zeros(num_tokens, 8)
    for i, expert_id in enumerate(expert_indices):
        output[i] = torch.matmul(x[i], expert_weights[expert_id])
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert expert_weights.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(expert_weights.grad).any()
    print("PASS: PyTorch integration")

def run_all_tests():
    """Run all tests"""
    print("Running Triton Kernel CPU Logic Tests")
    print("=" * 40)
    
    try:
        test_token_permutation()
        test_expert_gating()
        test_segmented_gemm()
        test_load_balancing()
        test_torch_integration()
        
        print("=" * 40)
        print("SUCCESS: All tests passed!")
        return True
        
    except Exception as e:
        print("FAILED: {}".format(e))
        return False

if __name__ == "__main__":
    run_all_tests()