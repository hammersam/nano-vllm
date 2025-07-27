#!/usr/bin/env python3
"""
Basic Triton kernel CPU tests using only built-in Python
"""

import random
import math

def softmax(x):
    """Compute softmax"""
    exp_x = [math.exp(val - max(x)) for val in x]
    sum_exp = sum(exp_x)
    return [val / sum_exp for val in exp_x]

def test_token_permutation():
    """Test token permutation logic"""
    print("Testing token permutation...")
    
    # Test data
    num_tokens = 8
    hidden_size = 4
    num_experts = 3
    
    # Create test data
    x = [[random.random() for _ in range(hidden_size)] for _ in range(num_tokens)]
    expert_indices = [random.randint(0, num_experts-1) for _ in range(num_tokens)]
    
    # Manual permutation
    expert_counts = [0] * num_experts
    for expert_id in expert_indices:
        expert_counts[expert_id] += 1
    
    # Calculate offsets
    expert_offsets = [0] * num_experts
    for i in range(1, num_experts):
        expert_offsets[i] = expert_offsets[i-1] + expert_counts[i-1]
    
    # Create permuted result
    permuted_x = [[0.0] * hidden_size for _ in range(num_tokens)]
    current_pos = expert_offsets[:]
    
    for token_idx, expert_id in enumerate(expert_indices):
        dest_idx = current_pos[expert_id]
        permuted_x[dest_idx] = x[token_idx][:]
        current_pos[expert_id] += 1
    
    # Verify
    assert len(permuted_x) == len(x)
    original_sum = sum(sum(row) for row in x)
    permuted_sum = sum(sum(row) for row in permuted_x)
    assert abs(original_sum - permuted_sum) < 1e-6
    
    # Test inverse
    recovered_x = [[0.0] * hidden_size for _ in range(num_tokens)]
    current_pos = expert_offsets[:]
    
    for token_idx, expert_id in enumerate(expert_indices):
        src_idx = current_pos[expert_id]
        recovered_x[token_idx] = permuted_x[src_idx][:]
        current_pos[expert_id] += 1
    
    # Verify recovery
    for i in range(num_tokens):
        for j in range(hidden_size):
            assert abs(recovered_x[i][j] - x[i][j]) < 1e-6
    
    print("PASS: token permutation")

def test_expert_gating():
    """Test expert gating logic"""
    print("Testing expert gating...")
    
    num_tokens = 4
    num_experts = 3
    top_k = 2
    
    # Create test data
    logits = [[random.random() * 10 - 5 for _ in range(num_experts)] for _ in range(num_tokens)]
    
    # Apply softmax to each token
    probs = [softmax(logit_row) for logit_row in logits]
    
    # Top-k selection
    topk_values = []
    topk_indices = []
    
    for prob_row in probs:
        # Create list of (value, index) pairs
        indexed_probs = [(val, idx) for idx, val in enumerate(prob_row)]
        # Sort by value descending
        indexed_probs.sort(key=lambda x: x[0], reverse=True)
        # Take top-k
        topk_values.append([val for val, _ in indexed_probs[:top_k]])
        topk_indices.append([idx for _, idx in indexed_probs[:top_k]])
    
    # Verify
    assert len(topk_values) == num_tokens
    assert len(topk_indices) == num_tokens
    for values, indices in zip(topk_values, topk_indices):
        assert len(values) == top_k
        assert len(indices) == top_k
        for val in values:
            assert 0 <= val <= 1
        for idx in indices:
            assert 0 <= idx < num_experts
    
    print("PASS: expert gating")

def test_segmented_gemm():
    """Test segmented GEMM logic"""
    print("Testing segmented GEMM...")
    
    num_tokens = 6
    hidden_size = 4
    intermediate_size = 8
    num_experts = 2
    
    # Create test data
    a = [[random.random() for _ in range(hidden_size)] for _ in range(num_tokens)]
    b = [[[random.random() for _ in range(intermediate_size)] for _ in range(hidden_size)] for _ in range(num_experts)]
    c = [[0.0] * intermediate_size for _ in range(num_tokens)]
    
    # Expert assignment
    expert_assignment = [0, 0, 0, 1, 1, 1]
    
    # Manual segmented GEMM
    for expert_id in range(num_experts):
        mask = [idx for idx, val in enumerate(expert_assignment) if val == expert_id]
        tokens_for_expert = [a[idx] for idx in mask]
        
        if tokens_for_expert:
            # Matrix multiplication
            for token_idx, token in enumerate(tokens_for_expert):
                result = [0.0] * intermediate_size
                for j in range(intermediate_size):
                    for k in range(hidden_size):
                        result[j] += token[k] * b[expert_id][k][j]
                
                # Place results
                original_idx = mask[token_idx]
                c[original_idx] = result
    
    assert len(c) == num_tokens
    assert len(c[0]) == intermediate_size
    print("PASS: segmented GEMM")

def test_load_balancing():
    """Test load balancing logic"""
    print("Testing load balancing...")
    
    num_experts = 6
    world_size = 3
    
    # Create test loads
    expert_loads = [random.random() for _ in range(num_experts)]
    
    # Manual load balancing
    assignments = [0] * num_experts
    device_loads = [0.0] * world_size
    
    # Sort by load descending
    expert_order = sorted(range(num_experts), key=lambda i: expert_loads[i], reverse=True)
    
    for expert_id in expert_order:
        least_loaded_device = device_loads.index(min(device_loads))
        assignments[expert_id] = least_loaded_device
        device_loads[least_loaded_device] += expert_loads[expert_id]
    
    assert len(assignments) == num_experts
    assert all(0 <= val < world_size for val in assignments)
    print("PASS: load balancing")

def run_all_tests():
    """Run all tests"""
    print("Running Triton Kernel CPU Logic Tests")
    print("=" * 40)
    
    try:
        test_token_permutation()
        test_expert_gating()
        test_segmented_gemm()
        test_load_balancing()
        
        print("=" * 40)
        print("SUCCESS: All tests passed!")
        return True
        
    except Exception as e:
        print("FAILED: {}".format(str(e)))
        return False

if __name__ == "__main__":
    run_all_tests()