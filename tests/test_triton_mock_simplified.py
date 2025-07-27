#!/usr/bin/env python3
"""
Simplified Triton kernel CPU tests
Tests kernel logic without actual Triton dependencies
"""

import numpy as np
import torch
import unittest

class TestMoEKernelsCPU(unittest.TestCase):
    """Test MoE kernel logic on CPU"""
    
    def test_token_permutation_logic(self):
        """Test token permutation logic"""
        print("Testing token permutation logic...")
        
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
        self.assertEqual(permuted_x.shape, x.shape)
        self.assertTrue(np.allclose(np.sum(permuted_x), np.sum(x)))
        
        # Test inverse
        recovered_x = np.zeros_like(x)
        current_pos = expert_offsets.copy()
        
        for token_idx, expert_id in enumerate(expert_indices):
            src_idx = current_pos[expert_id]
            recovered_x[token_idx] = permuted_x[src_idx]
            current_pos[expert_id] += 1
        
        self.assertTrue(np.allclose(recovered_x, x))
        print("✓ Token permutation test passed")
    
    def test_expert_gating_logic(self):
        """Test expert gating logic"""
        print("Testing expert gating logic...")
        
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
        self.assertEqual(topk_values.shape, (num_tokens, top_k))
        self.assertTrue(np.all(topk_values >= 0))
        self.assertTrue(np.all(topk_values <= 1))
        self.assertTrue(np.all(topk_indices >= 0))
        self.assertTrue(np.all(topk_indices < num_experts))
        print("✓ Expert gating test passed")
    
    def test_segmented_gemm_logic(self):
        """Test segmented GEMM logic"""
        print("Testing segmented GEMM logic...")
        
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
        
        self.assertEqual(c.shape, (num_tokens, intermediate_size))
        self.assertFalse(np.any(np.isnan(c)))
        print("✓ Segmented GEMM test passed")
    
    def test_load_balancing_logic(self):
        """Test load balancing logic"""
        print("Testing load balancing logic...")
        
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
        
        self.assertEqual(len(assignments), num_experts)
        self.assertTrue(np.all(assignments >= 0))
        self.assertTrue(np.all(assignments < world_size))
        print("✓ Load balancing test passed")
    
    def test_torch_integration(self):
        """Test PyTorch tensor operations"""
        print("Testing PyTorch integration...")
        
        torch.manual_seed(42)
        
        # Test parameters
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
        self.assertEqual(expert_counts.shape, (num_experts,))
        self.assertEqual(torch.sum(expert_counts), num_tokens)
        self.assertEqual(expert_offsets[0], 0)
        
        # Test gradient flow
        x.requires_grad_(True)
        expert_weights = torch.randn(num_experts, hidden_size, 8, requires_grad=True)
        
        # Forward pass
        output = torch.zeros(num_tokens, 8)
        for i, expert_id in enumerate(expert_indices):
            output[i] = torch.matmul(x[i], expert_weights[expert_id])
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(expert_weights.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(expert_weights.grad).any())
        print("✓ PyTorch integration test passed")
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("Testing edge cases...")
        
        # Empty inputs
        empty_x = np.empty((0, 4))
        empty_indices = np.empty(0, dtype=int)
        self.assertEqual(empty_x.shape[0], 0)
        
        # Single element
        single_x = np.array([[1.0, 2.0, 3.0, 4.0]])
        single_indices = np.array([0])
        expert_counts = np.bincount(single_indices, minlength=1)
        self.assertEqual(expert_counts[0], 1)
        
        # Large dimensions
        large_x = np.random.randn(100, 64)
        large_indices = np.random.randint(0, 8, 100)
        expert_counts = np.bincount(large_indices, minlength=8)
        self.assertEqual(len(expert_counts), 8)
        self.assertEqual(np.sum(expert_counts), 100)
        print("✓ Edge cases test passed")


def run_all_tests():
    """Run all tests"""
    print("🧪 Running Triton Kernel CPU Logic Tests")
    print("=" * 50)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoEKernelsCPU)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()