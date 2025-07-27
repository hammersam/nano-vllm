
import pytest
import torch
from nanovllm.layers.moe_kernel import moe_align_block_size, moe_compute_top_k_gating, moe_dispatch_expert_tokens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_kernels(dtype):
    # This is a placeholder test and will likely fail without a proper environment
    try:
        # moe_align_block_size
        top_k_ids = torch.randint(0, 8, (10,), device="cuda")
        aligned_top_k_ids = moe_align_block_size(top_k_ids, 4, 8)
        assert aligned_top_k_ids.shape[0] % 4 == 0

        # moe_compute_top_k_gating
        gating_output = torch.randn(10, 8, dtype=dtype, device="cuda")
        top_k_values, top_k_ids = moe_compute_top_k_gating(gating_output, 2)
        assert top_k_values.shape == (10, 2)
        assert top_k_ids.shape == (10, 2)

        # moe_dispatch_expert_tokens
        # This kernel is more complex and requires more setup
        # This is a very basic test
        tokens = torch.randn(10, 64, dtype=dtype, device="cuda")
        top_k_ids = torch.randint(0, 8, (10, 2), device="cuda")
        sorted_expert_ids = torch.sort(top_k_ids.flatten()).values
        dispatched_tokens = moe_dispatch_expert_tokens(tokens, top_k_ids, sorted_expert_ids)
        assert dispatched_tokens.shape == tokens.shape

    except ImportError:
        pytest.skip("Triton not available")
    except Exception as e:
        pytest.fail(f"Moe kernel tests failed with: {e}")
