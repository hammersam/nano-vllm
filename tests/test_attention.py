
import pytest
import torch
from nanovllm.layers.attention import Attention, store_kvcache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_store_kvcache(dtype):
    N, num_heads, head_dim = 2, 4, 64
    D = num_heads * head_dim
    key = torch.randn(N, num_heads, head_dim, dtype=dtype, device="cuda")
    value = torch.randn(N, num_heads, head_dim, dtype=dtype, device="cuda")
    k_cache = torch.zeros(N, D, dtype=dtype, device="cuda")
    v_cache = torch.zeros(N, D, dtype=dtype, device="cuda")
    slot_mapping = torch.arange(N, dtype=torch.int32, device="cuda")

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    assert torch.allclose(k_cache.view(N, num_heads, head_dim), key)
    assert torch.allclose(v_cache.view(N, num_heads, head_dim), value)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention(dtype):
    num_heads, head_dim, num_kv_heads = 4, 64, 2
    scale = 1.0
    attn = Attention(num_heads, head_dim, scale, num_kv_heads).to(dtype).to("cuda")

    q = torch.randn(2, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(2, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(2, num_kv_heads, head_dim, dtype=dtype, device="cuda")

    # This will likely fail if triton is not available
    try:
        output = attn(q, k, v)
        assert output.shape == (2, num_heads * head_dim)
    except ImportError:
        pytest.skip("Triton not available")
