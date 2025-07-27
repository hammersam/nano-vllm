
import pytest
import torch
from nanovllm.layers.sampler import Sampler


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sampler(dtype):
    # Create a sample vocabulary size and logits tensor
    vocab_size = 100
    logits = torch.randn(2, vocab_size, dtype=dtype, device="cuda")

    # Create the Sampler module
    sampler = Sampler(vocab_size)

    # Create sample temperature and top_p tensors
    temperature = torch.tensor([0.7, 0.9], dtype=dtype, device="cuda")
    top_p = torch.tensor([0.9, 0.8], dtype=dtype, device="cuda")

    # Forward pass
    next_tokens = sampler(logits, temperature, top_p)

    # Check the output shape
    assert next_tokens.shape == (2,)

    # Check the output dtype
    assert next_tokens.dtype == torch.int64

    # Check that the output tokens are within the vocabulary size
    assert torch.all(next_tokens >= 0)
    assert torch.all(next_tokens < vocab_size)
