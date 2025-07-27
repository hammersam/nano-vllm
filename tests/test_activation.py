
import pytest
import torch
from nanovllm.layers.activation import SiluAndMul


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_silu_and_mul(dtype):
    # Create a sample input tensor
    x = torch.randn(2, 10, dtype=dtype, device="cuda")

    # Create the SiluAndMul module
    silu_and_mul = SiluAndMul()

    # Forward pass
    output = silu_and_mul(x)

    # Check the output shape
    assert output.shape == (2, 5)

    # Check the output dtype
    assert output.dtype == dtype

    # Check the output values
    gate, up = x.chunk(2, dim=-1)
    expected_output = torch.nn.functional.silu(gate) * up
    assert torch.allclose(output, expected_output, atol=1e-3)
