# Triton Kernel CPU Testing Guide

This directory contains comprehensive tests for running Triton kernels on CPU-only environments using Triton's interpreter mode.

## Overview

The tests allow you to verify the correctness of Triton kernels without requiring GPU hardware by using:
- Triton's interpreter mode (`TRITON_INTERPRET=1`)
- CPU fallback mechanisms
- Mock CUDA environments

## Files

- `test_triton_kernels_cpu.py` - Main test file with comprehensive CPU tests
- `run_triton_cpu_tests.py` - Standalone test runner
- `TRITON_CPU_TESTING.md` - This documentation

## Quick Start

### Method 1: Using the standalone runner
```bash
# Make the runner executable
chmod +x tests/run_triton_cpu_tests.py

# Run all tests
python tests/run_triton_cpu_tests.py
```

### Method 2: Using pytest
```bash
# Install pytest if not already installed
pip install pytest

# Run with pytest
pytest tests/test_triton_kernels_cpu.py -v

# Run specific test class
pytest tests/test_triton_kernels_cpu.py::TestMoEKernelsCPU -v

# Run specific test
pytest tests/test_triton_kernels_cpu.py::TestMoEKernelsCPU::test_token_permutation_v1_cpu -v
```

### Method 3: Manual testing
```bash
# Set environment variables
export TRITON_INTERPRET=1
export CUDA_VISIBLE_DEVICES=""

# Run Python interactively
python -c "
import os
os.environ['TRITON_INTERPRET'] = '1'
from tests.test_triton_kernels_cpu import TestMoEKernelsCPU
TestMoEKernelsCPU().test_token_permutation_v1_cpu()
"
```

## Test Coverage

### MoE Kernels
- **Token Permutation (v1)**: Basic expert routing
- **Token Permutation (v2)**: Enhanced with multiple experts per token
- **Expert Gating**: Top-k expert selection with softmax
- **Load Balancing**: Expert assignment across devices
- **Segmented GEMM**: Expert-specific matrix multiplication
- **Gradient Computation**: Backward pass for expert networks
- **Fused Activation**: SiLU activation for expert computation

### Attention Kernels
- **KV Cache Storage**: Key-value cache management
- **Attention Masking**: Expert-specific attention patterns

### Edge Cases
- Empty inputs
- Single tokens
- Large hidden dimensions
- Zero expert counts
- Boundary conditions

## Environment Setup

### Requirements
```bash
# Basic requirements
pip install torch triton pytest

# Optional: Install triton-cpu for better CPU support
pip install triton-cpu
```

### Environment Variables
```bash
# Enable Triton interpreter mode (required)
export TRITON_INTERPRET=1

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Optional: Enable debug mode
export TRITON_DEBUG=1
```

## Test Structure

Each test class focuses on a specific aspect:

- `TestMoEKernelsCPU`: Core MoE functionality
- `TestAttentionKernelsCPU`: Attention-related kernels
- `TestKernelEdgeCasesCPU`: Boundary conditions and edge cases

## Debugging Tips

### Common Issues

1. **ImportError: No module named 'triton'**
   ```bash
   pip install triton
   ```

2. **CUDA not available errors**
   Ensure `CUDA_VISIBLE_DEVICES=""` is set

3. **Triton kernel compilation errors**
   Check `TRITON_INTERPRET=1` is set

### Debug Mode
```bash
# Enable verbose logging
export TRITON_DEBUG=1
export PYTHONPATH=/path/to/your/project:$PYTHONPATH

# Run tests with debug info
python -m pytest tests/test_triton_kernels_cpu.py -v -s
```

## Performance Notes

- CPU testing is significantly slower than GPU
- Interpreter mode has overhead compared to native execution
- Large tensor sizes may cause memory issues on CPU
- Tests use smaller dimensions for faster execution

## Extending Tests

### Adding New Kernel Tests

1. Create test method in appropriate test class
2. Follow the pattern:
   ```python
   def test_new_kernel_cpu(self):
       print("Testing new_kernel on CPU...")
       
       # Setup test data
       input_tensor = torch.randn(...)
       
       # Call kernel
       result = new_kernel(input_tensor, ...)
       
       # Verify results
       assert result.shape == expected_shape
       assert torch.allclose(result, expected_result)
       
       print("✓ new_kernel CPU test passed")
   ```

3. Add to appropriate test class

### Mocking GPU Functions

For kernels that require GPU-specific features:

```python
from unittest.mock import patch

@patch('torch.cuda.is_available', return_value=False)
def test_gpu_kernel_cpu_fallback(self, mock_cuda):
    # Test will run even without GPU
    result = gpu_kernel_cpu_version(...)
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# .github/workflows/triton-cpu-tests.yml
name: Triton CPU Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        pip install torch triton pytest
    - name: Run CPU tests
      run: |
        TRITON_INTERPRET=1 python -m pytest tests/test_triton_kernels_cpu.py -v
```

## Troubleshooting

### Test Failures

1. **Check Triton version compatibility**
   ```bash
   python -c "import triton; print(triton.__version__)"
   ```

2. **Verify interpreter mode**
   ```bash
   python -c "import os; print(os.environ.get('TRITON_INTERPRET'))"
   ```

3. **Check for CUDA interference**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Getting Help

- Check Triton documentation: https://triton-lang.org/
- Review test logs for specific error messages
- Enable debug mode for detailed output
- Test with minimal examples first

## Performance Benchmarking

For performance testing on CPU:

```bash
# Install additional tools
pip install memory_profiler

# Run with profiling
python -m memory_profiler tests/run_triton_cpu_tests.py
```

## Next Steps

1. **Integration Testing**: Combine with actual model testing
2. **Performance Profiling**: Add timing and memory usage tests
3. **Validation**: Compare CPU results with GPU results
4. **CI/CD**: Add to continuous integration pipeline