# Triton Kernel CPU Testing Suite

This directory contains comprehensive tests for verifying Triton kernel correctness on CPU-only environments.

## Quick Start

```bash
# Run basic tests (no dependencies)
python tests/test_triton_basic.py

# Run with pytest (if available)
pytest tests/test_triton_kernels_cpu.py -v
```

## Test Files Overview

### 1. Basic Tests (No Dependencies)
- **`test_triton_basic.py`** - Pure Python tests for kernel logic
  - ✅ No external dependencies
  - ✅ Tests core algorithm correctness
  - ✅ Fast execution

### 2. Comprehensive Tests (Requires PyTorch)
- **`test_triton_kernels_cpu.py`** - Full Triton CPU tests
  - ✅ Tests all MoE kernels
  - ✅ Tests attention kernels
  - ✅ Edge case testing
  - ✅ Uses Triton interpreter mode

### 3. Mock Tests (Fallback)
- **`test_triton_mock_cpu.py`** - Mock Triton functionality
  - ✅ No Triton installation required
  - ✅ Tests kernel logic
  - ✅ PyTorch integration

### 4. Test Runner
- **`run_triton_cpu_tests.py`** - Automated test runner
  - ✅ Environment setup
  - ✅ Dependency checking
  - ✅ Multiple test modes

## Test Coverage Matrix

| Kernel Type | Basic | Mock | Full | Description |
|-------------|-------|------|------|-------------|
| Token Permutation | ✅ | ✅ | ✅ | Expert routing |
| Expert Gating | ✅ | ✅ | ✅ | Top-k selection |
| Segmented GEMM | ✅ | ✅ | ✅ | Expert-specific matmul |
| Load Balancing | ✅ | ✅ | ✅ | Expert distribution |
| KV Cache | ❌ | ✅ | ✅ | Attention memory |
| Edge Cases | ✅ | ✅ | ✅ | Boundary conditions |

## Usage Examples

### Method 1: Basic Testing (Recommended for initial setup)
```bash
python tests/test_triton_basic.py
```

### Method 2: Comprehensive Testing (with PyTorch)
```bash
# Set environment
export TRITON_INTERPRET=1
export CUDA_VISIBLE_DEVICES=""

# Run tests
python tests/test_triton_kernels_cpu.py
```

### Method 3: Using Test Runner
```bash
python tests/run_triton_cpu_tests.py
```

### Method 4: Pytest Integration
```bash
pip install pytest
pytest tests/test_triton_kernels_cpu.py -v
```

## Environment Setup

### For Basic Tests
- Python 3.6+
- No additional dependencies

### For Full Tests
```bash
pip install torch
pip install triton  # Optional, for interpreter mode
```

### Environment Variables
```bash
export TRITON_INTERPRET=1      # Enable interpreter mode
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

## Test Results Summary

Based on current analysis, the project has:

- **3 Triton kernel files** with 8+ kernel functions
- **Multiple MoE operations** (permutation, gating, GEMM)
- **Attention kernels** (KV cache, masking)
- **Edge case handling** (empty inputs, single tokens)

## Quick Validation

Run this to verify your setup:

```bash
# Test basic functionality
python tests/test_triton_basic.py

# Expected output:
# Running Triton Kernel CPU Logic Tests
# ========================================
# Testing token permutation...
# PASS: token permutation
# Testing expert gating...
# PASS: expert gating
# Testing segmented GEMM...
# PASS: segmented GEMM
# Testing load balancing...
# PASS: load balancing
# ========================================
# SUCCESS: All tests passed!
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named torch**
   - Use `test_triton_basic.py` for pure Python testing
   - Install PyTorch: `pip install torch`

2. **Triton not available**
   - Use mock tests or basic tests
   - Install Triton: `pip install triton`

3. **CUDA errors**
   - Set `CUDA_VISIBLE_DEVICES=""`
   - Use `TRITON_INTERPRET=1`

### Debug Mode
```bash
# Enable debug logging
export TRITON_DEBUG=1
python tests/test_triton_kernels_cpu.py -v
```

## Adding New Tests

1. **For basic logic**: Add to `test_triton_basic.py`
2. **For Triton kernels**: Add to `test_triton_kernels_cpu.py`
3. **For mock testing**: Use `test_triton_mock_cpu.py`

## Performance Notes

- **Basic tests**: ~1ms per test
- **Mock tests**: ~10ms per test  
- **Full Triton tests**: ~100ms per test (interpreter mode)
- **Recommended**: Start with `test_triton_basic.py` for rapid iteration

## CI/CD Integration

For GitHub Actions:

```yaml
name: Triton CPU Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    - run: python tests/test_triton_basic.py
    - run: pip install torch
    - run: python tests/test_triton_kernels_cpu.py
```