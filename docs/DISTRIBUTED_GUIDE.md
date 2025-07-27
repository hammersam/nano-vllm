# 🌐 Distributed Serving Guide

This guide explains how to use distributed inference capabilities in nano-vLLM for scaling across multiple GPUs and machines.

## 📋 Overview

The distributed features provide:
- **Tensor Parallelism**: Split model across GPUs
- **Pipeline Parallelism**: Split layers across stages
- **Expert Parallelism** (for MoE models)
- **Load balancing** across workers
- **Fault tolerance** and health monitoring

## 🚀 Quick Start

### Single Machine, Multiple GPUs

```python
from nanovllm.engine.distributed_engine import DistributedEngine
from nanovllm.sampling_params import SamplingParams

# 2 GPUs on single machine
engine = DistributedEngine(
    "qwen3-0.6b",
    enable_distributed=True,
    world_size=2,
    tensor_parallel_size=2,
    pipeline_parallel_size=1
)

prompts = ["Hello, world!"]
params = SamplingParams(max_tokens=50)

results = engine.generate(prompts, params)
print(results[0]["text"])
```

### Multiple Machines

```bash
# Machine 1 (Rank 0)
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 distributed_example.py

# Machine 2 (Rank 1)
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.1.100" --master_port=29500 distributed_example.py
```

## 🔧 Configuration Options

### Parallelism Types

| Type | Parameter | Description | Example |
|------|-----------|-------------|---------|
| **Tensor Parallel** | `tensor_parallel_size` | Split attention heads | `tensor_parallel_size=4` |
| **Pipeline Parallel** | `pipeline_parallel_size` | Split model layers | `pipeline_parallel_size=2` |
| **Expert Parallel** | `enable_expert_parallel` | Split MoE experts | `enable_expert_parallel=True` |

### Distributed Setup

```python
engine = DistributedEngine(
    "qwen3-0.6b",
    enable_distributed=True,
    world_size=4,                    # Total processes
    tensor_parallel_size=2,          # TP across 2 GPUs
    pipeline_parallel_size=2,        # PP across 2 stages
    master_addr="localhost",         # Coordinator address
    master_port="29500",             # Coordinator port
    rank=0,                          # Current process rank
)
```

## 🏗️ Architecture

### Process Layout
```
World Size: 4
├── Tensor Parallel Size: 2
│   ├── GPU 0: Heads 0-15
│   └── GPU 1: Heads 16-31
└── Pipeline Parallel Size: 2
    ├── Stage 0: Layers 0-11
    └── Stage 1: Layers 12-23
```

### Communication Patterns
- **Tensor Parallel**: All-reduce operations
- **Pipeline Parallel**: Send/receive activations
- **Load Balancing**: Coordinator-based distribution

## 📊 Performance Optimization

### Optimal Configurations

| GPUs | Model Size | Recommended Setup |
|------|------------|-------------------|
| 2 | < 7B | `tensor_parallel_size=2` |
| 4 | 7B-30B | `tensor_parallel_size=2, pipeline_parallel_size=2` |
| 8 | 30B+ | `tensor_parallel_size=4, pipeline_parallel_size=2` |

### Memory Optimization
```python
# Reduce memory usage
engine = DistributedEngine(
    "qwen3-30b",
    enable_distributed=True,
    world_size=4,
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    gpu_memory_utilization=0.85  # Lower memory usage
)
```

## 🎯 Usage Patterns

### Basic Distributed Inference
```python
from nanovllm.engine.distributed_engine import launch_distributed_inference

# Launch with optimal configuration
engine = launch_distributed_inference(
    "qwen3-30b",
    world_size=4,
    tensor_parallel_size=2,
    pipeline_parallel_size=2
)

# Use normally
results = engine.generate(["Hello world"], SamplingParams(max_tokens=50))
```

### Manual Process Launch
```python
import torch.distributed as dist
from nanovllm.engine.distributed_engine import DistributedEngine

def run_worker(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:29500',
        world_size=world_size,
        rank=rank
    )
    
    engine = DistributedEngine(
        "qwen3-0.6b",
        enable_distributed=True,
        world_size=world_size,
        rank=rank,
        tensor_parallel_size=2
    )
    
    # Process requests
    # ... rest of code ...

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size)
```

### Health Monitoring
```python
# Check cluster health
health = engine.health_check()
print(f"Cluster status: {health}")

# Get distributed stats
stats = engine.get_distributed_stats()
print(f"Active workers: {len(stats['worker_loads'])}")
```

## 🔍 Debugging

### Common Issues

**Issue**: Process hanging
```bash
# Check network connectivity
nc -zv localhost 29500

# Verify NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**Issue**: CUDA out of memory
```python
# Reduce batch size per worker
engine = DistributedEngine(
    "model",
    max_num_seqs=32,  # Reduce from default 512
    max_num_batched_tokens=1024  # Reduce from default 32768
)
```

**Issue**: Load imbalance
```python
# Force load balancing
engine.load_balance()

# Check load distribution
stats = engine.get_distributed_stats()
for rank, load in stats['worker_loads'].items():
    print(f"Worker {rank}: {load} requests")
```

### Debug Commands
```bash
# Check process status
ps aux | grep python

# Monitor network
netstat -tulpn | grep 29500

# Check GPU usage
nvidia-smi
```

## ⚙️ Configuration Reference

### Distributed Parameters

```python
config = {
    # Basic distributed setup
    "enable_distributed": True,
    "world_size": 4,
    "rank": 0,
    
    # Parallelism dimensions
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 2,
    
    # Network configuration
    "master_addr": "localhost",
    "master_port": "29500",
    
    # Communication settings
    "rpc_timeout": 60,
    "max_rpc_retries": 3,
    
    # Performance tuning
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 32768,
}
```

### Environment Variables

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0
export NCCL_DEBUG=INFO
```

## 🚀 Advanced Features

### Custom Load Balancing
```python
from nanovllm.engine.distributed_scheduler import DistributedScheduler

class CustomScheduler(DistributedScheduler):
    def _select_worker_for_sequence(self, seq):
        # Custom load balancing logic
        return self._custom_load_balancing(seq)
```

### Fault Tolerance
```python
# Automatic retry on worker failure
engine = DistributedEngine(
    "model",
    enable_distributed=True,
    max_rpc_retries=5,
    rpc_timeout=30
)

# Monitor health
health = engine.health_check()
for worker, status in health.items():
    if not status['healthy']:
        print(f"Worker {worker} needs attention")
```

## 📊 Performance Benchmarks

### Expected Speedups

| Model | GPUs | Throughput (tokens/s) | Speedup |
|-------|------|---------------------|---------|
| 7B | 1 | 100 | 1.0x |
| 7B | 2 | 190 | 1.9x |
| 7B | 4 | 360 | 3.6x |
| 30B | 1 | 25 | 1.0x |
| 30B | 4 | 95 | 3.8x |
| 30B | 8 | 180 | 7.2x |

## 🔧 Deployment Examples

### Docker Deployment
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN pip install nanovllm torch torchvision

# Expose port for distributed communication
EXPOSE 29500

CMD ["python", "-m", "torch.distributed.launch", "--nproc_per_node=2", "app.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nanovllm-distributed
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: nanovllm
        image: nanovllm:latest
        env:
        - name: WORLD_SIZE
          value: "4"
        - name: MASTER_ADDR
          value: "nanovllm-coordinator"
        - name: MASTER_PORT
          value: "29500"
```

## 🔄 Backwards Compatibility

Distributed features are **opt-in** and **backwards compatible**:

```python
# Standard usage (no changes)
engine = LLM("qwen3-0.6b")

# Distributed usage (new features)
engine = DistributedEngine("qwen3-0.6b", enable_distributed=True)
```

## 🎯 Next Steps

1. **Auto-scaling**: Dynamic worker addition/removal
2. **Model Serving**: HTTP API for distributed inference
3. **Caching**: Distributed KV-cache sharing
4. **Monitoring**: Real-time performance metrics
5. **A/B Testing**: Gradual rollout strategies