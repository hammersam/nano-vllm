from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str = ''
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    # MoE-specific configurations
    num_experts: int = 0
    max_expert_load: int = 100
    enable_expert_parallel: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    
    # Multi-modal configurations
    enable_multimodal: bool = False
    vision_model: str = "openai/clip-vit-base-patch32"
    max_image_size: int = 224
    num_vision_tokens: int = 50  # Number of image patch tokens
    
    # Distributed serving configurations
    enable_distributed: bool = False
    num_workers: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    
    # RPC configurations
    rpc_timeout: int = 60
    max_rpc_retries: int = 3

    def __post_init__(self):
        assert self.model
        assert self.kvcache_block_size % 256 == 0
        if self.enable_distributed:
            assert self.tensor_parallel_size * self.pipeline_parallel_size <= self.world_size
