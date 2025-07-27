import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Union
import logging
import os
import time
from tqdm.auto import tqdm

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.multimodal_sequence import MultiModalSequence
from nanovllm.engine.distributed_scheduler import DistributedScheduler
from nanovllm.engine.worker_pool import WorkerPool, WorkerTask
from nanovllm.engine.multimodal_model_runner import MultiModalModelRunner
from nanovllm.engine.model_runner import ModelRunner


logger = logging.getLogger(__name__)


class DistributedEngine:
    """Distributed inference engine with support for tensor and pipeline parallelism."""
    
    def __init__(self, model, **kwargs):
        self.config = Config(model)
        
        # Apply configuration overrides
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Initialize distributed setup
        self._setup_distributed()
        
        # Initialize components based on configuration
        if self.config.enable_distributed:
            self.scheduler = DistributedScheduler(self.config)
            self.worker_pool = WorkerPool(self.config)
            
            # Initialize model runner for coordinator
            if self.config.enable_multimodal:
                self.model_runner = MultiModalModelRunner(self.config)
            else:
                self.model_runner = ModelRunner(self.config)
        else:
            # Fallback to standard engine
            from nanovllm.engine.llm_engine import LLMEngine
            self.engine = LLMEngine(model, **kwargs)
            return
    
    def _setup_distributed(self):
        """Setup distributed environment."""
        if not self.config.enable_distributed:
            return
        
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        
        # Initialize distributed process group
        if self.config.world_size > 1:
            dist.init_process_group(
                backend='nccl',
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        logger.info(f"Distributed engine initialized: rank={self.config.rank}, world_size={self.config.world_size}")
    
    def add_request(
        self,
        prompt: str | List[int],
        sampling_params: SamplingParams,
        images: Optional[List] = None
    ):
        """Add request for distributed processing."""
        if not self.config.enable_distributed:
            return self.engine.add_request(prompt, sampling_params)
        
        # Create appropriate sequence type
        if isinstance(prompt, str):
            token_ids = self.model_runner.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        
        if self.config.enable_multimodal and images:
            seq = MultiModalSequence(token_ids, sampling_params, images=images)
        else:
            seq = Sequence(token_ids, sampling_params)
        
        self.scheduler.add(seq)
    
    def step(self) -> tuple[List[tuple[int, List[int]]], int]:
        """Execute one step of distributed inference."""
        if not self.config.enable_distributed:
            return self.engine.step()
        
        # Use distributed scheduler
        seqs = self.scheduler.schedule_distributed()
        
        if not seqs:
            return [], 0
        
        # Process sequences based on worker assignment
        if self.scheduler.is_coordinator:
            return self._step_coordinator(seqs)
        else:
            return self._step_worker(seqs)
    
    def _step_coordinator(self, seqs: List[Sequence]) -> tuple[List[tuple[int, List[int]]], int]:
        """Coordinator step logic."""
        # Distribute work to workers
        worker_assignments = {}
        for seq in seqs:
            worker_rank = self.scheduler.distribute_sequence(seq)
            if worker_rank not in worker_assignments:
                worker_assignments[worker_rank] = []
            worker_assignments[worker_rank].append(seq)
        
        # Submit tasks to workers
        results = []
        for worker_rank, worker_seqs in worker_assignments.items():
            if worker_rank == 0:
                # Process locally
                is_prefill = all(seq.num_cached_tokens == 0 for seq in worker_seqs)
                token_ids = self.model_runner.run(worker_seqs, is_prefill)
                for seq, token_id in zip(worker_seqs, token_ids):
                    seq.append_token(token_id)
                    if (not seq.ignore_eos and token_id == self.config.eos) or \
                       seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.scheduler.block_manager.deallocate(seq)
                        results.append((seq.seq_id, seq[seq.num_prompt_tokens:]))
            else:
                # Submit to worker pool
                task = WorkerTask(
                    task_type='prefill' if all(seq.num_cached_tokens == 0 for seq in worker_seqs) else 'decode',
                    sequences=worker_seqs,
                    is_prefill=all(seq.num_cached_tokens == 0 for seq in worker_seqs),
                    request_id=f"step_{worker_rank}_{int(time.time() * 1000)}",
                    timestamp=time.time()
                )
                
                self.worker_pool.submit_task(task)
        
        # Collect results from workers
        collected_results = []
        for worker_rank, worker_seqs in worker_assignments.items():
            if worker_rank != 0:
                # Wait for worker results
                task_id = f"step_{worker_rank}_{int(time.time() * 1000)}"
                result = self.worker_pool.get_result(task_id, timeout=10.0)
                
                if result and result.status == 'success':
                    for seq, token_id in zip(worker_seqs, result.token_ids):
                        seq.append_token(token_id)
                        if (not seq.ignore_eos and token_id == self.config.eos) or \
                           seq.num_completion_tokens == seq.max_tokens:
                            seq.status = SequenceStatus.FINISHED
                            self.scheduler.block_manager.deallocate(seq)
                            collected_results.append((seq.seq_id, seq[seq.num_prompt_tokens:]))
        
        results.extend(collected_results)
        
        # Calculate throughput
        num_tokens = sum(len(seq) for seq in seqs) if seqs[0].num_cached_tokens == 0 else -len(seqs)
        
        return results, num_tokens
    
    def _step_worker(self, seqs: List[Sequence]) -> tuple[List[tuple[int, List[int]]], int]:
        """Worker step logic."""
        # Workers process their assigned sequences
        if not seqs:
            return [], 0
        
        is_prefill = all(seq.num_cached_tokens == 0 for seq in seqs)
        token_ids = self.model_runner.run(seqs, is_prefill)
        
        results = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.config.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.block_manager.deallocate(seq)
                results.append((seq.seq_id, seq[seq.num_prompt_tokens:]))
        
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return results, num_tokens
    
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        images: Optional[List[List]] = None,
        use_tqdm: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate responses using distributed inference."""
        if not self.config.enable_distributed:
            # Fallback to standard engine
            return self.engine.generate(prompts, sampling_params, use_tqdm)
        
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating (distributed)",
                dynamic_ncols=True,
            )
        
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        if images is None:
            images = [None] * len(prompts)
        
        # Add all requests
        for prompt, sp, img_list in zip(prompts, sampling_params, images):
            self.add_request(prompt, sp, img_list)
        
        # Process requests
        outputs = {}
        completed_count = 0
        
        while completed_count < len(prompts):
            results, _ = self.step()
            
            for seq_id, token_ids in results:
                outputs[seq_id] = token_ids
                completed_count += 1
                
                if use_tqdm:
                    pbar.update(1)
        
        # Process results
        results = []
        for seq_id in sorted(outputs):
            token_ids = outputs[seq_id]
            text = self.model_runner.tokenizer.decode(token_ids)
            
            results.append({
                "text": text,
                "token_ids": token_ids,
                "rank": self.config.rank
            })
        
        if use_tqdm:
            pbar.close()
        
        return results
    
    def is_finished(self) -> bool:
        """Check if all requests are finished."""
        if not self.config.enable_distributed:
            return self.engine.is_finished()
        
        return self.scheduler.is_finished()
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get distributed inference statistics."""
        if not self.config.enable_distributed:
            return {"distributed_enabled": False}
        
        stats = {
            "distributed_enabled": True,
            "world_size": self.config.world_size,
            "rank": self.config.rank,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "pipeline_parallel_size": self.config.pipeline_parallel_size,
            "is_coordinator": self.scheduler.is_coordinator if hasattr(self.scheduler, 'is_coordinator') else True
        }
        
        if hasattr(self.scheduler, 'get_distributed_stats'):
            stats.update(self.scheduler.get_distributed_stats())
        
        if hasattr(self, 'worker_pool') and self.worker_pool.initialized:
            stats['worker_stats'] = self.worker_pool.get_worker_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of distributed system."""
        if not self.config.enable_distributed:
            return {"status": "single_process"}
        
        health = {
            "status": "distributed",
            "rank": self.config.rank,
            "world_size": self.config.world_size
        }
        
        if hasattr(self.scheduler, 'health_check'):
            health['scheduler_health'] = self.scheduler.health_check()
        
        return health
    
    def synchronize(self):
        """Synchronize state across all workers."""
        if not self.config.enable_distributed:
            return
        
        if hasattr(self.scheduler, 'synchronize_state'):
            self.scheduler.synchronize_state()
    
    def load_balance(self):
        """Perform load balancing across workers."""
        if not self.config.enable_distributed:
            return
        
        if hasattr(self.scheduler, 'load_balance'):
            self.scheduler.load_balance()
    
    def shutdown(self):
        """Gracefully shutdown distributed engine."""
        if not self.config.enable_distributed:
            return
        
        if hasattr(self, 'worker_pool'):
            self.worker_pool.cleanup()
        
        if dist.is_initialized():
            dist.destroy_process_group()
        
        logger.info("Distributed engine shutdown completed")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass


class DistributedLLM(DistributedEngine):
    """Convenience alias for DistributedEngine."""
    pass


def launch_distributed_inference(
    model: str,
    world_size: int,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    **kwargs
) -> DistributedEngine:
    """
    Launch distributed inference with specified configuration.
    
    Args:
        model: Model identifier
        world_size: Total number of processes
        tensor_parallel_size: Tensor parallelism size
        pipeline_parallel_size: Pipeline parallelism size
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured DistributedEngine instance
    """
    config = Config(
        model=model,
        enable_distributed=True,
        world_size=world_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        **kwargs
    )
    
    return DistributedEngine(model, **config.__dict__)


if __name__ == "__main__":
    # Example usage for distributed inference
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed nano-vLLM inference")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--world-size", type=int, default=2, help="World size")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--rank", type=int, default=0, help="Current rank")
    
    args = parser.parse_args()
    
    engine = DistributedEngine(
        args.model,
        enable_distributed=True,
        world_size=args.world_size,
        tensor_parallel_size=args.tensor_parallel,
        pipeline_parallel_size=args.pipeline_parallel,
        rank=args.rank
    )
    
    print(f"Distributed engine initialized on rank {args.rank}")