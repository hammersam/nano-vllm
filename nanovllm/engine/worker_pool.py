import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Dict, Any, Optional, Callable
import os
import logging
import threading
import queue
import time
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.multimodal_model_runner import MultiModalModelRunner


logger = logging.getLogger(__name__)


@dataclass
class WorkerTask:
    """Task sent to worker for execution."""
    task_type: str  # 'prefill', 'decode', 'initialize'
    sequences: List[Sequence]
    is_prefill: bool
    request_id: str
    timestamp: float


@dataclass
class WorkerResult:
    """Result returned from worker."""
    request_id: str
    token_ids: List[int]
    status: str  # 'success', 'error'
    error_message: Optional[str] = None
    processing_time: float = 0.0


class WorkerProcess:
    """Individual worker process for distributed inference."""
    
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.model_runner = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the worker process."""
        try:
            # Initialize distributed communication
            dist.init_process_group(
                backend='nccl',
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.rank
            )
            
            # Create model runner based on configuration
            if self.config.enable_multimodal:
                self.model_runner = MultiModalModelRunner(self.config)
            else:
                self.model_runner = ModelRunner(self.config)
            
            self.initialized = True
            logger.info(f"Worker {self.rank} initialized successfully")
            
        except Exception as e:
            logger.error(f"Worker {self.rank} initialization failed: {e}")
            raise
    
    def process_task(self, task: WorkerTask) -> WorkerResult:
        """Process a task and return result."""
        start_time = time.time()
        
        try:
            if not self.initialized:
                self.initialize()
            
            # Process sequences
            if task.task_type in ['prefill', 'decode']:
                token_ids = self.model_runner.run(task.sequences, task.is_prefill)
                
                return WorkerResult(
                    request_id=task.request_id,
                    token_ids=token_ids,
                    status='success',
                    processing_time=time.time() - start_time
                )
            
            elif task.task_type == 'initialize':
                return WorkerResult(
                    request_id=task.request_id,
                    token_ids=[],
                    status='success',
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Worker {self.rank} task processing failed: {e}")
            return WorkerResult(
                request_id=task.request_id,
                token_ids=[],
                status='error',
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics from this worker."""
        if not self.model_runner:
            return {}
        
        stats = {
            'rank': self.rank,
            'device': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu',
            'memory_usage': self._get_memory_usage()
        }
        
        if hasattr(self.model_runner, 'get_vision_stats'):
            stats.update(self.model_runner.get_vision_stats())
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Get GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        return {
            'total_memory': total_memory,
            'allocated_memory': allocated_memory,
            'cached_memory': cached_memory,
            'free_memory': total_memory - cached_memory
        }


class WorkerPool:
    """Pool of worker processes for distributed inference."""
    
    def __init__(self, config: Config):
        self.config = config
        self.workers = []
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.processes = []
        self.initialized = False
        
        # Thread-safe communication
        self._lock = threading.Lock()
        self._results = {}
        self._worker_stats = {}
        
    def initialize(self) -> bool:
        """Initialize the worker pool."""
        if self.config.world_size <= 1:
            logger.info("Single process mode - no worker pool needed")
            return True
        
        try:
            # Setup spawn method for better compatibility
            mp.set_start_method('spawn', force=True)
            
            # Create worker processes
            for rank in range(1, self.config.world_size):
                process = mp.Process(
                    target=self._worker_target,
                    args=(self.config, rank, self.task_queue, self.result_queue)
                )
                process.start()
                self.processes.append(process)
                logger.info(f"Started worker process {rank}")
            
            self.initialized = True
            logger.info(f"Worker pool initialized with {len(self.processes)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Worker pool initialization failed: {e}")
            self.cleanup()
            return False
    
    def _worker_target(self, config: Config, rank: int, task_queue: mp.Queue, result_queue: mp.Queue):
        """Target function for worker processes."""
        try:
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(config.world_size)
            os.environ['MASTER_ADDR'] = config.master_addr
            os.environ['MASTER_PORT'] = config.master_port
            
            worker = WorkerProcess(config, rank)
            worker.initialize()
            
            while True:
                try:
                    task = task_queue.get(timeout=1.0)
                    if task is None:  # Shutdown signal
                        break
                    
                    result = worker.process_task(task)
                    result_queue.put(result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {rank} error: {e}")
                    result_queue.put(WorkerResult(
                        request_id="unknown",
                        token_ids=[],
                        status='error',
                        error_message=str(e)
                    ))
                    
        except Exception as e:
            logger.error(f"Worker {rank} fatal error: {e}")
    
    def submit_task(self, task: WorkerTask) -> str:
        """Submit a task to the worker pool."""
        if not self.initialized and self.config.world_size > 1:
            self.initialize()
        
        with self._lock:
            self.task_queue.put(task)
            return task.request_id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[WorkerResult]:
        """Get result for a specific request."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.result_queue.get(timeout=0.1)
                if result.request_id == request_id:
                    return result
                else:
                    # Store for later retrieval
                    self._results[result.request_id] = result
            except queue.Empty:
                continue
        
        # Check cached results
        with self._lock:
            if request_id in self._results:
                return self._results.pop(request_id)
        
        return None
    
    def broadcast_task(self, task: WorkerTask) -> List[str]:
        """Broadcast a task to all workers."""
        if self.config.world_size <= 1:
            return [task.request_id]
        
        request_ids = []
        for rank in range(1, self.config.world_size):
            new_task = WorkerTask(
                task_type=task.task_type,
                sequences=task.sequences,
                is_prefill=task.is_prefill,
                request_id=f"{task.request_id}_{rank}",
                timestamp=task.timestamp
            )
            self.submit_task(new_task)
            request_ids.append(new_task.request_id)
        
        return request_ids
    
    def get_worker_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics from all workers."""
        if self.config.world_size <= 1:
            return {0: {'status': 'single_process'}}
        
        # Request stats from workers
        task = WorkerTask(
            task_type='stats',
            sequences=[],
            is_prefill=False,
            request_id=f"stats_{time.time()}",
            timestamp=time.time()
        )
        
        request_ids = self.broadcast_task(task)
        stats = {}
        
        for request_id in request_ids:
            result = self.get_result(request_id, timeout=5.0)
            if result and result.status == 'success':
                # Parse stats from result
                stats[len(stats) + 1] = result.__dict__
        
        return stats
    
    def cleanup(self):
        """Clean up worker processes."""
        if not self.initialized:
            return
        
        # Send shutdown signal
        for _ in self.processes:
            self.task_queue.put(None)
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
        
        # Close queues
        self.task_queue.close()
        self.result_queue.close()
        
        self.processes.clear()
        self.initialized = False
        logger.info("Worker pool cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()


class TensorParallelWorker:
    """Worker for tensor parallelism within a single process."""
    
    def __init__(self, config: Config, rank: int, tp_rank: int, pp_rank: int):
        self.config = config
        self.global_rank = rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        
        # Calculate local model partition
        self.model_partition = self._calculate_model_partition()
        
    def _calculate_model_partition(self) -> Dict[str, Any]:
        """Calculate which parts of the model this worker handles."""
        total_layers = self.config.hf_config.num_hidden_layers
        
        # Pipeline parallel partitioning
        pp_size = self.config.pipeline_parallel_size
        layers_per_stage = total_layers // pp_size
        start_layer = self.pp_rank * layers_per_stage
        end_layer = (self.pp_rank + 1) * layers_per_stage
        
        # Tensor parallel partitioning (for attention heads)
        tp_size = self.config.tensor_parallel_size
        heads_per_worker = self.config.hf_config.num_attention_heads // tp_size
        start_head = self.tp_rank * heads_per_worker
        end_head = (self.tp_rank + 1) * heads_per_worker
        
        return {
            'start_layer': start_layer,
            'end_layer': end_layer,
            'start_head': start_head,
            'end_head': end_head,
            'layers': list(range(start_layer, end_layer)),
            'heads': list(range(start_head, end_head))
        }
    
    def forward_pass(self, hidden_states: torch.Tensor, stage: int) -> torch.Tensor:
        """Perform forward pass for this tensor parallel partition."""
        # This would be integrated with the actual model
        # For now, return identity
        return hidden_states


class PipelineParallelWorker:
    """Worker for pipeline parallelism."""
    
    def __init__(self, config: Config, stage: int):
        self.config = config
        self.stage = stage
        self.num_stages = config.pipeline_parallel_size
        
    def process_stage(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process this pipeline stage."""
        # This would integrate with model layers
        return hidden_states
    
    def send_to_next_stage(self, hidden_states: torch.Tensor):
        """Send activations to next pipeline stage."""
        if self.stage < self.num_stages - 1:
            next_stage = self.stage + 1
            # Send to next stage
            dist.send(hidden_states, dst=next_stage)
    
    def receive_from_prev_stage(self) -> torch.Tensor:
        """Receive activations from previous stage."""
        if self.stage > 0:
            prev_stage = self.stage - 1
            # Receive from previous stage
            hidden_states = torch.empty(...)  # Size would be determined
            dist.recv(hidden_states, src=prev_stage)
            return hidden_states
        return torch.empty(0)