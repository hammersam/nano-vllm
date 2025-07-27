import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional
import logging
from collections import deque
import threading
import time

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler


logger = logging.getLogger(__name__)


class DistributedScheduler(Scheduler):
    """Distributed scheduler that coordinates work across multiple workers."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        
        self.world_size = config.world_size
        self.rank = config.rank
        
        # Distributed coordination
        self.is_coordinator = self.rank == 0
        self.worker_ranks = list(range(1, self.world_size)) if self.is_coordinator else []
        
        # Load balancing state
        self.worker_loads = {rank: 0 for rank in range(self.world_size)}
        self.worker_expert_loads = {rank: [0] * config.num_experts for rank in range(self.world_size)}
        
        # Pipeline parallelism state
        self.pipeline_stages = {}
        self.current_stage = 0
        
        # Tensor parallelism state
        self.tensor_parallel_groups = self._setup_tensor_parallel_groups()
        
        # Communication buffers
        self.request_buffer = deque()
        self.result_buffer = deque()
        self.pending_requests = {}
        
        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        
    def _setup_tensor_parallel_groups(self) -> List[List[int]]:
        """Setup tensor parallel groups based on configuration."""
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        
        groups = []
        for i in range(0, self.world_size, tp_size):
            group = list(range(i, min(i + tp_size, self.world_size)))
            groups.append(group)
        
        return groups
    
    def distribute_sequence(self, seq: Sequence) -> int:
        """Distribute sequence to appropriate worker based on load balancing."""
        if self.is_coordinator:
            # Coordinator decides which worker to assign
            target_worker = self._select_worker_for_sequence(seq)
            self.worker_loads[target_worker] += 1
            
            if hasattr(seq, 'expert_ids'):
                for expert_id in seq.expert_ids:
                    self.worker_expert_loads[target_worker][expert_id] += 1
            
            return target_worker
        else:
            # Worker will receive assignment from coordinator
            return self.rank
    
    def _select_worker_for_sequence(self, seq: Sequence) -> int:
        """Select optimal worker for sequence based on load and expert affinity."""
        # Simple round-robin with expert affinity
        if hasattr(seq, 'expert_ids') and seq.expert_ids:
            # Find worker with lowest expert load for required experts
            best_worker = 0
            min_load = float('inf')
            
            for worker_rank in self.worker_ranks:
                total_expert_load = sum(
                    self.worker_expert_loads[worker_rank][expert_id]
                    for expert_id in seq.expert_ids
                )
                
                if total_expert_load < min_load:
                    min_load = total_expert_load
                    best_worker = worker_rank
            
            return best_worker
        
        # Fallback to round-robin based on general load
        return min(self.worker_ranks, key=lambda w: self.worker_loads[w])
    
    def schedule_distributed(self) -> List[Sequence]:
        """Schedule sequences across distributed workers."""
        if self.is_coordinator:
            return self._schedule_coordinator()
        else:
            return self._schedule_worker()
    
    def _schedule_coordinator(self) -> List[Sequence]:
        """Coordinator scheduling logic."""
        scheduled_seqs = []
        num_seqs = 0
        
        # Process waiting sequences
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check if we can schedule this sequence
            if not self._can_schedule_distributed(seq):
                break
            
            # Distribute to worker
            target_worker = self.distribute_sequence(seq)
            seq.distributed_worker = target_worker
            
            # Allocate resources
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            
            num_seqs += 1
        
        return scheduled_seqs
    
    def _schedule_worker(self) -> List[Sequence]:
        """Worker scheduling logic."""
        # Workers only process sequences assigned to them
        worker_seqs = [
            seq for seq in self.running 
            if hasattr(seq, 'distributed_worker') and seq.distributed_worker == self.rank
        ]
        
        return worker_seqs
    
    def _can_schedule_distributed(self, seq: Sequence) -> bool:
        """Check if sequence can be scheduled in distributed mode."""
        # Check basic constraints
        if not self.block_manager.can_allocate(seq):
            return False
        
        # Check pipeline constraints
        if self.config.pipeline_parallel_size > 1:
            # Check if we have capacity for pipeline stages
            required_stages = self._calculate_pipeline_stages(seq)
            if required_stages > self.config.pipeline_parallel_size:
                return False
        
        return True
    
    def _calculate_pipeline_stages(self, seq: Sequence) -> int:
        """Calculate required pipeline stages for sequence."""
        # Simple heuristic based on sequence length
        seq_len = len(seq)
        
        if seq_len < 512:
            return 1
        elif seq_len < 1024:
            return 2
        else:
            return min(4, self.config.pipeline_parallel_size)
    
    def broadcast_request(self, seq: Sequence) -> None:
        """Broadcast request to all workers."""
        if not self.is_coordinator:
            return
        
        # Prepare request data
        request_data = {
            'seq_id': seq.seq_id,
            'token_ids': seq.token_ids,
            'sampling_params': {
                'temperature': seq.temperature,
                'max_tokens': seq.max_tokens,
                'ignore_eos': seq.ignore_eos
            }
        }
        
        # Broadcast to all workers
        for worker_rank in self.worker_ranks:
            dist.send(
                torch.tensor([len(str(request_data))], dtype=torch.long),
                dst=worker_rank
            )
            dist.send(
                torch.tensor(list(str(request_data).encode()), dtype=torch.uint8),
                dst=worker_rank
            )
    
    def gather_results(self, seq: Sequence) -> Dict[str, Any]:
        """Gather results from distributed workers."""
        if self.is_coordinator:
            # Gather from assigned worker
            worker_rank = seq.distributed_worker
            if worker_rank != 0:
                # Receive result from worker
                result_length = torch.tensor([0], dtype=torch.long)
                dist.recv(result_length, src=worker_rank)
                
                result_data = torch.zeros(result_length.item(), dtype=torch.uint8)
                dist.recv(result_data, src=worker_rank)
                
                return eval(result_data.tobytes().decode())
        
        return {}
    
    def update_worker_stats(self, worker_rank: int, stats: Dict[str, int]) -> None:
        """Update load statistics for worker."""
        if self.is_coordinator:
            self.worker_loads[worker_rank] = stats.get('load', 0)
            if 'expert_loads' in stats:
                self.worker_expert_loads[worker_rank] = stats['expert_loads']
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get distributed scheduling statistics."""
        return {
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'worker_loads': self.worker_loads,
            'worker_expert_loads': self.worker_expert_loads,
            'is_coordinator': self.is_coordinator,
            'world_size': self.world_size,
            'rank': self.rank
        }
    
    def load_balance(self) -> None:
        """Perform load balancing across workers."""
        if not self.is_coordinator:
            return
        
        # Check for load imbalance
        max_load = max(self.worker_loads.values())
        min_load = min(self.worker_loads.values())
        
        if max_load - min_load > 10:  # Threshold for rebalancing
            # Identify sequences to migrate
            overloaded_worker = max(self.worker_loads, key=self.worker_loads.get)
            underloaded_worker = min(self.worker_loads, key=self.worker_loads.get)
            
            # Find sequences to migrate
            sequences_to_migrate = [
                seq for seq in self.running
                if hasattr(seq, 'distributed_worker') and seq.distributed_worker == overloaded_worker
            ]
            
            # Migrate sequences
            for seq in sequences_to_migrate[:5]:  # Limit migration batch size
                seq.distributed_worker = underloaded_worker
                self.worker_loads[overloaded_worker] -= 1
                self.worker_loads[underloaded_worker] += 1
    
    def health_check(self) -> Dict[int, bool]:
        """Check health of all workers."""
        if not self.is_coordinator:
            return {self.rank: True}
        
        health_status = {0: True}  # Coordinator is healthy
        
        for worker_rank in self.worker_ranks:
            try:
                # Send health check ping
                ping = torch.tensor([1], dtype=torch.long)
                dist.send(ping, dst=worker_rank)
                
                # Wait for response
                response = torch.tensor([0], dtype=torch.long)
                dist.recv(response, src=worker_rank)
                
                health_status[worker_rank] = response.item() == 1
                
            except Exception as e:
                logger.warning(f"Worker {worker_rank} health check failed: {e}")
                health_status[worker_rank] = False
        
        return health_status
    
    def synchronize_state(self) -> None:
        """Synchronize scheduler state across workers."""
        if self.world_size <= 1:
            return
        
        if self.is_coordinator:
            # Broadcast current state to workers
            state_data = {
                'waiting_count': len(self.waiting),
                'running_count': len(self.running),
                'finished_count': self.num_finished
            }
            
            for worker_rank in self.worker_ranks:
                dist.send(
                    torch.tensor([state_data['waiting_count']], dtype=torch.long),
                    dst=worker_rank
                )
                dist.send(
                    torch.tensor([state_data['running_count']], dtype=torch.long),
                    dst=worker_rank
                )
                dist.send(
                    torch.tensor([state_data['finished_count']], dtype=torch.long),
                    dst=worker_rank
                )
        else:
            # Receive state from coordinator
            waiting_count = torch.tensor([0], dtype=torch.long)
            running_count = torch.tensor([0], dtype=torch.long)
            finished_count = torch.tensor([0], dtype=torch.long)
            
            dist.recv(waiting_count, src=0)
            dist.recv(running_count, src=0)
            dist.recv(finished_count, src=0)
            
            logger.info(
                f"Worker {self.rank} state sync: "
                f"waiting={waiting_count.item()}, "
                f"running={running_count.item()}, "
                f"finished={finished_count.item()}"
            )