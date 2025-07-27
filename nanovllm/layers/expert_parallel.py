import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class ExpertParallelConfig:
    """Configuration for expert parallelism."""
    num_experts: int
    world_size: int
    rank: int
    expert_capacity_factor: float = 1.0
    max_expert_load: int = 100
    enable_expert_parallel: bool = True
    communication_backend: str = "nccl"
    expert_balance_strategy: str = "round_robin"  # or "load_balancing"


class ExpertLoadBalancer:
    """Dynamic load balancing for expert parallelism."""
    
    def __init__(self, num_experts: int, world_size: int):
        self.num_experts = num_experts
        self.world_size = world_size
        self.expert_loads = torch.zeros(num_experts, dtype=torch.float32)
        self.device_loads = torch.zeros(world_size, dtype=torch.float32)
        self.lock = threading.Lock()
        
    def update_expert_load(self, expert_id: int, load: float):
        """Update load for a specific expert."""
        with self.lock:
            self.expert_loads[expert_id] = load
            
    def get_optimal_expert_assignment(self) -> torch.Tensor:
        """Calculate optimal expert to device mapping based on current loads."""
        # Simple load balancing - distribute based on current loads
        expert_device_map = torch.zeros(self.num_experts, dtype=torch.int64)
        
        # Sort experts by load
        sorted_experts = torch.argsort(self.expert_loads, descending=True)
        
        # Assign to least loaded device
        for expert_id in sorted_experts:
            least_loaded_device = torch.argmin(self.device_loads)
            expert_device_map[expert_id] = least_loaded_device
            self.device_loads[least_loaded_device] += self.expert_loads[expert_id]
            
        return expert_device_map
    
    def redistribute_experts(self, threshold: float = 0.2) -> bool:
        """Check if expert redistribution is needed."""
        max_load = torch.max(self.device_loads)
        min_load = torch.min(self.device_loads)
        imbalance = (max_load - min_load) / (max_load + 1e-8)
        return imbalance > threshold


class ExpertCommunicator:
    """Handles communication between devices for expert parallelism."""
    
    def __init__(self, config: ExpertParallelConfig):
        self.config = config
        self.rank = config.rank
        self.world_size = config.world_size
        
    def send_tokens_to_expert(
        self,
        tokens: torch.Tensor,
        expert_id: int,
        target_rank: int
    ) -> torch.Tensor:
        """Send tokens to expert on target device."""
        if self.rank == target_rank:
            return tokens
            
        if dist.is_initialized():
            dist.send(tokens.contiguous(), dst=target_rank)
            return torch.empty(0, device=tokens.device, dtype=tokens.dtype)
        else:
            # Fallback for single device
            return tokens
    
    def receive_tokens_from_expert(
        self,
        expert_id: int,
        source_rank: int,
        expected_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """Receive tokens from expert on source device."""
        if self.rank == source_rank:
            return torch.empty(expected_shape, dtype=dtype, device=device)
            
        if dist.is_initialized():
            received = torch.empty(expected_shape, dtype=dtype, device=device)
            dist.recv(received, src=source_rank)
            return received
        else:
            return torch.empty(expected_shape, dtype=dtype, device=device)
            
    def all_reduce_expert_outputs(
        self,
        expert_outputs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """Aggregate expert outputs across all devices."""
        if not dist.is_initialized() or self.world_size == 1:
            return expert_outputs
            
        # Prepare tensors for all-reduce
        all_outputs = []
        expert_ids = []
        
        for expert_id, output in expert_outputs.items():
            all_outputs.append(output)
            expert_ids.append(expert_id)
            
        if all_outputs:
            # Concatenate all outputs
            concatenated = torch.cat(all_outputs, dim=0)
            dist.all_reduce(concatenated, op=dist.ReduceOp.SUM)
            
            # Split back to individual experts
            split_sizes = [out.shape[0] for out in all_outputs]
            split_outputs = torch.split(concatenated, split_sizes)
            
            # Rebuild dictionary
            result = {}
            for expert_id, output in zip(expert_ids, split_outputs):
                result[expert_id] = output
                
            return result
            
        return expert_outputs


class ExpertParallelManager:
    """Main manager for expert parallelism."""
    
    def __init__(self, config: ExpertParallelConfig):
        self.config = config
        self.load_balancer = ExpertLoadBalancer(config.num_experts, config.world_size)
        self.communicator = ExpertCommunicator(config)
        
        # Expert assignment
        self.expert_to_device = self._initialize_expert_assignment()
        self.device_to_experts = self._create_device_expert_map()
        
        # Statistics
        self.expert_stats = {
            'tokens_processed': torch.zeros(config.num_experts),
            'load_history': [],
            'redistribution_count': 0
        }
        
    def _initialize_expert_assignment(self) -> torch.Tensor:
        """Initialize expert to device mapping."""
        if self.config.expert_balance_strategy == "round_robin":
            return torch.arange(self.config.num_experts) % self.config.world_size
        elif self.config.expert_balance_strategy == "load_balancing":
            return self.load_balancer.get_optimal_expert_assignment()
        else:
            # Default round-robin
            return torch.arange(self.config.num_experts) % self.config.world_size
            
    def _create_device_expert_map(self) -> Dict[int, List[int]]:
        """Create mapping from device to experts."""
        device_experts = {}
        for device_id in range(self.config.world_size):
            experts = torch.where(self.expert_to_device == device_id)[0]
            device_experts[device_id] = experts.tolist()
        return device_experts
    
    def get_expert_device(self, expert_id: int) -> int:
        """Get device ID for a specific expert."""
        return int(self.expert_to_device[expert_id])
    
    def get_local_experts(self) -> List[int]:
        """Get experts assigned to current device."""
        return self.device_to_experts.get(self.config.rank, [])
    
    def route_tokens(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Route tokens to appropriate devices based on expert assignment.
        
        Returns:
            Dictionary with keys:
            - 'local_tokens': tokens for local experts
            - 'local_experts': expert indices for local tokens
            - 'local_weights': routing weights for local tokens
            - 'remote_routing': dict of expert_id -> (tokens, weights, target_device)
        """
        result = {
            'local_tokens': [],
            'local_experts': [],
            'local_weights': [],
            'remote_routing': {}
        }
        
        # Process each expert assignment
        for expert_id in torch.unique(expert_indices):
            expert_mask = expert_indices == expert_id
            target_device = self.get_expert_device(expert_id)
            
            if target_device == self.config.rank:
                # Local processing
                result['local_tokens'].append(tokens[expert_mask])
                result['local_experts'].append(expert_indices[expert_mask])
                result['local_weights'].append(routing_weights[expert_mask])
            else:
                # Remote routing
                if expert_id not in result['remote_routing']:
                    result['remote_routing'][expert_id] = {
                        'tokens': [],
                        'weights': [],
                        'target_device': target_device
                    }
                result['remote_routing'][expert_id]['tokens'].append(tokens[expert_mask])
                result['remote_routing'][expert_id]['weights'].append(routing_weights[expert_mask])
        
        # Concatenate local tensors
        if result['local_tokens']:
            result['local_tokens'] = torch.cat(result['local_tokens'], dim=0)
            result['local_experts'] = torch.cat(result['local_experts'], dim=0)
            result['local_weights'] = torch.cat(result['local_weights'], dim=0)
        else:
            result['local_tokens'] = torch.empty(0, tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
            result['local_experts'] = torch.empty(0, device=tokens.device, dtype=torch.int64)
            result['local_weights'] = torch.empty(0, device=tokens.device, dtype=tokens.dtype)
        
        # Concatenate remote tensors
        for expert_id, routing in result['remote_routing'].items():
            if routing['tokens']:
                routing['tokens'] = torch.cat(routing['tokens'], dim=0)
                routing['weights'] = torch.cat(routing['weights'], dim=0)
            else:
                routing['tokens'] = torch.empty(0, tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
                routing['weights'] = torch.empty(0, device=tokens.device, dtype=tokens.dtype)
        
        return result
    
    def redistribute_experts_if_needed(self) -> bool:
        """Check and perform expert redistribution if load imbalance is detected."""
        if self.load_balancer.redistribute_experts():
            new_assignment = self.load_balancer.get_optimal_expert_assignment()
            self.expert_to_device = new_assignment
            self.device_to_experts = self._create_device_expert_map()
            self.expert_stats['redistribution_count'] += 1
            logger.info(f"Expert redistribution performed on rank {self.config.rank}")
            return True
        return False
    
    def update_expert_stats(self, expert_id: int, num_tokens: int):
        """Update statistics for expert processing."""
        self.expert_stats['tokens_processed'][expert_id] += num_tokens
        
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get current expert statistics."""
        return {
            'rank': self.config.rank,
            'expert_assignment': self.expert_to_device.tolist(),
            'local_experts': self.get_local_experts(),
            'stats': self.expert_stats
        }


class ExpertParallelEngine:
    """High-level engine for expert parallelism."""
    
    def __init__(self, num_experts: int, expert_capacity_factor: float = 1.0):
        self.num_experts = num_experts
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        config = ExpertParallelConfig(
            num_experts=num_experts,
            world_size=self.world_size,
            rank=self.rank,
            expert_capacity_factor=expert_capacity_factor
        )
        
        self.manager = ExpertParallelManager(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def process_with_expert_parallelism(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Process tokens with expert parallelism.
        
        Args:
            tokens: Input tokens [batch_size * seq_len, hidden_size]
            expert_indices: Expert assignments [batch_size * seq_len, num_experts_per_tok]
            routing_weights: Expert routing weights
            expert_weights: Expert weight matrices
            
        Returns:
            Processed tokens with expert computation
        """
        # Route tokens based on expert assignment
        routing_result = self.manager.route_tokens(tokens, expert_indices, routing_weights)
        
        # Process local experts
        local_output = self._process_local_experts(
            routing_result['local_tokens'],
            routing_result['local_experts'],
            routing_result['local_weights'],
            expert_weights
        )
        
        # Process remote experts (async)
        remote_futures = []
        for expert_id, routing in routing_result['remote_routing'].items():
            if routing['tokens'].numel() > 0:
                future = self.executor.submit(
                    self._process_remote_expert,
                    expert_id,
                    routing['tokens'],
                    routing['weights'],
                    routing['target_device']
                )
                remote_futures.append((expert_id, future))
        
        # Collect remote results
        remote_outputs = {}
        for expert_id, future in remote_futures:
            remote_outputs[expert_id] = future.result()
        
        # Combine results
        final_output = self._combine_expert_outputs(
            local_output, 
            remote_outputs, 
            tokens.shape
        )
        
        return final_output
    
    def _process_local_experts(
        self,
        tokens: torch.Tensor,
        expert_indices: torch.Tensor,
        weights: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Process tokens for local experts."""
        local_outputs = {}
        
        for local_expert_id in self.manager.get_local_experts():
            expert_mask = expert_indices == local_expert_id
            if expert_mask.any():
                expert_tokens = tokens[expert_mask]
                expert_weights_local = weights[expert_mask]
                
                # Compute expert output
                expert_weight = expert_weights[local_expert_id]
                output = torch.matmul(expert_tokens, expert_weight.T)
                output = output * expert_weights_local.unsqueeze(-1)
                
                local_outputs[local_expert_id] = output
                
        return local_outputs
    
    def _process_remote_expert(
        self,
        expert_id: int,
        tokens: torch.Tensor,
        weights: torch.Tensor,
        target_device: int
    ) -> torch.Tensor:
        """Process tokens on remote expert."""
        # Send tokens to remote device
        remote_tokens = self.manager.communicator.send_tokens_to_expert(
            tokens, expert_id, target_device
        )
        
        if self.rank == target_device:
            # Process on this device
            # This would be the actual expert computation
            output = torch.matmul(remote_tokens, torch.randn_like(remote_tokens))
            output = output * weights.unsqueeze(-1)
            return output
        
        return torch.empty(0, tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
    
    def _combine_expert_outputs(
        self,
        local_outputs: Dict[int, torch.Tensor],
        remote_outputs: Dict[int, torch.Tensor],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """Combine outputs from all experts."""
        # This is a simplified combination
        # In practice, would need proper reordering based on original token positions
        all_outputs = {**local_outputs, **remote_outputs}
        
        if not all_outputs:
            return torch.zeros(original_shape, device=original_shape.device)
        
        # Simple concatenation for now
        combined = torch.cat(list(all_outputs.values()), dim=0)
        
        # Pad or truncate to match original shape
        if combined.shape[0] < original_shape[0]:
            padding = torch.zeros(
                original_shape[0] - combined.shape[0],
                combined.shape[1],
                device=combined.device,
                dtype=combined.dtype
            )
            combined = torch.cat([combined, padding], dim=0)
        elif combined.shape[0] > original_shape[0]:
            combined = combined[:original_shape[0]]
            
        return combined
    
    def get_stats(self) -> Dict[str, Any]:
        """Get expert parallelism statistics."""
        return self.manager.get_expert_stats()
    
    def shutdown(self):
        """Shutdown the expert parallel engine."""
        self.executor.shutdown(wait=True)