from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        # prevent OOM error from processing too many prompt tokens at once
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # the scheduler constantly asks for memory from block_manager
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # holding all the new requests that haven't started processing yet
        self.waiting: deque[Sequence] = deque()
        # requets currently being processed, i.e. actively generating tokens on gpu
        self.running: deque[Sequence] = deque()
        self.num_finished = 0
        self.num_tokens = 0
        
        # MoE-specific optimizations
        self.expert_load = [0] * config.num_experts  # Track load per expert
        self.expert_affinity = {}  # Sequence to expert affinity mapping
        self.expert_parallel_groups = []  # Groups for expert parallelism
        self.max_expert_load = config.max_expert_load if hasattr(config, 'max_expert_load') else 100
        self.enable_expert_parallel = hasattr(config, 'enable_expert_parallel') and config.enable_expert_parallel
        self.expert_capacity_factor = getattr(config, 'expert_capacity_factor', 1.0)
        self.last_expert_usage = {}  # Track recent expert usage for load balancing

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], SequenceStatus]:
        # Prefill phase with MoE optimization
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # Group sequences by expert affinity for MoE optimization
        if self.enable_expert_parallel:
            # Sort waiting sequences by expert affinity to batch similar requests
            self.waiting = deque(sorted(
                self.waiting, 
                key=lambda s: hash(tuple(self.expert_affinity.get(s.id, [])))
            ))
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check if adding this sequence would exceed expert capacity
            if self.enable_expert_parallel:
                seq_experts = self.expert_affinity.get(seq.id, set())
                if any(self.expert_load[e] >= self.max_expert_load for e in seq_experts):
                    # Skip this sequence if any expert is overloaded
                    self.waiting.rotate(-1)  # Move to end of queue
                    continue
            
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
                not self.block_manager.can_allocate(seq)):
                break
                
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            
            # Update expert load and affinity for MoE sequences
            if self.enable_expert_parallel and hasattr(seq, 'expert_ids'):
                for eid in seq.expert_ids:
                    self.expert_load[eid] += 1
                    # Update affinity based on recent usage
                    if eid not in self.expert_affinity:
                        self.expert_affinity[eid] = set()
                    self.expert_affinity[eid].add(seq.id)
                # Track last used experts for load balancing
                self.last_expert_usage[seq.id] = seq.expert_ids

        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode phase with MoE optimization
        assert scheduled_seqs == []
        # Select sequences considering expert load balancing
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Check expert load for MoE sequences
            if self.enable_expert_parallel and hasattr(seq, 'expert_ids'):
                if any(self.expert_load[e] >= self.max_expert_load for e in seq.expert_ids):
                    # If experts are overloaded, preempt this sequence
                    self.preempt(seq)
                    continue
            
            preempted_self = False
            while not self.block_manager.can_append(seq):
                if not self.running:
                    self.preempt(seq)
                    preempted_self = True
                    break
                self.preempt(self.running.pop())

            if not preempted_self:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                
                # Update expert load for MoE sequences
                if self.enable_expert_parallel and hasattr(seq, 'expert_ids'):
                    for eid in seq.expert_ids:
                        self.expert_load[eid] += 1
        
        # Requeue non-scheduled sequences
        running = deque(scheduled_seqs)
        running.extend(self.running)
        self.running = running
        
        if scheduled_seqs:
            return scheduled_seqs, False
        else:
            # No sequences scheduled, return empty list
            return [], False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        # Reduce expert load when preempting MoE sequences
        if self.enable_expert_parallel and hasattr(seq, 'expert_ids'):
            for eid in seq.expert_ids:
                if self.expert_load[eid] > 0:
                    self.expert_load[eid] -= 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        self.num_tokens += len(token_ids)
        finished_seqs = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                self.num_finished += 1
                finished_seqs.append(seq)
                
                # Update expert load and clean up affinity for finished sequences
                if self.enable_expert_parallel and hasattr(seq, 'expert_ids'):
                    for eid in seq.expert_ids:
                        if self.expert_load[eid] > 0:
                            self.expert_load[eid] -= 1
                        # Clean up affinity mapping
                        if eid in self.expert_affinity and seq.id in self.expert_affinity[eid]:
                            self.expert_affinity[eid].remove(seq.id)
                    # Clean last expert usage
                    if seq.id in self.last_expert_usage:
                        del self.last_expert_usage[seq.id]

        return finished_seqs
