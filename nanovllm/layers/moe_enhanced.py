import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_kernel_v2 import (
    token_permutation_v2,
    invoke_segmented_gemm_v2,
    expert_gating_v2,
    load_balancing_v2,
    compute_gradients_v2,
    fused_activation_v2,
    expert_attention_mask_v2
)
from nanovllm.layers.expert_parallel import ExpertParallelEngine

logger = logging.getLogger(__name__)


class EnhancedMoEGate(nn.Module):
    """Enhanced MoE gate with expert parallelism support."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        gate_logit_softcapping: Optional[float] = None,
        use_triton: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_logit_softcapping = gate_logit_softcapping
        self.use_triton = use_triton
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional Triton optimization."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Compute gate logits
        gate_logits = self.gate(hidden_states_flat)
        
        # Apply soft capping if specified
        if self.gate_logit_softcapping is not None:
            gate_logits = gate_logits / self.gate_logit_softcapping
            gate_logits = torch.tanh(gate_logits)
            gate_logits = gate_logits * self.gate_logit_softcapping
        
        if self.use_triton:
            # Use Triton-optimized gating
            routing_weights, selected_experts = expert_gating_v2(
                gate_logits, self.num_experts_per_tok
            )
        else:
            # Standard PyTorch implementation
            routing_weights = F.softmax(gate_logits, dim=-1)
            routing_weights, selected_experts = torch.topk(
                routing_weights, self.num_experts_per_tok, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


class EnhancedMoEExpert(nn.Module):
    """Enhanced MoE expert with optimized computation."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Use parallel linear layers for efficiency
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=use_bias,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
        )
        
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class EnhancedSparseMoE(nn.Module):
    """Enhanced Sparse MoE with expert parallelism and optimized kernels."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        shared_expert_intermediate_size: int,
        num_experts_per_tok: int,
        num_shared_experts: int = 0,
        gate_logit_softcapping: Optional[float] = None,
        hidden_act: str = "silu",
        enable_expert_parallel: bool = False,
        use_triton: bool = True,
        expert_capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.enable_expert_parallel = enable_expert_parallel
        self.use_triton = use_triton
        
        # Expert parallelism
        if enable_expert_parallel:
            self.expert_parallel_engine = ExpertParallelEngine(
                num_experts=num_experts,
                expert_capacity_factor=expert_capacity_factor
            )
        
        # Enhanced gating
        self.gate = EnhancedMoEGate(
            hidden_size,
            num_experts,
            num_experts_per_tok,
            gate_logit_softcapping,
            use_triton
        )
        
        # Experts
        self.experts = nn.ModuleList([
            EnhancedMoEExpert(hidden_size, intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])
        
        # Shared experts
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                EnhancedMoEExpert(hidden_size, shared_expert_intermediate_size, hidden_act)
                for _ in range(num_shared_experts)
            ])
        
        # Activation
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
        
        # Pre-compute expert weight stacks for efficiency
        self._update_expert_weights()
        
    def _update_expert_weights(self):
        """Update stacked expert weights for efficient computation."""
        # Stack expert weights
        w1_weights = []
        w2_weights = []
        
        for expert in self.experts:
            w1_weights.append(expert.gate_up_proj.weight)
            w2_weights.append(expert.down_proj.weight)
        
        self.register_buffer("w1_stacked", torch.stack(w1_weights, dim=0))
        self.register_buffer("w2_stacked", torch.stack(w2_weights, dim=0))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced MoE computation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states_flat.shape[0]
        
        # Expert gating
        routing_weights, selected_experts = self.gate(hidden_states_flat)
        
        # Handle multiple experts per token
        expert_indices = selected_experts.flatten()
        
        # Expand hidden states for multiple experts
        hidden_states_expanded = hidden_states_flat.repeat_interleave(
            self.num_experts_per_tok, dim=0
        )
        
        if self.enable_expert_parallel:
            return self._forward_expert_parallel(
                hidden_states_expanded,
                expert_indices,
                routing_weights,
                batch_size,
                seq_len,
                num_tokens
            )
        else:
            return self._forward_standard(
                hidden_states_expanded,
                expert_indices,
                routing_weights,
                batch_size,
                seq_len,
                num_tokens
            )
    
    def _forward_standard(
        self,
        hidden_states_expanded: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_tokens: int
    ) -> torch.Tensor:
        """Standard forward pass without expert parallelism."""
        
        # Token permutation
        permuted_hidden_states, expert_counts, token_mapping = token_permutation_v2(
            hidden_states_expanded, 
            expert_indices, 
            self.num_experts, 
            self.num_experts_per_tok,
            is_forward=True
        )
        
        if permuted_hidden_states.numel() == 0:
            return torch.zeros(batch_size, seq_len, self.hidden_size, 
                             device=hidden_states_expanded.device)
        
        # Expert computation
        intermediate_size = self.w1_stacked.shape[2] // 2  # gate + up projection
        intermediate_result = torch.empty(
            (permuted_hidden_states.shape[0], intermediate_size),
            device=permuted_hidden_states.device,
            dtype=permuted_hidden_states.dtype
        )
        
        # Up projection
        intermediate_act = invoke_segmented_gemm_v2(
            permuted_hidden_states,
            self.w1_stacked,
            intermediate_result,
            expert_counts,
            expert_indices
        )
        
        # Apply activation
        intermediate_act = self.act_fn(intermediate_act)
        
        # Down projection
        final_result = torch.empty(
            (intermediate_act.shape[0], self.hidden_size),
            device=intermediate_act.device,
            dtype=intermediate_act.dtype
        )
        
        permuted_output = invoke_segmented_gemm_v2(
            intermediate_act,
            self.w2_stacked,
            final_result,
            expert_counts,
            expert_indices
        )
        
        # Apply routing weights
        routing_weights_flat = routing_weights.flatten()
        _, sorted_indices = torch.sort(expert_indices)
        sorted_routing_weights = routing_weights_flat[sorted_indices]
        permuted_output = permuted_output * sorted_routing_weights.unsqueeze(-1)
        
        # Inverse permutation
        inverse_permuted_output, _ = token_permutation_v2(
            permuted_output,
            expert_indices,
            self.num_experts,
            self.num_experts_per_tok,
            is_forward=False
        )
        
        # Sum across experts for each token
        final_hidden_states = inverse_permuted_output.view(
            num_tokens, self.num_experts_per_tok, self.hidden_size
        ).sum(dim=1)
        
        return final_hidden_states.view(batch_size, seq_len, self.hidden_size)
    
    def _forward_expert_parallel(
        self,
        hidden_states_expanded: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_tokens: int
    ) -> torch.Tensor:
        """Forward pass with expert parallelism."""
        
        # Use expert parallel engine
        result = self.expert_parallel_engine.process_with_expert_parallelism(
            hidden_states_expanded,
            expert_indices,
            routing_weights.flatten(),
            self.w1_stacked,
            self.w2_stacked
        )
        
        # Reshape to final output
        return result.view(batch_size, seq_len, self.hidden_size)
    
    def get_expert_stats(self) -> Dict[str, Any]:
        """Get expert statistics."""
        if hasattr(self, 'expert_parallel_engine'):
            return self.expert_parallel_engine.get_stats()
        return {"expert_count": self.num_experts, "parallelism": "disabled"}
    
    def redistribute_experts(self) -> bool:
        """Trigger expert redistribution if needed."""
        if hasattr(self, 'expert_parallel_engine'):
            return self.expert_parallel_engine.manager.redistribute_experts_if_needed()
        return False


class MoEWithExpertParallel(nn.Module):
    """Complete MoE implementation with expert parallelism support."""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        self.moe = EnhancedSparseMoE(
            hidden_size=config['hidden_size'],
            intermediate_size=config['intermediate_size'],
            num_experts=config['num_experts'],
            shared_expert_intermediate_size=config.get('shared_expert_intermediate_size', config['intermediate_size']),
            num_experts_per_tok=config['num_experts_per_tok'],
            num_shared_experts=config.get('num_shared_experts', 0),
            gate_logit_softcapping=config.get('gate_logit_softcapping'),
            hidden_act=config.get('hidden_act', 'silu'),
            enable_expert_parallel=config.get('enable_expert_parallel', False),
            use_triton=config.get('use_triton', True),
            expert_capacity_factor=config.get('expert_capacity_factor', 1.0)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.moe(hidden_states)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MoE statistics."""
        return self.moe.get_expert_stats()