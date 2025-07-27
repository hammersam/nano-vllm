import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.moe_kernel import token_permutation, invoke_segmented_gemm


class MoEGate(nn.Module):
    """MoE 门控路由器，用于选择激活的专家"""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        gate_logit_softcapping: Optional[float] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_logit_softcapping = gate_logit_softcapping
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor):
        # [batch_size, seq_len, hidden_size] -> [batch_size * seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # 计算门控分数
        gate_logits = self.gate(hidden_states)
        
        # 可选的软封顶
        if self.gate_logit_softcapping is not None:
            gate_logits = gate_logits / self.gate_logit_softcapping
            gate_logits = torch.tanh(gate_logits)
            gate_logits = gate_logits * self.gate_logit_softcapping
        
        # Top-k 选择
        routing_weights = F.softmax(gate_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1
        )
        
        # 归一化权重
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


class MoEExpert(nn.Module):
    """单个 MoE 专家网络"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class SparseMoE(nn.Module):
    """稀疏 MoE 层实现"""
    
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        
        # 路由专家
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size, hidden_act)
            for _ in range(num_experts)
        ])
        
        # 共享专家（如果有）
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                MoEExpert(hidden_size, shared_expert_intermediate_size, hidden_act)
                for _ in range(num_shared_experts)
            ])
        
        # 门控网络
        self.gate = MoEGate(
            hidden_size, 
            num_experts, 
            num_experts_per_tok,
            gate_logit_softcapping
        )

        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

        # Stack expert weights for segmented GEMM
        w1 = [e.gate_up_proj.weight.T for e in self.experts]
        w2 = [e.down_proj.weight.T for e in self.experts]
        
        self.register_buffer("w1_stacked", torch.stack(w1, dim=0))
        self.register_buffer("w2_stacked", torch.stack(w2, dim=0))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        num_tokens = hidden_states_flat.shape[0]

        # k: num_experts_per_token
        # routing_weights, selected_experts: (num_tokens, k)
        routing_weights, selected_experts = self.gate(hidden_states_flat)

        expert_indices = selected_experts.flatten()
        assert len(expert_indices.shape) == 1
        assert expert_indices.shape[0] == num_tokens * self.num_experts_per_tok
        # (num_tokens * k, hidden_size)
        # 每一个token都得有相同的k份，发给选中的k个experts
        hidden_states_expanded = hidden_states_flat.repeat_interleave(self.num_experts_per_tok, dim=0)
        assert hidden_states_expanded.shape[0] == num_tokens * self.num_experts_per_tok

        # 将所有token按照被发送给的expert的id进行排序，使得所有被
        # 发送给同一个expert的token在内存中是连续排列的，方便之后的计算
        permuted_hidden_states, expert_counts = token_permutation(
            hidden_states_expanded, expert_indices, self.num_experts, is_forward=True
        )
        assert permuted_hidden_states.shape[0] == num_tokens * self.num_experts_per_tok
        assert expert_counts.shape[0] == self.num_experts
        assert expert_counts.sum() == num_tokens * self.num_experts_per_tok

        intermediate_shape = (permuted_hidden_states.shape[0], self.w1_stacked.shape[2])
        intermediate_result = torch.empty(intermediate_shape, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # FIXME: The current invoke_segmented_gemm is a placeholder.
        # It requires a fully functional segmented GEMM kernel to be efficient.

        # up-projection
        intermediate_act = invoke_segmented_gemm(
            permuted_hidden_states, self.w1_stacked, intermediate_result, expert_counts
        )

        # apply activation function
        intermediate_act = self.act_fn(intermediate_act)

        output_shape = (intermediate_act.shape[0], self.w2_stacked.shape[2])
        permuted_output = torch.empty(output_shape, device=hidden_states.device, dtype=hidden_states.dtype)

        # down-projection
        permuted_output = invoke_segmented_gemm(
            intermediate_act, self.w2_stacked, permuted_output, expert_counts
        )

        # routing_weights: (num_tokens, k)
        # 每个token对于其选中的各个expert的权重
        routing_weights_flat = routing_weights.flatten()
        # expert_indices: (num_tokens * k,)
        # 每个token选中的各个expert的index
        # 需要知道按照expert id排序之后各个expert在`expert_indices`中的
        # 初始位置来获取权重
        # [0, 2, 3, 1, 2, 3] => [0, 1, 2, 2, 3, 3], [0, 3, 1, 4, 2, 5]
        _, sorted_indices = torch.sort(expert_indices)
        # sorted_routing_weights: (num_tokens * k,)
        sorted_routing_weights = routing_weights_flat[sorted_indices]

        # 每个专家计算出来的结果仍然需要乘以token对专家的路由权重
        # permuted_output: (num_tokens * k, hidden_size), orderd in expert id
        permuted_output *= sorted_routing_weights.unsqueeze(-1)

        # 输入仍然是按照expert进行分组的，需要再次调用token_permutation进行
        # 逆向操作，将结果从expert分组的顺序“散布”回token的原始顺序
        inverse_permuted_output, _ = token_permutation(
            permuted_output, expert_indices, self.num_experts, is_forward=False
        )

        # 将每个token在多个expert处的加权处理结果求和
        final_hidden_states = inverse_permuted_output.view(
            num_tokens, self.num_experts_per_tok, hidden_size
        ).sum(dim=1)

        if hasattr(self, 'shared_experts'):
            # (num_tokens, hidden_size)
            shared_output = torch.zeros_like(hidden_states_flat)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(hidden_states_flat)
            final_hidden_states += shared_output

        return final_hidden_states.view(batch_size, seq_len, hidden_size)