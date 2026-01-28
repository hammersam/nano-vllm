import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        # vocab_size
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x: (total_num_tokens,)
        if self.tp_size > 1:
            # 筛选出这个gpu负责计算embedding的token            
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 为了能够正确地获取相应的embedding，将全局token id转化为
            # 局部index，并将不属于本gpu的token index置为0
            x = mask * (x - self.vocab_start_idx)
        # y: (total_num_tokens, embedding_dim)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # 仅提取每个序列中最后一个token的hidden state
            # cu_seqlens_q: [0, L1, L1 + L2, L1 + L2 + L3, ...]
            last_indices = context.cu_seqlens_q[1:] - 1
            # (total_num_tokens, hidden_size) -> (batch_size, hidden_size)
            x = x[last_indices].contiguous()
        # (batch_size, vocab_size_per_partition)
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有rank0才需要执行收集和拼接分布在各个gpu中的logits的任务
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 各个gpu只需要把自己计算得到的logits发送给rank0即可，不需要发送给所有gpu
            dist.gather(logits, all_logits, 0)
            # (batch_size, vocab_size)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
