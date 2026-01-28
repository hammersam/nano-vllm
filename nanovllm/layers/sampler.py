import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperature(<1 or >1)用于放大或者缩小logits数值差异，使模型输出更加确定或者随机
        # logits: (batch_size, vocab_size), temperatures: (batch_size,)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 使用服从指数分布的随机噪声对probs进行扰动，之后选择数值最大的索引作为采样的token
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
