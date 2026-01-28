import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿着最后一个维度一分为二
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
