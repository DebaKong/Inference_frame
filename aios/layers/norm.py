from __future__ import annotations

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6):
        self.weight = torch.empty(size)
        self.eps = eps # 普通 float，以 "eps" 命名不以 _ 开头...

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim = True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight
        