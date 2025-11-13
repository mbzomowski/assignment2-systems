import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.randn(self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # RMSNorm
        result = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        result = result * x * self.weights
        return result.to(in_dtype)
