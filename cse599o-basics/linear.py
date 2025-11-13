import torch
import torch.nn as nn

import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.device = device
        self.dtype = dtype
        self.weights = nn.Parameter(torch.randn(out_features, in_features))

        mean = 0
        var = 2 / (self.in_f + self.out_f)
        std = math.sqrt(var)
        a = -3 * std
        b = 3 * std

        self.weights = nn.init.trunc_normal_(self.weights, mean, std, a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T
