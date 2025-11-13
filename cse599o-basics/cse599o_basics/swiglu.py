import torch.nn as nn
from torch import Tensor

from jaxtyping import Float

from cse599o_basics.linear import Linear
import cse599o_basics.utils as utils


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: Float[Tensor, " ... d_model"]):
        temp = self.w1(x)
        temp = utils.silu(temp)
        temp = temp * self.w3(x)
        temp = self.w2(temp)
        return temp
