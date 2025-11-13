import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Float

from cse599o_basics.multiheadselfattention import MultiheadSelfAttention
from cse599o_basics.rmsnorm import RMSNorm
from cse599o_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsn1 = RMSNorm(self.d_model)
        self.mhsa = MultiheadSelfAttention(self.d_model, self.num_heads, self.max_seq_len, self.theta)
        self.rmsn2 = RMSNorm(self.d_model)
        self.swig = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"], token_positions: torch.Tensor) -> Float[Tensor, " batch sequence_length d_model"]:
        temp_x = self.mhsa(self.rmsn1(x), token_positions)
        temp_x += x

        temp_temp_x = self.rmsn2(temp_x)
        temp_temp_x = self.swig(temp_temp_x)

        return temp_x + temp_temp_x
