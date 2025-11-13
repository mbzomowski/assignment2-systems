import torch, torch.nn as nn
from torch import Tensor

from jaxtyping import Float, Int
from . import utils
from cse599o_basics.rotaryposembed import RotaryPositionalEmbedding


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model / self.num_heads
        assert self.d_k.is_integer()
        self.d_k = int(self.d_k)

        self.q_proj_weight = nn.Parameter(torch.randn(self.model, self.d_model))
        self.k_proj_weight = nn.Parameter(torch.randn(self.model, self.d_model))
        self.v_proj_weight = nn.Parameter(torch.randn(self.model, self.d_model))
        self.o_proj_weight = nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rot = None

        if self.max_seq_len is not None and self.theta is not None:
            self.rot = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)

    def forward(
        self,
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] = None
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        w_q = in_features @ self.q_proj_weight.mT
        w_k = in_features @ self.k_proj_weight.mT
        w_v = in_features @ self.v_proj_weight.mT
        w_o = self.o_proj_weight

        out = self.multi_head(w_q, w_k, w_v, token_positions) @ w_o.mT
        return out

    def multi_head(self, w_q, w_k, w_v, token_positions=None):
        split_q = torch.tensor_split(w_q, sections=self.num_heads, dim=-1)
        split_k = torch.tensor_split(w_k, sections=self.num_heads, dim=-1)
        split_v = torch.tensor_split(w_v, sections=self.num_heads, dim=-1)

        S = split_q[0].shape[-2]
        mask = ~torch.triu(torch.ones((S, S), dtype=torch.bool, device=w_q.device), diagonal=1)

        out = []
        for i in range(len(split_q)):
            temp_q = split_q[i]
            temp_k = split_k[i]
            if self.rot is not None:
                temp_q = self.rot(split_q[i], token_positions)
                temp_k = self.rot(split_k[i], token_positions)
            x = utils.scaled_dot_product_attention(temp_q, temp_k, split_v[i], mask)
            out.append(x)

        # Concatenate heads along feature dimension
        return torch.cat(out, dim=-1)
