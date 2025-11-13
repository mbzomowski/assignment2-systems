import torch.nn as nn
import torch
from torch import Tensor

from jaxtyping import Int, Float

from cse599o_basics.embedding import Embedding
from cse599o_basics.transformerblock import TransformerBlock
from cse599o_basics.linear import Linear
from cse599o_basics.rmsnorm import RMSNorm
import cse599o_basics.utils as utils


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.embed = Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.context_length, self.rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(self.d_model)
        self.linear = Linear(self.d_model, self.vocab_size)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embed(x)
        token_positions = torch.arange(x.shape[-2], device=x.device)
        for layer in self.transformer_layers:
            x = layer(x, token_positions)
        x = self.norm(x)
        x = self.linear(x)

        return x
