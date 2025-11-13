import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.randn(self.num_embeddings, self.embedding_dim))
        self.weights = nn.init.trunc_normal_(self.weights)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
