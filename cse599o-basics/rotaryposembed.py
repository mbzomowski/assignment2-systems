import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        # Create on CPU; move/cast in forward to match x
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k // 2, dtype=torch.float32) / (d_k // 2)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        token_positions: (L,) or (B, L) (ints or floats)
        returns: (B, L, D)
        """
        B, L, D = x.shape
        if D != self.d_k:
            raise ValueError(f"Last dim of x ({D}) must equal d_k ({self.d_k})")

        # normalize positions to (B_or_1, L)
        if token_positions.dim() == 1:
            pos = token_positions[None, :]           # (1, L)
        elif token_positions.dim() == 2:
            pos = token_positions                     # (B, L)
        else:
            raise ValueError("token_positions must be (L,) or (B, L)")

        if pos.shape[-1] != L:
            raise ValueError(f"token_positions length {pos.shape[-1]} != L={L}")
        if pos.shape[0] not in (1, B):
            raise ValueError(f"token_positions batch {pos.shape[0]} must be 1 or B={B}")

        # move/cast buffers to match x
        inv = self.inv_freq.to(device=x.device, dtype=x.dtype)          # (D/2,)
        pos = pos.to(device=x.device, dtype=x.dtype)                    # (B_or_1, L)

        # angles: (B_or_1, L, D/2)
        angles = pos.unsqueeze(-1) * inv.view(1, 1, -1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # HALF-SPLIT rotate: split last dim into two halves
        D2 = D // 2
        x1, x2 = x[..., :D2], x[..., D2:]                  # (B, L, D/2) each

        # broadcast cos/sin to full D by concatenation
        cos_full = torch.cat([cos, cos], dim=-1)           # (B_or_1, L, D)
        sin_full = torch.cat([sin, sin], dim=-1)           # (B_or_1, L, D)

        # rotate_half(x) = [-x2, x1]
        rot_half = torch.cat([-x2, x1], dim=-1)

        # apply rotation per position (no token mixing)
        #out = x * cos_full + rot_half * sin_full
        xe, xo = x[..., 0::2], x[..., 1::2]
        xe_rot = xe * cos - xo * sin
        xo_rot = xe * sin + xo * cos
        out = torch.empty_like(x)
        out[..., 0::2] = xe_rot
        out[..., 1::2] = xo_rot

        return out
