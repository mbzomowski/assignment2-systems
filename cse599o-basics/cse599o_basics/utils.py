import torch
import math

from torch import Tensor
import numpy.typing as npt

from jaxtyping import Float, Int
from collections.abc import Iterable


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    m = x - x.amax(dim=dim, keepdim=True)
    a = m.exp()
    b = a.sum(dim=dim, keepdim=True)

    return a / b


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    d_k = Q.shape[-1]
    a = Q @ K.transpose(-2, -1)
    b = math.sqrt(d_k)
    S = a / b

    if mask is not None:
        S = S.masked_fill(~mask, float('-inf'))

    return softmax(S, -1) @ V


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:

    adj_inp = inputs - inputs.amax(dim=-1, keepdim=True)
    a = adj_inp.exp().sum(dim=-1, keepdim=True)
    a = torch.log(a)

    probs = a - adj_inp

    rows = torch.arange(targets.size(0))
    p = probs[rows, targets]
    return p.mean()


def learning_rate_schedule(it: int, max_learning_rate: int, min_learning_rate: int, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    params = [p for p in parameters if p.grad is not None]
    grads = [p.grad for p in params]
    total_norm = math.sqrt(sum([torch.norm(g) ** 2 for g in grads]))
    if total_norm > max_l2_norm:
        for p in params:
            p.grad.mul_(max_l2_norm / (total_norm + eps))


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length
    if max_start <= 0:
        raise ValueError("dataset too small for given context_length")

    starts = torch.randint(0, max_start, (batch_size,), device="cpu")

    offsets = torch.arange(context_length, device="cpu").unsqueeze(0)
    x_idx = starts.unsqueeze(1) + offsets
    y_idx = x_idx + 1

    ds = torch.from_numpy(dataset)

    x = ds[x_idx]
    y = ds[y_idx]

    x = x.to(device)
    y = y.to(device)
    return x, y


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    model_params = model.state_dict()
    opt_params = optimizer.state_dict()
    all_params = {"model": model_params, "optimizer": opt_params, "iteration": iteration}
    torch.save(all_params, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    all_params = torch.load(src)
    model_params = all_params["model"]
    opt_params = all_params["optimizer"]
    iteration = all_params["iteration"]

    model.load_state_dict(model_params)
    optimizer.load_state_dict(opt_params)

    return iteration
