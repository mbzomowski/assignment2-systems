import torch
import math

from collections.abc import Callable
from typing import Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        self.keyword_args = kwargs
        super().__init__(params, kwargs)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m = state['m']
                v = state['v']
                state['step'] += 1
                t = state['step']
                grad = p.grad.data

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * torch.square(grad)
                state['m'] = m
                state['v'] = v
                lr_t = lr * (math.sqrt(1 - betas[1] ** t)/(1 - betas[0] ** t))

                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data
                state['t'] = t + 1
        return loss
