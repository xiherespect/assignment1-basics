"""Functional ops / pure functions for the CS336 basics assignment.

Put stateless helper functions here (softmax, silu, rmsnorm, cross_entropy, ...).
nn.Module subclasses that carry parameters belong in model.py instead.
"""

import torch
from torch import Tensor, tensor


def softmax(x, dim) -> Tensor:
    # Subtract the max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:

    z_t = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (batch,)
    # sum = torch.logsumexp(logits, dim=-1)  # (batch,) + (batch,) → (batch,)

    m = logits.max(dim=-1, keepdim=True).values  # (batch, 1)
    shifted = logits - m  # (batch, num_classes)
    sum_exp = torch.exp(shifted).sum(dim=-1)  # (batch,)

    logsumexp = torch.log(sum_exp) + m.squeeze(-1)  # (batch,)

    loss_per_sample = logsumexp - z_t  # (batch,)
    return loss_per_sample.mean()


def silu(x: Tensor) -> Tensor:
    x = x / (1 + torch.exp(-x))
    return x
