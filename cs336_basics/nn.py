"""Functional ops / pure functions for the CS336 basics assignment.

Put stateless helper functions here (softmax, silu, rmsnorm, cross_entropy, ...).
nn.Module subclasses that carry parameters belong in model.py instead.
"""

from math import sqrt
import torch
from torch import Tensor, tensor
from einops import einsum
from jaxtyping import Bool, Float, Int


def softmax(x, dim) -> Tensor:
    # Subtract the max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # 两种写法，取出targets对应的logits的值
    # z_t = logits[torch.arange(logits.shape[0]), targets]
    z_t = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    )  # (batch,)，把targets变成二维，然后取出的数变成一维
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


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    total_sum = torch.sqrt(sum(torch.sum(grad * grad) for grad in grads))
    if total_sum > max_l2_norm:
        clip = max_l2_norm / total_sum
        for grad in grads:
            grad.mul_(clip)

    return None


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    s = einsum(
        Q,
        K,
        "... queries d_k,... keys d_k->... queries keys",
    )
    s = s / sqrt(Q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    r = softmax(s, -1)
    att = einsum(
        r,
        V,
        "... queries keys,... keys d_v->... queries d_v",
    )
    return att
