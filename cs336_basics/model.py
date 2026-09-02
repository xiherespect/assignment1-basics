from __future__ import annotations

import math
from typing import Any

from einops import einsum, rearrange, repeat
import torch
from torch import nn

from cs336_basics.nn import scaled_dot_product_attention


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings, self.embedding_dim, device=device, dtype=dtype
            )
        )
        std = math.sqrt(2 / (self.num_embeddings + self.embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # 初始化？？按层尺寸缩放的随机初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转为fl32，防止平方时溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight.to(torch.float32)
        return result.to(in_dtype)


class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = 64 * math.ceil((8 * d_model / 3) / 64)
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        gate = x1 * torch.sigmoid(x1)
        out = gate * x3
        return self.w2(out)


class myRotaryPositionalEmbedding(nn.Module):
    """旋转位置编码（RoPE）。

    把输入向量的维度两两配对 (x[0],x[1]), (x[2],x[3]), ...
    对位置 i 的 token，第 k 对按角度 theta_{i,k} = i / theta^(2k/d_k) 旋转。
    等价于左乘一个块对角旋转矩阵，但这里用逐元素运算实现，无需构造矩阵。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 每对维度的速度不一样
        idx = torch.arange(d_k // 2, dtype=torch.float32, device=device)
        inv_freq = 1 / theta ** (2 * idx / d_k)
        angles = (
            torch.arange(max_seq_len, dtype=torch.float32, device=device)[:, None]
            * inv_freq[None, :]
        )

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", tensor=angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        x_pairs = rearrange(x, "... seq (pair two)->... seq pair two", two=2)
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)

        rope = torch.stack((cos, sin, -sin, cos), dim=-1)
        rope = rearrange(rope, "... seq pair (i o)->... seq pair i o ", i=2, o=2)

        # 注意：torch.einsum 只认单字母下标，"pair" 会被当成 p/a/i/r 四个轴！
        # 多字母轴名要用 einops.einsum（文件顶部已导入），且 seq 轴必须写进模式
        result = einsum(
            x_pairs, rope, "... seq pair i, ... seq pair i o -> ... seq pair o"
        )
        return rearrange(result, "... pair o->... (pair o)")


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力（可选 RoPE）。

    MultiHeadSelfAttention(x) = W_O · Concat(head_1, ..., head_h)
    其中 Q = W_Q x, K = W_K x, V = W_V x，
    head_i = Attention(Q_i, K_i, V_i)，
    Q_i/K_i/V_i 是把投影输出沿 d_model 维切成 h 份后的第 i 份。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 四个投影都是无 bias 的线性层，权重形状 (d_model, d_model)。
        # 注意 W_O（o_proj）：把 h 个头拼起来的结果混合回 d_model 维。
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE 作用在“每个头”的 d_head 维度上，所以这里用 d_head 而不是 d_model
        self.rope = (
            myRotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        # 1) 三次矩阵乘完成全部头的 Q/K/V 投影
        Q = self.q_proj(x)  # (..., seq, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2) 拆头：(..., seq, d_model) -> (..., h, seq, d_head)
        #    "h d" 的写法保证第 i 个头拿到的是输出维度的第 i 个连续切片
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # 3) RoPE 只作用于 Q 和 K（不作用于 V），且每个头用同一套旋转
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # 位置编号复制给每个头：(..., seq) -> (..., h, seq)，各头用同一套位置
            pos = repeat(token_positions, "... seq -> ... h seq", h=self.num_heads)
            Q = self.rope(Q, pos)
            K = self.rope(K, pos)

        # 4) 因果掩码：位置 i 只能看到 j <= i；形状 (seq, seq) 自动广播
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        out = scaled_dot_product_attention(Q, K, V, causal_mask)

        # 5) 拼回头：(..., h, seq, d_head) -> (..., seq, h*d_head)
        out = rearrange(out, "... h seq d -> ... seq (h d)")

        # 6) 输出投影 W_O：混合各头信息，映射回 d_model
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGlu(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = x + self.attn(self.ln1(x))
        x_2 = x_1 + self.ffn(self.ln2(x_1))
        return x_2


class TransformerLM(nn.Module):
    """完整 Transformer 语言模型：embed → N × block → final norm → lm_head。

    子模块命名与参考权重的 key 一一对应，可直接 load_state_dict：
    token_embeddings / layers.{i} / ln_final / lm_head
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)  # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x)  # N 个 Transformer block
        x = self.ln_final(x)  # 最后一个 RMSNorm
        return self.lm_head(x)  # (batch, seq, vocab_size) logits，不 softmax
