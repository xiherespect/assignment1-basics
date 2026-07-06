import torch
from torch import nn

# # d_model: int,
#     eps: float,
#     weights: Float[Tensor, " d_model"],
#     in_features: Float[Tensor, " ... d_model"],
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = 
