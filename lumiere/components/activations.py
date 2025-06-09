from torch import nn
from torch.nn import functional as F

SWIGLU_SCALE_FACTOR = 2.67

class SwiGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = int(dim * SWIGLU_SCALE_FACTOR)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))
