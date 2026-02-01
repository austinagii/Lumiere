import torch
from torch import nn
from torch.nn import Identity


class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = Identity()
        dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.layers.register_parameter("dummy", dummy_param)

    def forward(self, x):
        return self.layers(x)
