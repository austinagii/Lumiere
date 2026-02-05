"""AdamW optimizer registration."""

import torch

from lumiere.training.optimizer_loader import optimizer


# Register PyTorch's AdamW optimizer directly
optimizer("adamw")(torch.optim.AdamW)
