"""AdamW optimizer registration."""

import torch

from lumiere.internal.registry import register


register(torch.optim.Optimizer, "adamw", torch.optim.AdamW)
