"""AdamW optimizer registration."""

import torch

from lumiere.discover import register


register(torch.optim.Optimizer, "adamw", torch.optim.AdamW)
