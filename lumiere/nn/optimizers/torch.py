"""PyTorch optimizer registrations.

This module registers PyTorch optimizers with the Lumiere discovery system,
making them available for use in training configurations.
"""

import torch

from lumiere.internal.registry import register


register(torch.optim.Optimizer, "adamw", torch.optim.AdamW)
