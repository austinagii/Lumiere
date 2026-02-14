"""PyTorch optimizer registrations.

This module registers PyTorch optimizers with the Lumiere discovery system,
making them available for use in training configurations.
"""

from torch import nn

from lumiere.internal.registry import register


register(nn.Module, "normalization.layer", nn.LayerNorm)
register(nn.Module, "normalization.rms", nn.RMSNorm)
