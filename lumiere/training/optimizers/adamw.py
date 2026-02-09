"""AdamW optimizer registration."""

import torch

from lumiere.discover import discover


# Register PyTorch's AdamW optimizer directly
@discover(torch.optim.Optimizer, "adamw")
class AdamW(torch.optim.AdamW):
    """Wrapper for PyTorch's AdamW optimizer."""
    pass
