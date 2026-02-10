"""Cosine annealing learning rate scheduler with warmup."""

import torch

from lumiere.internal.registry import discover


@discover(torch.optim.lr_scheduler.LRScheduler, "cosine-annealing")
class CosineAnnealingScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine annealing scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_epochs: int,
        epoch_steps: int,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to schedule.
            warmup_steps: Number of warmup steps for linear warmup.
            max_epochs: Maximum number of training epochs.
            epoch_steps: Number of steps per epoch.
        """
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.epoch_steps = epoch_steps
        super().__init__(optimizer)

    def get_lr(self):
        """Calculate learning rate for current step."""
        step = self.last_epoch + 1

        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (
                self.max_epochs * self.epoch_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        return [base_lr * scale for base_lr in self.base_lrs]
