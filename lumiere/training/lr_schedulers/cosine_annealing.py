"""Cosine annealing learning rate scheduler with warmup."""

import torch

from lumiere.training.scheduler_loader import scheduler


def cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_epochs: int,
    epoch_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a cosine annealing learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps for linear warmup.
        max_epochs: Maximum number of training epochs.
        epoch_steps: Number of steps per epoch.

    Returns:
        A configured LambdaLR scheduler.

    Example:
        >>> scheduler_config = {
        ...     "name": "cosine-annealing",
        ...     "warmup_steps": 500,
        ...     "max_epochs": 250,
        ...     "epoch_steps": 2000
        ... }
        >>> scheduler = load_scheduler(scheduler_config, optimizer)
    """

    def lr_lambda(step):
        # Add 1 to step to ensure that the initial learning rate is non zero.
        step += 1

        if step <= warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (
                max_epochs * epoch_steps - warmup_steps
            )
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Register the scheduler function
scheduler("cosine-annealing")(cosine_annealing_scheduler)
