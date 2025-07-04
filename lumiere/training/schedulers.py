import torch


def cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_epochs: int,
    epoch_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing learning rate scheduler."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_epochs * epoch_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
