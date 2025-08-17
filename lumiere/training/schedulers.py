import torch


# TODO: Consider using the torch.optim.lr_scheduler.CosineAnnealingLR instead.
def cosine_annealing_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_epochs: int,
    epoch_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing learning rate scheduler."""

    def lr_lambda(step):
        # Add 1 to step to ensure that the initial learning rate is non zero.
        step += 1

        if step <= warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_epochs * epoch_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    # TODO: Confirm if last_epoch should be specified when resuming training from a
    # checkpoint.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
