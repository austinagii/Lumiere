from collections.abc import Iterable
from dataclasses import dataclass
from time import time

import torch
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from torch.nn import functional as F
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from lumiere.data.tokenizer import SPECIAL_TOKENS
from lumiere.models.transformer import Transformer
from lumiere.training.state import TrainingState


@dataclass
class EvalMetrics:
    avg_loss: float
    avg_perplexity: float
    num_batches: int


def train(
    state: TrainingState,
    model: Transformer,
    data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    gradient_clip_norm: float,
    device: torch.device = torch.device("cpu"),
    # TODO: Replace two below with custom stat logger.
    wandb_run: Run | None = None,
    wandb_log_interval: int = 50,
) -> EvalMetrics:
    """Train the transformer model on the specified data for one epoch.

    Args:
        model: Transformer model to train.
        data: Iterable over batches of training data, each batch should be a tuple of
            - input_tokens: shape (batch_size, context_size)
            - padding_mask: shape (batch_size, context_size)
        gradient_clip_norm: Maximum norm for gradient clipping.
        global_step: Global step count across all epochs.
        device: Device to run the training on. Defaults to CPU.
        wandb_run: Wandb run for logging. Set to None to disable logging (default: None)
        wandb_log_interval: Log training metrics to wandb every N steps (default: 50).

    Returns:
        EvalMetrics: The training state after the epoch.
    """
    # Prepare the model for training.
    model.to(device)
    model.zero_grad()
    model.train()

    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    epoch_steps = 0

    start_time = time()
    with tqdm(data, desc=f"Epoch {state.current_epoch:>04d}", leave=False) as pbar:
        for x, padding_mask in pbar:
            # Shift the input tokens to the left by one position to get the targets.
            y = x[:, 1:].to(device)
            # Shift x and its padding mask accordingly.
            x = x[:, :-1].to(device)
            padding_mask = padding_mask[:, :-1].to(device)

            # Process the batch and calculate the loss.
            logits, _ = model(x, padding_mask)
            batch_loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                y.reshape(-1),
                ignore_index=SPECIAL_TOKENS["padding"].id,
            )
            batch_perplexity = torch.exp(batch_loss)

            # Update the model weights.
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )

            assert model.optimizer and model.scheduler
            model.optimizer.step()
            model.scheduler.step()
            model.optimizer.zero_grad()

            # Update the epoch training stats.
            total_loss += batch_loss.item()
            total_perplexity += batch_perplexity.item()
            num_batches += 1
            epoch_steps += 1
            state.global_step += 1
            current_lr = model.scheduler.get_last_lr()[0]

            # Update the progress bar.
            pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "perplexity": f"{batch_perplexity:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "grad_norm": f"{grad_norm:.2f}",
                    "epoch_steps": epoch_steps,
                }
            )

            # Log the training stats to wandb.
            if wandb_run is not None and state.global_step % wandb_log_interval == 0:
                with disable_tokenizer_parallelism():
                    wandb_run.log(
                        {
                            "train/loss": batch_loss.item(),
                            "train/perplexity": batch_perplexity.item(),
                            "train/lr": current_lr,
                            "train/grad_norm": grad_norm,
                        }
                    )

    end_time = time()
    state.total_time_taken += end_time - start_time

    metrics = EvalMetrics(
        avg_loss=total_loss / num_batches,
        avg_perplexity=total_perplexity / num_batches,
        num_batches=num_batches,
    )

    return metrics
