from dataclasses import dataclass
from time import time
from typing import Iterable

import torch
from deepscale.storage.clients.azure_blob_storage_client import (
    disable_tokenizer_parallelism,
)
from torch.nn import functional as F
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from lumiere.research.src.data.tokenizer import SPECIAL_TOKENS
from lumiere.research.src.models.transformer import Transformer


@dataclass
class EvaluationState:
    avg_loss: float
    avg_perplexity: float
    num_batches: int
    time_taken: float


def evaluate(
    model: Transformer,
    data: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
    wandb_run: Run | None = None,
) -> EvaluationState:
    """Evaluate the transformer model on the specified data.

    Args:
        model: Transformer model to evaluate.
        data: Iterable over batches of evaluation data, each batch should be a tuple of
            - input_tokens: shape (batch_size, context_size)
            - padding_mask: shape (batch_size, context_size)
        device: Device to run the evaluation on. Defaults to CPU.
        wandb_run: Wandb run for logging. Set to None to disable logging (default: None)

    Returns:
        EvaluationState: The evaluation state after processing all data.
    """
    # Prepare the model for evaluation.
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    start_time = time()
    with torch.no_grad():
        with tqdm(data, desc="Evaluating", leave=False) as pbar:
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

                # Update the evaluation stats.
                total_loss += batch_loss.item()
                total_perplexity += batch_perplexity.item()
                num_batches += 1

                # Update the progress bar.
                pbar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "perplexity": f"{batch_perplexity:.4f}",
                    }
                )

    end_time = time()
    time_taken = end_time - start_time

    # Log the evaluation results to wandb.
    if wandb_run is not None:
        with disable_tokenizer_parallelism():
            wandb_run.log(
                {
                    "eval/loss": total_loss / num_batches,
                    "eval/perplexity": total_perplexity / num_batches,
                }
            )

    return EvaluationState(
        avg_loss=total_loss / num_batches,
        avg_perplexity=total_perplexity / num_batches,
        num_batches=num_batches,
        time_taken=time_taken,
    )
