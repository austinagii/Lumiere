from dataclasses import dataclass
from time import time

import torch
from torch.nn import functional as F

from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.utils.data import to_batches


@dataclass
class EvaluationState:
    avg_loss: float
    avg_perplexity: float
    num_batches: int
    time_taken: float


def evaluate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    validation_dataset: torch.utils.data.Dataset,
    batch_size: int,
    context_size: int,
    device: torch.device,
) -> EvaluationState:
    """Evaluates the model on the dataset"""
    # Evaluate performance on validation set.
    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    validation_batches = to_batches(
        tokenizer, validation_dataset, batch_size, context_size + 1
    )

    model.eval()
    start_time = time()
    with torch.no_grad():
        for validation_batch in validation_batches:
            x, y = (
                validation_batch[:, :-1].to(device),
                validation_batch[:, 1:].to(device),
            )
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1)
            )
            total_loss += loss.item()
            total_perplexity += torch.exp(loss).item()
            num_batches += 1
    end_time = time()
    time_taken = end_time - start_time
    model.train()

    eval_state = EvaluationState(
        avg_loss=total_loss / num_batches,
        avg_perplexity=total_perplexity / num_batches,
        num_batches=num_batches,
        time_taken=time_taken,
    )

    return eval_state
