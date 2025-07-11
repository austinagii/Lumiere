from dataclasses import dataclass
from time import time
from typing import Iterable

import torch
from torch.nn import functional as F

from lumiere.preprocessing.tokenizer import Tokenizer
from lumiere.training.utils import ContextBatchManager


@dataclass
class EvaluationState:
    avg_loss: float
    avg_perplexity: float
    num_batches: int
    time_taken: float


def evaluate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    data: Iterable[str],
    batch_manager: ContextBatchManager,
    device: torch.device,
) -> EvaluationState:
    """Evaluates the model on the dataset"""
    # Evaluate performance on validation set.
    total_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    tokenized_text = tokenizer.tokenize_all(data, lazy=True)
    validation_batches = batch_manager.to_batches(tokenized_text)

    model.eval()
    start_time = time()
    with torch.no_grad():
        for validation_batch in validation_batches:
            tokenized_batch, mask = validation_batch
            encoded_batch = tokenizer.encode_all(tokenized_batch)
            tokenized_batch = torch.tensor(encoded_batch, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)

            x, y = (
                tokenized_batch[:, :-1].to(device),
                tokenized_batch[:, 1:].to(device),
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
