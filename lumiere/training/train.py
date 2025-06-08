from dataclasses import dataclass
from time import time

from tqdm import tqdm
import torch
from torch.nn import functional as F

from lumiere.utils.data import to_batches
from lumiere.preprocessing.tokenizer import Tokenizer


@dataclass
class TrainingState:
    avg_loss: float
    avg_perplexity: float 
    num_batches: int
    current_lr: float
    time_taken: float


def train(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    dataset: torch.utils.data.Dataset,
    current_epoch: int,
    max_epochs: int,
    batch_size: int,
    context_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    gradient_clip_norm: float,
    device: torch.device
) -> TrainingState:
    """Trains the model on the dataset"""
    total_loss = 0.0
    total_perplexity = 0.0 
    num_batches = 0

    batches = to_batches(tokenizer, dataset, batch_size, context_size+1)
    start_time = time()
    with tqdm(batches, desc=f"Epoch {current_epoch}/{max_epochs}", leave=False) as pbar:
        for batch in pbar:
            # Evaluate the model on the current batch.
            x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
            logits, _ = model(x)
            batch_loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size), y.reshape(-1))
            batch_perplexity = torch.exp(batch_loss)

            # Update the model weights.
            batch_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Calculate running training stats.
            total_loss += batch_loss.item()
            total_perplexity += batch_perplexity.item()
            num_batches += 1

            # Update progress bar.
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'perplexity': f'{batch_perplexity:.4f}',
                'lr': f'{current_lr:.2e}',
                'grad_norm': f'{grad_norm:.2f}'
            })
    end_time = time()
    time_taken = end_time - start_time

    return TrainingState(
        avg_loss=total_loss / num_batches,
        avg_perplexity=total_perplexity / num_batches,
        num_batches=num_batches,
        current_lr=current_lr,
        time_taken=time_taken
    )