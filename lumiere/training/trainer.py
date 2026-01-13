import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import time

import torch
from tqdm import tqdm

from lumiere.data.pipeline import Pipeline
from lumiere.models.transformer import Transformer

from .loss import Loss


logger = logging.getLogger(__name__)


@dataclass
class BatchStatistics:
    loss: Loss
    lr: float | None = None
    grad_norm: float | None = None
    total_steps: int | None = None


@dataclass
class EvalMetrics:
    avg_loss: float
    num_batches: int


@dataclass
class TrainingState:
    """Persistent store of information between epochs."""

    total_time_taken: float = 0.0
    prev_loss: float = float("inf")
    best_loss: float = float("inf")
    best_perplexity: float = float("inf")
    global_step: int = 0
    current_epoch: int = 1
    patience: int = 0
    patience_counter: int = 0
    stopping_threshold: int = 0
    current_lr: int | None = None


class Trainer:
    """Trains transformer models.

    Args:
        model: Transformer model to train.
        data: Iterable over batches of training data, each batch should be a tuple of
            - input_tokens: shape (batch_size, context_size)
            - padding_mask: shape (batch_size, context_size)

    """

    def __init__(
        self,
        model: Transformer,
        pipeline: Pipeline,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], Loss],
        max_epochs: int = -1,
        stopping_threshold: float = 1e-3,
        patience: int = 10,
        gradient_clip_norm: float = 1e-6,
        progress_prefix: Callable | None = None,
        progress_suffix: Callable | None = None,
        device: str | torch.device = "cpu",
    ):
        """Initiallize the trainer.

        Arguments:
            model: The model to be trained.
            data: The data to be trained on.
            loss_fn: The loss function to evaluate model performance.
            max_epochs: The maximum number of epochs for training.
            stopping_threshold: The minimum performance improvement required before
                to register improvement.
            patience: The maximum number of epochs without improvement before stopping
                training.
            gradient_clip_norm: Maximum norm for gradient clipping.
            progress_prefix: The variable part of the progress bar.
            progress_suffix: The title of the progress bar.
            device: Device to run the training on. Defaults to CPU.

        """
        assert model.optimizer and model.scheduler
        self.model = model
        self.pipeline = pipeline
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.stopping_threshold = stopping_threshold
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        self.progress_prefix = progress_prefix
        self.progress_suffix = progress_suffix
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        self._state = TrainingState()

        self.model = model.to(device)  # Extract to function.

    def train(self):
        while self._state.current_epoch <= self.max_epochs:
            self._execute_pre_epoch_hooks()

            self._execute_pre_fit_hooks()
            train_metrics = self._fit()
            self._execute_post_fit_hooks(train_metrics)

            self._execute_pre_eval_hooks()
            eval_metrics = self._eval()
            self._execute_post_eval_hooks(eval_metrics)

            # TODO: Consider updating state before post eval hooks are executed.
            if eval_metrics.avg_loss < self.state.best_loss - self.stopping_threshold:
                self._state.best_loss = eval_metrics.avg_loss
                self._state.patience_counter = 0
                self._execute_improvement_hooks()
            else:
                self._state.patience_counter += 1
                if self._state.patience_counter >= self._state.patience:
                    logger.info(
                        f"Training stopped after {self._state.patience} epochs without improvement."  # noqa: E501
                    )
                    break

            self._execute_post_epoch_hooks()
            self._state.current_epoch += 1

        logger.info(f"Training completed after {self._state.current_epoch} epochs.")
        logger.info("--------------------------------")

    def _fit(self) -> EvalMetrics:
        """Fit the transformer model on the specified data for one epoch."""
        # Prepare the model for training.
        # self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()

        total_loss = 0.0
        num_batches = 0
        epoch_steps = 0
        start_time = time()
        with tqdm(
            self.pipeline.batches(), desc=self._progress_bar_prefix(), leave=False
        ) as pbar:
            for x, padding_mask in pbar:
                # Shift the input tokens to the left by one position to get the targets.
                y = x[:, 1:].to(self.device)
                # Shift x and its padding mask accordingly.
                x = x[:, :-1].to(self.device)
                padding_mask = padding_mask[:, :-1].to(self.device)

                # Process the batch and calculate the loss.
                logits, _ = self.model(x, padding_mask)
                batch_loss = self.loss_fn(y, logits)

                # Update the model weights.
                batch_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

                self.model.optimizer.step()
                self.model.scheduler.step()
                self.model.optimizer.zero_grad()

                # Update the epoch training stats.
                total_loss += batch_loss
                num_batches += 1
                epoch_steps += 1
                self._state.global_step += 1
                current_lr = self.model.scheduler.get_last_lr()[0]

                batch_stats = BatchStatistics(
                    loss=batch_loss,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    total_steps=epoch_steps,
                )

                # Update the progress bar.
                pbar.set_postfix(self._progress_bar_suffix(self.state, batch_stats))

        end_time = time()
        self._state.total_time_taken += end_time - start_time

        metrics = EvalMetrics(
            avg_loss=total_loss / num_batches,
            num_batches=num_batches,
        )

        return metrics

    def _eval(self) -> EvalMetrics:
        """Evaluate the transformer model on the specified data."""
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        with (
            torch.no_grad(),
            tqdm(
                self.pipeline.batches(), desc=self._progress_bar_prefix(), leave=False
            ) as pbar,
        ):
            for x, padding_mask in pbar:
                # Shift the input tokens to the left by one position to get the targets.
                y = x[:, 1:].to(self.device)
                # Shift x and its padding mask accordingly.
                x = x[:, :-1].to(self.device)
                padding_mask = padding_mask[:, :-1].to(self.device)

                # Process the batch and calculate the loss.
                logits, _ = self.model(x, padding_mask)
                batch_loss = self.loss_fn(y, logits)
                # Update the evaluation stats.
                batch_stats = BatchStatistics(loss=batch_loss)

                # Update the progress bar.
                pbar.set_postfix(self._progress_bar_suffix(self._state, batch_stats))

                total_loss += batch_loss
                num_batches += 1

        return EvalMetrics(
            avg_loss=total_loss / num_batches,
            num_batches=num_batches,
        )

    def _execute_pre_fit_hooks(self):
        pass

    def _execute_post_fit_hooks(self, fit_metrics):
        pass

    def _execute_pre_eval_hooks(self):
        pass

    def _execute_post_eval_hooks(self, eval_metrics):
        pass

    def _execute_improvement_hooks(self):
        pass

    def _execute_pre_epoch_hooks(self):
        pass

    def _execute_post_epoch_hooks(self):
        pass

    def _progress_bar_prefix(self):
        return f"Epoch {self._state.current_epoch:>04d}"

    def _progress_bar_suffix(self, train_state, batch_stats):
        stats = {
            "loss": f"{batch_stats.loss:.4f}",
            "perplexity": f"{torch.exp(batch_stats.loss):.4f}",
        }
        if batch_stats.lr is not None:
            stats["lr"] = f"{batch_stats.lr:.2f}"
        if batch_stats.grad_norm is not None:
            stats["grad_norm"] = f"{batch_stats.grad_norm:.2f}"
        if batch_stats.total_steps is not None:
            stats["epoch_steps"] = batch_stats.total_steps
        return stats

    @property
    def state(self) -> TrainingState:
        return self._state
