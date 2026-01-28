import functools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import time

import torch
from torch import nn
from tqdm import tqdm

from lumiere.data.pipeline import Pipeline
from lumiere.data.preprocessing import Preprocessor

from .loss import Loss


logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    avg_loss: float
    num_batches: int
    num_steps: int | None = None
    time_taken: float | None = None


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
    """Trains models.

    Attributes:
        state (TrainingState): The state of training
    """

    def __init__(
        self,
        model: nn.Module,
        pipeline: Pipeline,
        preprocessors: Iterable[Preprocessor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], Loss],
        max_epochs: int = -1,
        stopping_threshold: float = 1e-3,
        patience: int = 10,
        gradient_clip_norm: float = 1e-6,
        device: str | torch.device = "cpu",
        state: TrainingState | None = None,
    ):
        """Initiallize the trainer.

        Arguments:
            model: The model to be trained.
            pipeline: The pipeline that produces the training data.
            preprocessors: The preprocessors to be applied to the training data.
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
            state: The initial state of the trainer.

        """
        assert model.optimizer and model.scheduler
        self.model = model.to(device)
        self.pipeline = pipeline
        self.preprocessors = preprocessors
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.stopping_threshold = stopping_threshold
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        if state is None:
            self.state = TrainingState()

    def train(self) -> TrainingState:
        """Train a model."""
        while self.state.current_epoch <= self.max_epochs:
            self._execute_pre_epoch_hooks()

            self._execute_pre_fit_hooks()
            train_metrics = self._fit()
            self.state.total_time_taken += train_metrics.time_taken
            self.state.global_step += train_metrics.num_steps
            self._execute_post_fit_hooks(train_metrics)

            self._execute_pre_eval_hooks()
            eval_metrics = self._eval()
            self._execute_post_eval_hooks(eval_metrics)

            # TODO: Consider updating state before post eval hooks are executed.
            if eval_metrics.avg_loss < self.state.best_loss - self.stopping_threshold:
                self.state.best_loss = eval_metrics.avg_loss
                self.state.patience_counter = 0
                self._execute_improvement_hooks()
            else:
                self.state.patience_counter += 1
                if self.state.patience_counter >= self.state.patience:
                    logger.info(
                        f"Training stopped after {self.state.patience} epochs without improvement."  # noqa: E501
                    )
                    break

            self._execute_post_epoch_hooks()
            self.state.current_epoch += 1

            logger.info(f"Training completed after {self.state.current_epoch} epochs.")
            logger.info("--------------------------------")

        return self.state

    def _fit(self) -> EvalMetrics:
        """Fit the model for next token prediction.

        Processes the entire data provided across the specified number of batches.
        The models parameters are updated using the gradient of the average batch
        loss.

        Args:
            model: The model to be fit to the data.
            data: The data the model is to be fitted to. An iterable of tuples containing
                a tensor of tokens, and a corresponding padding mask.
            loss_fn: The function to evaluate the model's performance.
            state: The current training state for this model.

        """
        self.model.train()
        self.model.zero_grad()

        total_loss = 0.0
        num_batches = 0
        num_steps = 0
        start_time = time()

        batches = (
            functools.reduce(
                lambda x, f: f(*x),
                self.preprocessors,
                batch,
            )
            for batch in self.pipeline.batches()
        )

        with tqdm(
            batches,
            desc=f"Epoch {self.state.current_epoch:>04d}",
            leave=False,
        ) as pbar:
            for batch in pbar:
                samples, targets = batch
                logits, _ = self.model(*samples)
                batch_loss = self.loss_fn(logits, targets)

                batch_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

                self.model.optimizer.step()
                self.model.scheduler.step()
                self.model.optimizer.zero_grad()

                total_loss += batch_loss
                num_batches += 1
                num_steps += 1
                current_lr = self.model.scheduler.get_last_lr()[0]

                pbar.set_postfix(
                    {
                        "batch_no": f"{num_batches:>05d}",
                        "loss": f"{batch_loss:>07.4f}",
                        "perplexity": f"{torch.exp(batch_loss):>09.4f}",
                        "lr": f"{current_lr:.4f}",
                        "grad_norm": f"{grad_norm:.4f}",
                    }
                )

        end_time = time()

        return EvalMetrics(
            num_steps=num_steps,
            time_taken=end_time - start_time,
            avg_loss=total_loss / num_batches,
            num_batches=num_batches,
        )

    def _eval(self) -> EvalMetrics:
        """Evaluate the transformer model on the specified data."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        batches = (
            functools.reduce(
                lambda x, f: f(*x),
                (preprocessor for preprocessor in self.preprocessors),
                batch,
            )
            for batch in self.pipeline.batches()
        )

        with (
            torch.no_grad(),
            tqdm(
                batches,
                desc=f"Epoch {self.state.current_epoch:>04d}",
                leave=False,
            ) as pbar,
        ):
            for samples, targets in pbar:
                # Process the batch and calculate the loss.
                out, _ = self.model(*samples)
                batch_loss = self.loss_fn(out, targets)

                # Update the progress bar.
                pbar.set_postfix(
                    {
                        "loss": f"{batch_loss:.4f}",
                        "perplexity": f"{torch.exp(batch_loss):.4f}",
                    }
                )

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
