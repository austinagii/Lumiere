import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import time

import torch
from torch import nn
from tqdm import tqdm

from lumiere.data import DataLoader
from lumiere.data.pipeline import Pipeline

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

    current_epoch: int = 0
    global_step: int = 0
    current_lr: int | None = None
    total_time_taken: float = 0.0
    prev_loss: float = float("inf")
    best_loss: float = float("inf")
    patience_counter: int = 0


class Trainer:
    """Trains models.

    Attributes:
        state (TrainingState): The state of training
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        pipeline: Pipeline,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], Loss],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_epochs: int | None = None,
        stopping_threshold: float = 1e-3,
        patience: int | None = None,
        gradient_clip_norm: float = 1e-6,
        device: str | torch.device = "cpu",
        state: TrainingState | None = None,
    ):
        """Initiallize the trainer.

        Arguments:
            model: The model to be trained.
            pipeline: The pipeline that produces the training data.
            loss_fn: The loss function to evaluate model performance.
            optimizer: The optimizer for updating model parameters.
            scheduler: The learning rate scheduler.
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
        self.model = model.to(device)
        self.dataloader = dataloader
        self.pipeline = pipeline
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.stopping_threshold = stopping_threshold
        self.patience = patience
        self.gradient_clip_norm = gradient_clip_norm
        if state is None:
            self.state = TrainingState()
        self._hooks: dict[str, list[Callable]] = {
            "pre_epoch": [],
            "post_epoch": [],
            "post_fit": [],
            "post_eval": [],
            "eval_improvement": [],
        }

    def train(self) -> TrainingState:
        """Train a model."""
        while self.max_epochs is None or self.state.current_epoch < self.max_epochs:
            self.state.current_epoch += 1
            self._execute_hooks("pre_epoch")

            train_metrics = self._fit()
            self.state.total_time_taken += train_metrics.time_taken
            self.state.global_step += train_metrics.num_steps
            self._execute_hooks("post_fit", train_metrics)

            eval_metrics = self._eval()
            self.state.prev_loss = eval_metrics.avg_loss
            self._execute_hooks("post_eval", eval_metrics)

            if self.state.prev_loss < self.state.best_loss - self.stopping_threshold:
                self.state.best_loss = self.state.prev_loss
                self.state.patience_counter = 0
                self._execute_hooks("eval_improvement")
            else:
                self.state.patience_counter += 1

            logger.info("--------------------------------")
            self._execute_hooks("post_epoch")

            if (
                self.patience is not None
                and self.state.patience_counter >= self.patience
            ):
                logger.info(
                    f"Training stopped after {self.patience} epochs without "
                    + "improvement."
                )
                break

        logger.info(f"Training completed after {self.state.current_epoch} epochs.")

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

        with tqdm(
            self.pipeline.batches(self.dataloader["train"]),
            desc=f"Epoch {self.state.current_epoch:>04d}",
            leave=False,
        ) as pbar:
            for batch in pbar:
                samples, targets = batch
                # What if samples is just a single tensor rather than a tuple of things?
                if isinstance(samples, tuple):
                    outputs = self.model(*samples)
                else:
                    # Running into issues in flexibility, pipeline is intended to be flexible
                    # for various models and outputs, but it's hardcoded to only work with models
                    # that produce multiple outputs here. What's the expectation? Account for this
                    # in the loss function?
                    outputs = self.model(samples)

                batch_loss = self.loss_fn(outputs, targets)

                batch_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += batch_loss
                num_batches += 1
                num_steps += 1
                # TODO: If no scheduler then we need to pull LR from somewhere else.
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0

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

        with (
            torch.no_grad(),
            tqdm(
                self.pipeline.batches(self.dataloader["validation"]),
                desc=f"Epoch {self.state.current_epoch:>04d}",
                leave=False,
            ) as pbar,
        ):
            for samples, targets in pbar:
                # Process the batch and calculate the loss.
                if isinstance(samples, tuple):
                    outputs = self.model(*samples)
                else:
                    # Running into issues in flexibility, pipeline is intended to be flexible
                    # for various models and outputs, but it's hardcoded to only work with models
                    # that produce multiple outputs here. What's the expectation? Account for this
                    # in the loss function?
                    outputs = self.model(samples)

                batch_loss = self.loss_fn(outputs, targets)

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

    def register_pre_epoch_hook(self, fn: Callable[["Trainer"], None]) -> None:
        """Register a function to execute before processing the data for an epoch."""
        self._hooks["pre_epoch"].append(fn)

    def register_post_epoch_hook(self, fn: Callable[["Trainer"], None]) -> None:
        """Register a function to execute after processing the data for an epoch."""
        self._hooks["post_epoch"].append(fn)

    def register_post_fit_hook(
        self, fn: Callable[["Trainer", EvalMetrics], None]
    ) -> None:
        """Register a function to execute after fitting the model to the data."""
        self._hooks["post_fit"].append(fn)

    def register_post_eval_hook(self, fn: Callable[["Trainer", EvalMetrics], None]):
        """Register a function to execute after evaluating the model on the validation data."""
        self._hooks["post_eval"].append(fn)

    def register_eval_improvement_hook(self, fn: Callable):
        """Register a function to execute if validation performance improves after evaluation."""
        self._hooks["eval_improvement"].append(fn)

    def _execute_hooks(self, event, /, *args):
        hooks = self._hooks[event]
        for hook in hooks:
            hook(self, *args)
