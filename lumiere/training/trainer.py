import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import time

import torch
from torch import nn
from tqdm import tqdm

from lumiere.data import DataLoader, Pipeline

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

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


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
        """Initialize the trainer.

        Args:
            model: The model to be trained.
            dataloader: The dataloader that provides train/validation splits.
            pipeline: The pipeline that processes and batches the training data.
            loss_fn: The loss function to evaluate model performance.
            optimizer: The optimizer for updating model parameters.
            scheduler: Optional learning rate scheduler. Defaults to `None`.
            max_epochs: The maximum number of epochs for training. If `None`, trains
                indefinitely until early stopping criteria are met. Defaults to `None`.
            stopping_threshold: The minimum performance improvement required to
                register improvement. Defaults to `1e-3`.
            patience: The maximum number of epochs without improvement before stopping
                training. If `None`, trains until `max_epochs` is reached. Defaults to `None`.
            gradient_clip_norm: Maximum norm for gradient clipping. Defaults to `1e-6`.
            device: Device to run the training on. Can be `"cpu"`, `"cuda"`, or `"mps"`.
                Defaults to `"cpu"`.
            state: The initial state of the trainer. If `None`, creates a new `TrainingState`.
                Defaults to `None`.
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
        self.state = TrainingState() if state is None else state
        self._hooks: dict[str, list[Callable]] = {
            "pre_epoch": [],
            "post_epoch": [],
            "post_fit": [],
            "post_eval": [],
            "eval_improvement": [],
        }

    def train(self) -> TrainingState:
        """Train the model and execute the training loop.

        Performs iterative training with automatic validation, early stopping, and
        hook execution. Training continues until `max_epochs` is reached or early
        stopping criteria are met (if `patience` is specified).

        Returns:
            Final training state containing metrics and counters.
        """
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
        """Fit the model on the training data for one epoch.

        Processes the entire training dataset, updating model parameters using
        gradient descent with the configured optimizer and optional learning rate
        scheduler. Applies gradient clipping to prevent gradient explosion.

        Returns:
            Training metrics including average loss, number of batches processed,
            number of steps taken, and time elapsed.
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
        """Evaluate the model on the validation data.

        Runs the model in evaluation mode (no gradient computation) on the validation
        split to compute loss metrics.

        Returns:
            Evaluation metrics including average loss and number of batches processed.
        """
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
        """Register a function to execute before each epoch begins.

        Args:
            fn: Callback function that receives the trainer instance.
        """
        self._hooks["pre_epoch"].append(fn)

    def register_post_epoch_hook(self, fn: Callable[["Trainer"], None]) -> None:
        """Register a function to execute after each epoch completes.

        Args:
            fn: Callback function that receives the trainer instance.
        """
        self._hooks["post_epoch"].append(fn)

    def register_post_fit_hook(
        self, fn: Callable[["Trainer", EvalMetrics], None]
    ) -> None:
        """Register a function to execute after training on each epoch.

        Args:
            fn: Callback function that receives the trainer instance and training metrics.
        """
        self._hooks["post_fit"].append(fn)

    def register_post_eval_hook(self, fn: Callable[["Trainer", EvalMetrics], None]):
        """Register a function to execute after validation on each epoch.

        Args:
            fn: Callback function that receives the trainer instance and evaluation metrics.
        """
        self._hooks["post_eval"].append(fn)

    def register_eval_improvement_hook(self, fn: Callable):
        """Register a function to execute when validation loss improves.

        Args:
            fn: Callback function that receives the trainer instance.
        """
        self._hooks["eval_improvement"].append(fn)

    def _execute_hooks(self, event, /, *args):
        hooks = self._hooks[event]
        for hook in hooks:
            hook(self, *args)
